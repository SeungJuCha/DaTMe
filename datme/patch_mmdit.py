"""Inject DaTMe image-token matching into FLUX MMDiT blocks."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch

from . import merge_flux
from .utils import do_nothing, init_generator, isinstance_str

# ── RoPE slicer ───────────────────────────────────────────────────────────────
def _slice_rope(
    image_rotary_emb: Optional[torch.Tensor],
    N_txt: int,
    kept_idx: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Slice image_rotary_emb to match merged image token count for FLUX.
    image_rotary_emb: (B, 1, N_txt + N_img, ...)
    kept_idx: (N_img',) - flat indices into the image portion
    """
    if image_rotary_emb is None or kept_idx is None:
        return image_rotary_emb
    rope_txt = image_rotary_emb[:, :, :N_txt]           # (B, 1, N_txt, ...)
    rope_img = image_rotary_emb[:, :, N_txt:]           # (B, 1, N_img, ...)
    rope_img_m = rope_img[:, :, kept_idx]               # (B, 1, N_img', ...)
    return torch.cat([rope_txt, rope_img_m], dim=2)


def _kept_idx_from_merge_indices(
    indices: Optional[Dict[str, Any]],
    N: int,
    device: torch.device,
) -> torch.Tensor:
    """Return image RoPE order matching DaTMe merge output: [indep, dst]."""
    if not indices:
        return torch.arange(N, device=device)

    indep_idx = indices.get("indep_idx")
    b_idx = indices.get("b_idx")
    if indep_idx is None or b_idx is None:
        return torch.arange(N, device=device)

    return torch.cat([indep_idx[0, :, 0], b_idx[0, :, 0]], dim=0).to(device=device, dtype=torch.long)


# ── compute_merge_mmdit ────────────────────────────────────────────────────────
def compute_merge_mmdit(
    img_tokens: torch.Tensor,
    datme_info: Dict[str, Any],
) -> Tuple[Callable, Callable, Optional[torch.Tensor]]:
    """Build the fixed DaTMe merge/unmerge functions for FLUX image tokens."""
    args = datme_info["args"]
    width = datme_info.get("img_w")
    height = datme_info.get("img_h")
    token_count = img_tokens.shape[1]
    if width is None or height is None or args.get("merge_mode") == "none":
        kept = torch.arange(token_count, device=img_tokens.device)
        return do_nothing, do_nothing, kept

    remove_count = int(token_count * args.get("ratio", 0.0))
    if remove_count <= 0 or args.get("signal_maps") is None:
        kept = torch.arange(token_count, device=img_tokens.device)
        return do_nothing, do_nothing, kept

    if args["generator"] is None:
        args["generator"] = init_generator(img_tokens.device)
    elif args["generator"].device != img_tokens.device:
        args["generator"] = init_generator(
            img_tokens.device, fallback=args["generator"]
        )

    batch_size = img_tokens.shape[0]
    cache_key = (height, width, img_tokens.dtype, str(img_tokens.device))
    if datme_info.get("_ijs_cache_key") != cache_key:
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij"),
            dim=-1,
        ).reshape(1, -1, 2)
        datme_info["_ijs_cached"] = coordinates.to(
            dtype=img_tokens.dtype, device=img_tokens.device
        )
        datme_info["_ijs_cache_key"] = cache_key
    metric = datme_info["_ijs_cached"].expand(batch_size, -1, -1)

    step = args.get("step", -1)
    index_cache = datme_info.setdefault("_index_cache", {})
    merge_cache_key = (step, height, width, token_count)
    cached = index_cache.get(merge_cache_key)
    cache_store = {} if cached is None else None
    merge_fn, unmerge_fn = merge_flux.matching_datme(
        metric,
        width,
        height,
        remove_count,
        _cached_indices=cached,
        _index_cache_store=cache_store,
        **args,
    )
    if cache_store:
        for stale_key in [key for key in index_cache if key[0] != step]:
            del index_cache[stale_key]
        index_cache[merge_cache_key] = cache_store
    indices = cache_store or cached
    kept_idx = _kept_idx_from_merge_indices(indices, token_count, img_tokens.device)
    return merge_fn, unmerge_fn, kept_idx


# ── FLUX Double-Stream ────────────────────────────────────────────────────────
def make_datme_block_flux_double(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class DaTMeFluxDoubleBlock(block_class):
        _parent = block_class

        def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None):
            layer_current = self._datme_info["states"]["double_layer_current"]
            self._datme_info["states"]["double_layer_current"] += 1

            double_blocks = self._datme_info["args"].get("double_layer_indices", [])

            if layer_current not in double_blocks:
                return super().forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb)

            residual_hs = hidden_states

            norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            norm_enc, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)

            m, u, kept_idx = compute_merge_mmdit(norm_hs, self._datme_info)
            norm_hs_m = m(norm_hs, mode="mean")
            N_txt = encoder_hidden_states.shape[1]
            rope_m = _slice_rope(image_rotary_emb, N_txt, kept_idx)

            attn_output_m, context_attn_output = self.attn(
                hidden_states=norm_hs_m,
                encoder_hidden_states=norm_enc,
                image_rotary_emb=rope_m,
            )

            attn_output = u(attn_output_m)
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = residual_hs + attn_output

            residual_hs2 = hidden_states
            norm_hs2 = self.norm2(hidden_states)
            norm_hs2 = norm_hs2 * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hs2)

            ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = residual_hs2 + ff_output

            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_enc2 = self.norm2_context(encoder_hidden_states)
            norm_enc2 = norm_enc2 * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_enc2)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

            return encoder_hidden_states, hidden_states
    return DaTMeFluxDoubleBlock


# ── FLUX Single-Stream ────────────────────────────────────────────────────────
def make_datme_block_flux_single(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class DaTMeFluxSingleBlock(block_class):
        _parent = block_class

        def forward(self, hidden_states, temb, image_rotary_emb=None):
            layer_current = self._datme_info["states"]["single_layer_current"]
            self._datme_info["states"]["single_layer_current"] += 1

            single_blocks = self._datme_info["args"].get("single_layer_indices", [])

            if layer_current not in single_blocks:
                return super().forward(hidden_states, temb, image_rotary_emb)

            residual = hidden_states
            norm_hs, gate = self.norm(hidden_states, emb=temb)

            N_txt = self._datme_info.get("N_txt", 0)
            txt_norm = norm_hs[:, :N_txt, :]
            img_norm = norm_hs[:, N_txt:, :]

            m, u, kept_idx = compute_merge_mmdit(img_norm, self._datme_info)
            img_norm_m = m(img_norm, mode="mean")
            rope_m = _slice_rope(image_rotary_emb, N_txt, kept_idx)

            norm_hs_attn = torch.cat([txt_norm, img_norm_m], dim=1)

            mlp_hidden = self.act_mlp(self.proj_mlp(norm_hs))
            attn_out_m = self.attn(hidden_states=norm_hs_attn, image_rotary_emb=rope_m)

            txt_attn = attn_out_m[:, :N_txt, :]
            img_attn_m = attn_out_m[:, N_txt:, :]
            img_attn = u(img_attn_m)
            attn_out = torch.cat([txt_attn, img_attn], dim=1)

            hs_cat = torch.cat([attn_out, mlp_hidden], dim=2)
            hs_out_full = self.proj_out(hs_cat)
            hs_out_full = gate.unsqueeze(1) * hs_out_full
            
            hidden_states = residual + hs_out_full

            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            return hidden_states
    return DaTMeFluxSingleBlock


# ── Patch Application ────────────────────────────────────────────────────────
def apply_patch(
    model: torch.nn.Module,
    ratio: float = 0.5,
    use_rand: bool = True,
    signal_maps=None,
    merge_mode: str = "datme",
    double_layer_indices: Union[List[int], str] = "all",
    single_layer_indices: Union[List[int], str] = "none",
    **kwargs,
) -> torch.nn.Module:
    remove_patch(model)

    diffusion_model = model.transformer if hasattr(model, "transformer") else model

    def normalize_layers(value: Union[List[int], str], count: int) -> List[int]:
        if isinstance(value, list):
            return value
        if value == "all":
            return list(range(count))
        if value == "none":
            return []
        raise ValueError(f"Layer selection must be 'all', 'none', or a list, got {value!r}.")

    double_count = len(getattr(diffusion_model, "transformer_blocks", []))
    single_count = len(getattr(diffusion_model, "single_transformer_blocks", []))

    diffusion_model._datme_info = {
        "img_w": None,
        "img_h": None,
        "N_txt": 0,
        "states": {"double_layer_current": 0, "single_layer_current": 0},
        "args": {
            "ratio": ratio,
            "use_rand": use_rand,
            "generator": None,
            "signal_maps": signal_maps,
            "merge_mode": merge_mode,
            "double_layer_indices": normalize_layers(double_layer_indices, double_count),
            "single_layer_indices": normalize_layers(single_layer_indices, single_count),
        },
    }
    diffusion_model._datme_info["args"].update(kwargs)

    # Patch FLUX double-stream blocks
    if hasattr(diffusion_model, "transformer_blocks"):
        for block in diffusion_model.transformer_blocks:
            if isinstance_str(block, "FluxTransformerBlock"):
                block.__class__ = make_datme_block_flux_double(block.__class__)
            block._datme_info = diffusion_model._datme_info

    # Patch FLUX single-stream blocks
    if hasattr(diffusion_model, "single_transformer_blocks"):
        for block in diffusion_model.single_transformer_blocks:
            if isinstance_str(block, "FluxSingleTransformerBlock"):
                block.__class__ = make_datme_block_flux_single(block.__class__)
            block._datme_info = diffusion_model._datme_info

    # Hook for resetting layer count per step (model forward)
    def forward_pre_hook(module, args, kwargs):
        module._datme_info["states"]["double_layer_current"] = 0
        module._datme_info["states"]["single_layer_current"] = 0
        return None

    diffusion_model._datme_info["hook"] = diffusion_model.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)

    return model


def update_patch(model: torch.nn.Module, **kwargs):
    diffusion_model = model.transformer if hasattr(model, "transformer") else model
    if hasattr(diffusion_model, "_datme_info"):
        diffusion_model._datme_info["args"].update(kwargs)


def remove_patch(model: torch.nn.Module) -> torch.nn.Module:
    diffusion_model = model.transformer if hasattr(model, "transformer") else model

    for getattr_name in ("transformer_blocks", "single_transformer_blocks"):
        if hasattr(diffusion_model, getattr_name):
            blocks = getattr(diffusion_model, getattr_name)
            for block in blocks:
                if block.__class__.__name__ in ("DaTMeFluxDoubleBlock", "DaTMeFluxSingleBlock"):
                    block.__class__ = block._parent
                if hasattr(block, "_datme_info"):
                    del block._datme_info

    if hasattr(diffusion_model, "_datme_info"):
        if "hook" in diffusion_model._datme_info:
            diffusion_model._datme_info["hook"].remove()
        del diffusion_model._datme_info

    return model
