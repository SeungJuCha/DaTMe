"""Inject DaTMe matching into PixArt transformer blocks."""

import math
from typing import Type, Dict, Any, Tuple, Callable, Optional

import torch

from . import merge_pixart_base as merge
from . import merge_pixart
from .utils import isinstance_str, init_generator


def compute_merge(x: torch.Tensor, datme_info: Dict[str, Any], ratio: float = 0) -> Tuple[Callable, ...]:
    """Create merge functions from the current step's DaTMe signals."""
    original_h, original_w = datme_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = datme_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        if ratio > 0:
            r = int(x.shape[1] * ratio)
        else:
            r = int(x.shape[1] * args["ratio"])

        # Keep the random generator on the same device as the current block.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        # Odd batches cannot pair conditional and unconditional samples.
        use_rand = False if x.shape[0] % 2 == 1 else args.get("use_rand", True)

        original_tome = args["original_tome"]
        feature_based = args.get("feature_based", False)
        if not original_tome and not feature_based:
            # Use cached 2D coordinates for locality-aware matching.
            hh, ww = original_h // 2, original_w // 2
            # Reuse the coordinate grid across blocks and denoising steps.
            _cache_key = (hh, ww, x.dtype, str(x.device))
            if datme_info.get("_ijs_cache_key") != _cache_key:
                _ijs = torch.stack(
                    torch.meshgrid(torch.arange(hh), torch.arange(ww), indexing="ij"),
                    dim=-1,
                )
                _ijs = _ijs.reshape(-1, 2).unsqueeze(0).to(dtype=x.dtype, device=x.device)
                datme_info["_ijs_cached"] = _ijs
                datme_info["_ijs_cache_key"] = _cache_key
            x = datme_info["_ijs_cached"].expand(x.shape[0], -1, -1)  # B, h*w, 2

        signal_maps = args.get("signal_maps", None)
        merge_mode = args.get("merge_mode", "datme")

        if merge_mode == "datme" and signal_maps is not None:
            m, u = merge_pixart.matching_datme_pixart(
                x, w, h, r,
                add_rand_small=use_rand, **args
            )
        elif signal_maps is not None:
            m, u = merge.matching_by_heat_map_dit(
                x, w, h, r, generator=args["generator"],
                error_mask=signal_maps, pp=1, original_tome=original_tome
            )
        else:
            m, u = merge.bipartite_soft_matching_random2d_dit(
                x, w, h, 
                args.get("sx", 2), args.get("sy", 2), r, no_rand=not use_rand, generator=args["generator"], original_tome=original_tome
            )
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args.get("merge_attn", True) else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args.get("merge_crossattn", True) else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args.get("merge_mlp", False) else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m


def make_diffusers_datme_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """Wrap a PixArt transformer block with DaTMe merge/unmerge calls."""
    class DaTMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:
            # Build merge/unmerge functions from this denoising step's signals.
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._datme_info)

            batch_size = hidden_states.shape[0]

            # normalization
            if True:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm1(hidden_states, timestep)
                elif self.norm_type == "ada_norm_zero":
                    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                        hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                    )
                elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm1(hidden_states)
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif self.norm_type == "ada_norm_single":
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                    ).chunk(6, dim=1)
                    norm_hidden_states = self.norm1(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                else:
                    raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            # Merge image tokens before self-attention.
            reduce_mode = self._datme_info["args"].get("merge_mode", "datme")
            if reduce_mode == "datme": reduce_mode = "mean"
            norm_hidden_states = m_a(norm_hidden_states, mode=reduce_mode)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output

            # Restore the full token sequence after self-attention.
            hidden_states = u_a(attn_output) + hidden_states
            
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 1.2 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # 3. Cross-Attention
            if self.attn2 is not None:
                if True:
                    if self.norm_type == "ada_norm":
                        norm_hidden_states = self.norm2(hidden_states, timestep)
                    elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                        norm_hidden_states = self.norm2(hidden_states)
                    elif self.norm_type == "ada_norm_single":
                        norm_hidden_states = hidden_states
                    elif self.norm_type == "ada_norm_continuous":
                        norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                    else:
                        raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                # Optionally merge image tokens before cross-attention.
                norm_hidden_states = m_c(norm_hidden_states, mode=reduce_mode)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                
                # Restore the full token sequence after cross-attention.
                hidden_states = u_c(attn_output) + hidden_states

            # 4. Feed-forward
            if True:
                if self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif not self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm3(hidden_states)

                if self.norm_type == "ada_norm_zero":
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                if self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm2(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            # Optionally merge image tokens before the feed-forward layer.
            norm_hidden_states = m_m(norm_hidden_states, mode=reduce_mode)

            ff_output = self.ff(norm_hidden_states)

            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output

            # Restore the full token sequence after the feed-forward layer.
            hidden_states = u_m(ff_output) + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states

    return DaTMeBlock


def hook_datme_model(model: torch.nn.Module):
    """Capture the latent image size used to align the two signal maps."""
    def hook(module, args):
        module._datme_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._datme_info["hooks"].append(model.register_forward_pre_hook(hook))


def update_patch(model: torch.nn.Module, **kwargs):
    """Update shared DaTMe arguments without re-patching transformer blocks.
    
    Use this inside the denoising loop instead of calling apply_patch every step.
    Only updates the shared args dict that all patched blocks reference.
    """
    diffusion_model = model.transformer if hasattr(model, "transformer") else model
    if not hasattr(diffusion_model, "_datme_info"):
        return  # not patched yet
    args = diffusion_model._datme_info["args"]
    args.update(kwargs)

def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 2,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = True,
        merge_mlp: bool = False, 
        signal_maps = None,
        merge_mode: str = "datme",
        original_tome: bool = False,
        merge_layer_start: int = 7, # Default to start from beginning
        merge_layer_end: int = 21,  # Default to apply till end (PixArt has 28 blocks)
        **kwargs):
    """
    Patch PixArt transformer blocks with DaTMe token merging.

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
     - merge_layer_start: The index of the first transformer block to apply merging to.
     - merge_layer_end: The index of the last transformer block (exclusive) to apply merging to.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply DaTMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    assert is_diffusers

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.transformer" and "transformer"
        diffusion_model = model.transformer if hasattr(model, "transformer") else model

    diffusion_model._datme_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp, 
            "signal_maps": signal_maps, 
            "merge_mode": merge_mode, 
            "original_tome": original_tome,
        }
    }
    diffusion_model._datme_info["args"].update(kwargs)
    hook_datme_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            block_ii = int(_.split('.')[1])

            if block_ii not in range(merge_layer_start, merge_layer_end): continue
            
            assert is_diffusers
            make_datme_block_fn = make_diffusers_datme_block
            module.__class__ = make_datme_block_fn(module.__class__)
            module._datme_info = diffusion_model._datme_info

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model


def remove_patch(model: torch.nn.Module):
    """Remove DaTMe hooks and restore the original transformer block classes."""
    # For diffusers
    model = model.transformer if hasattr(model, "transformer") else model

    for _, module in model.named_modules():
        if hasattr(module, "_datme_info"):
            for hook in module._datme_info["hooks"]:
                hook.remove()
            module._datme_info["hooks"].clear()

        if module.__class__.__name__ == "DaTMeBlock":
            module.__class__ = module._parent
    
    return model
