"""Inject DaTMe matching into SD and SDXL transformer blocks.

Pipelines provide the temporal and spatial signal maps for each denoising step;
this module forwards them to ``merge_unet_datme`` and applies the resulting
merge/unmerge functions around attention.
"""

import math
from typing import Any, Callable, Dict, List, Tuple, Type

import torch

from . import merge_unet, merge_unet_datme
from .utils import isinstance_str, init_generator


def compute_merge(x: torch.Tensor, datme_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    """Create merge functions from the current step's DaTMe signals."""
    original_h, original_w = datme_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens / x.shape[1])))

    args = datme_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])

        # Keep the random generator on the same device as the current block.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        # Odd batches cannot pair conditional and unconditional samples.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]

        if args["merge_mode"] == "datme":
            # Temporal/spatial signals select the same SRC/DST/IND indices for
            # every block in one denoising step, so compute them only once.
            step = args.get("step", -1)
            index_cache = datme_info.setdefault("_index_cache", {})
            cached = index_cache.get(step)
            cache_store = {} if cached is None else None

            m, u = merge_unet_datme.matching_datme_unet(
                x,
                w,
                h,
                r,
                _cached_indices=cached,
                _index_cache_store=cache_store,
                **args,
            )

            if cache_store:
                index_cache.clear()
                index_cache[step] = cache_store
        elif args["signal_maps"] is not None:
            m, u = merge_unet.matching_by_heat_map(
                x, w, h, r, error_mask=args["signal_maps"], **args
            )
        else:
            m, u = merge_unet.bipartite_soft_matching_random2d(
                x, w, h, r=r, no_rand=not use_rand, **args
            )
    else:
        m, u = (merge_unet.do_nothing, merge_unet.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge_unet.do_nothing, merge_unet.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge_unet.do_nothing, merge_unet.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge_unet.do_nothing, merge_unet.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m


def make_diffusers_datme_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """Wrap a diffusers transformer block with DaTMe merge/unmerge calls."""
    class DaTMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # Build merge/unmerge functions from this denoising step's signals.
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._datme_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # Merge image tokens before self-attention.
            reduce_mode = self._datme_info["args"]["merge_mode"]
            if reduce_mode.startswith("mine") or reduce_mode == "datme":
                reduce_mode = "mean"
            norm_hidden_states = m_a(norm_hidden_states, mode=reduce_mode)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # Restore the full token sequence after self-attention.
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # Optionally merge image tokens before cross-attention.
                # Reuse reduce_mode calculated earlier
                norm_hidden_states = m_c(norm_hidden_states, mode=reduce_mode)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # Restore the full token sequence after cross-attention.
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # Optionally merge image tokens before the feed-forward layer.
            # Reuse reduce_mode calculated earlier
            norm_hidden_states = m_m(norm_hidden_states, mode=reduce_mode)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # Restore the full token sequence after the feed-forward layer.
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return DaTMeBlock


def hook_datme_model(model: torch.nn.Module):
    """Capture the latent image size used to align the two signal maps."""
    def hook(module, args):
        module._datme_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._datme_info["hooks"].append(model.register_forward_pre_hook(hook))


def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,
        signal_maps = None,
        merge_mode: str = "datme",
        step: int = -1,
        merge_layer_start: int = None,  # None = no lower bound (all layers)
        merge_layer_end: int = None,    # None = no upper bound (all layers)
        merge_layers: List[int] = None, # List of specific indices to patch
        **kwargs): # Accept extra kwargs
    """Patch SD or SDXL transformer blocks with DaTMe token merging."""

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
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model
    
    # Store args
    args_dict = {
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
            "step": step,
    }
    # Update with kwargs (hyperparameters)
    args_dict.update(kwargs)

    diffusion_model._datme_info = {
        "size": None,
        "hooks": [],
        "args": args_dict
    }
    hook_datme_model(diffusion_model)

    block_counter = 0
    for name, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            idx = block_counter
            block_counter += 1
            # Layer-index filtering
            if merge_layers is not None:
                if idx not in merge_layers:
                    continue
            else:
                if merge_layer_start is not None and idx < merge_layer_start:
                    continue
                if merge_layer_end is not None and idx >= merge_layer_end:
                    continue
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
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_datme_info"):
            for hook in module._datme_info["hooks"]:
                hook.remove()
            module._datme_info["hooks"].clear()

        if module.__class__.__name__ == "DaTMeBlock":
            module.__class__ = module._parent
    
    return model
