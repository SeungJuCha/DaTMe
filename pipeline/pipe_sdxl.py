"""Stable Diffusion XL pipeline with DaTMe token merging.

The temporal signal measures the inter-step change in the Tweedie estimate.
The spatial signal measures its residual from a coarse-scale reconstruction.
Both signals are computed in the denoising loop and consumed by the patched
U-Net blocks when selecting source, destination, and constraint tokens.
``datme.patch_unet`` installs those hooks, and ``datme.merge_unet_datme``
performs token partitioning and matching.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from diffusers import StableDiffusionXLPipeline as StableDiffusionXLPipeline_Base
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

from datme.patch_unet import apply_patch, remove_patch


class DaTMeStableDiffusionXLPipeline(StableDiffusionXLPipeline_Base):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[tuple] = None,
        crops_coords_top_left: tuple = (0, 0),
        target_size: Optional[tuple] = None,
        clip_skip: Optional[int] = None,
        # ── Token merging params ────────────────────────────────────────
        prune_start_step: Optional[int] = -1,
        merge_start_step: Optional[int] = -1,
        compression_ratio: Optional[float] = 0.0,
        importance_metric: str = "datme",
        max_downsample: int = 2,         # 2 → 64×64 layers only
        merge_layer_start: Optional[int] = None,  # None = all
        merge_layer_end: Optional[int] = None,    # None = all
        merge_layers: Optional[List[int]] = None, # Specific indices
        lf_kernel: int = 17,             # kernel size for local coherence avg pooling (padding = k//2)
        spatial_signal_mode: str = "coarse_residual",
        spatial_scale: int = 2,
        num_spatial_partitions: int = 4,
        **kwargs,
    ):
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # ── 1) Encode prompts ───────────────────────────────────────────
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=clip_skip,
        )

        # ── 2) Timesteps ────────────────────────────────────────────────
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # ── 3) Prepare latents ──────────────────────────────────────────
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            1 * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # ── 4) Extra kwargs for scheduler.step ─────────────────────────
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ── 5) Build SDXL added conditioning ───────────────────────────
        original_size = original_size or (height, width)
        target_size   = target_size   or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if hasattr(self, '_get_add_time_ids'):
            add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pooled_prompt_embeds.shape[-1]),
            )
        else:
            # fallback
            add_time_ids = torch.tensor(
                [list(original_size) + list(crops_coords_top_left) + list(target_size)],
                dtype=prompt_embeds.dtype,
            )
        negative_add_time_ids = add_time_ids  # same for simplicity

        if do_classifier_free_guidance:
            prompt_embeds       = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds     = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids        = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds   = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids    = add_time_ids.to(device)

        # ── 6) State for importance metrics ────────────────────────────
        self._num_timesteps = len(timesteps)
        self.previous_tweedie_estimate   = None
        signal_maps = None
        patch_installed = False

        # ── 7) Denoising loop ───────────────────────────────────────────
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # Update the DaTMe patch for this denoising step
                if compression_ratio > 0 and (merge_start_step >= 0 or prune_start_step >= 0):
                    assert compression_ratio > 0 and merge_start_step > prune_start_step

                    if i >= merge_start_step:
                        if importance_metric == "datme" and signal_maps is not None:
                            if not patch_installed:
                                self.unet = apply_patch(
                                    self.unet,
                                    ratio=compression_ratio,
                                    signal_maps=signal_maps,
                                    merge_mode="datme",
                                    step=i,
                                    max_downsample=max_downsample,
                                    merge_layer_start=merge_layer_start,
                                    merge_layer_end=merge_layer_end,
                                    merge_layers=merge_layers,
                                    num_spatial_partitions=num_spatial_partitions,
                                    **kwargs,
                                )
                                patch_installed = True
                            else:
                                patch_args = self.unet._datme_info["args"]
                                patch_args["signal_maps"] = signal_maps
                                patch_args["step"] = i
                        else:
                            if patch_installed:
                                remove_patch(self.unet)
                                patch_installed = False
                    elif i == prune_start_step:
                        self.unet = apply_patch(
                            self.unet,
                            ratio=compression_ratio,
                            merge_mode="none",
                            max_downsample=max_downsample,
                            merge_layer_start=merge_layer_start,
                            merge_layer_end=merge_layer_end,
                            merge_layers=merge_layers,
                            num_spatial_partitions=num_spatial_partitions,
                        )
                        patch_installed = False

                # UNet forward
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # CFG
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred_cfg = noise_pred

                # Build DaTMe signals for the next denoising step.
                if compression_ratio > 0 and importance_metric == "datme":
                    t_idx = int(t.item()) if torch.is_tensor(t) else int(t)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t_idx].to(latents.device)
                    beta_prod_t = 1 - alpha_prod_t
                    
                    tweedie_estimate = (latents - beta_prod_t ** 0.5 * noise_pred_cfg) / (alpha_prod_t ** 0.5 + 1e-8)

                    if self.previous_tweedie_estimate is not None:
                        # Temporal signal: inter-step change of the Tweedie estimate.
                        temporal_delta = torch.abs(tweedie_estimate - self.previous_tweedie_estimate)
                        temporal_signal = temporal_delta.norm(p=2, dim=1)  # (B, H, W)
                        
                        # Spatial signal: detail lost by coarse reconstruction.
                        tweedie_l2 = tweedie_estimate.norm(p=2, dim=1, keepdim=True) # (B, 1, H, W)
                        spatial_signal = None
                        
                        if spatial_signal_mode == "none":
                            spatial_signal = torch.zeros_like(tweedie_l2.squeeze(1))
                        elif spatial_signal_mode == "local_residual":
                            avg_pool_kernel = kwargs.get("avg_pool_kernel", 3)
                            local_mean = F.avg_pool2d(
                                tweedie_l2,
                                kernel_size=avg_pool_kernel,
                                stride=1,
                                padding=avg_pool_kernel // 2,
                            )
                            spatial_signal = torch.abs(tweedie_l2 - local_mean).squeeze(1)
                        elif spatial_signal_mode == "coarse_residual":
                            if tweedie_estimate.shape[-2] < spatial_scale * 2:
                                spatial_signal = torch.zeros_like(tweedie_l2.squeeze(1))
                            else:
                                down_h = max(1, tweedie_estimate.shape[-2] // spatial_scale)
                                down_w = max(1, tweedie_estimate.shape[-1] // spatial_scale)
                                x_down = F.interpolate(tweedie_estimate, size=(down_h, down_w), mode="area")
                                x_up = F.interpolate(x_down, size=tweedie_estimate.shape[-2:], mode="bilinear", align_corners=False)
                                diff = tweedie_estimate - x_up
                                diff_n = diff / (diff.norm(p=2, dim=[2, 3], keepdim=True) + 1e-8)
                                spatial_signal = diff_n.norm(p=2, dim=1)
                        elif spatial_signal_mode == "local_variance":
                            avg_pool_kernel = kwargs.get("avg_pool_kernel", 3)
                            local_mean = F.avg_pool2d(
                                tweedie_l2, kernel_size=avg_pool_kernel, stride=1, padding=avg_pool_kernel // 2
                            )
                            local_sq_mean = F.avg_pool2d(
                                tweedie_l2 ** 2, kernel_size=avg_pool_kernel, stride=1, padding=avg_pool_kernel // 2
                            )
                            var = F.relu(local_sq_mean - local_mean ** 2)
                            spatial_signal = torch.sqrt(var + 1e-6).squeeze(1)
                        else: # low-frequency spatial signal
                            spatial_signal = F.avg_pool2d(
                                tweedie_l2.float(), kernel_size=lf_kernel, stride=1, padding=lf_kernel // 2
                            ).squeeze(1) # (B, H, W)
                        
                        signal_maps = {
                            "temporal_signal": temporal_signal.detach(),
                            "spatial_signal": spatial_signal.detach(),
                        }
                    else:
                        signal_maps = None

                    self.previous_tweedie_estimate = tweedie_estimate.detach()

                # Scheduler step
                latents = self.scheduler.step(noise_pred_cfg, t, latents, **extra_step_kwargs, return_dict=False)[0]
                progress_bar.update()


        # Remove patch
        if patch_installed or (merge_start_step >= 0 or prune_start_step >= 0):
            remove_patch(self.unet)

        # Decode — SDXL VAE overflows in fp16, must upcast to fp32 first
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        image = self.image_processor.postprocess(image, output_type=output_type)


        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
