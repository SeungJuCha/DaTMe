"""PixArt-alpha pipeline with DaTMe token merging.

This module owns the denoising loop and constructs the temporal and spatial
signals. ``datme.patch_pixart`` injects the transformer-block hooks, and
``datme.merge_pixart`` performs token partitioning and coordinate matching.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from .pipe_pixart_base import (
    ASPECT_RATIO_1024_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_256_BIN,
    retrieve_timesteps,
)
from .pipe_pixart_base import PixArtAlphaPipeline as PixArtAlphaPipeline_Base
from datme.patch_pixart import apply_patch, remove_patch, update_patch


class DaTMePixArtAlphaPipeline(PixArtAlphaPipeline_Base):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,
        # DaTMe parameters
        prune_start_step: Optional[int] = -1,
        merge_start_step: Optional[int] = -1,
        compression_ratio: Optional[float] = 0,
        importance_metric: str = "datme",
        temporal_metric: str = "tweedie_diff",
        lf_kernel: int = 17,
        avg_pool_kernel: int = 3,
        spatial_scale: int = 4,
        spatial_signal_mode: str = "coarse_residual",
        num_spatial_partitions: int = 4,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        orig_height, orig_width = height, width
        
        
        if use_resolution_binning:
            if self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        self.check_inputs(
            prompt, height, width, negative_prompt, callback_steps,
            prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt, do_classifier_free_guidance, negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt, device=device,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption, max_sequence_length=max_sequence_length,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, latent_channels, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)
            if do_classifier_free_guidance:
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    
        
        self.previous_tweedie_estimate = None
        self.prev_score_fn = None
        signal_maps = None
        patch_installed = False
        if (merge_start_step >= 0 or prune_start_step >= 0) and compression_ratio>0:
            if importance_metric == "datme":
                datme_merge_mode = "datme"
            else:
                datme_merge_mode = "none"
            
            self.transformer = apply_patch(
                self.transformer,
                ratio=compression_ratio,
                signal_maps=None,
                merge_mode="none",
                importance_metric=importance_metric,
                step=0,
                num_spatial_partitions=num_spatial_partitions,
                **kwargs,
            )
            patch_installed = True


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                # Update the DaTMe patch for this denoising step
                if compression_ratio > 0 and (merge_start_step >= 0 or prune_start_step >= 0):
                    assert merge_start_step > prune_start_step

                patch_active = (merge_start_step >= 0 or i >= prune_start_step) and compression_ratio > 0

                current_ratio = compression_ratio
                step_kwargs = {
                    **kwargs,
                    "num_spatial_partitions": num_spatial_partitions,
                }

                if patch_active and patch_installed:
                    if i >= merge_start_step:
                        update_patch(self.transformer, ratio=current_ratio, signal_maps=signal_maps, merge_mode=datme_merge_mode, step=i, **step_kwargs)
                    elif i >= prune_start_step:
                        update_patch(self.transformer, ratio=current_ratio, signal_maps=signal_maps, merge_mode="none", step=i, **step_kwargs)

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # Normalize scalar timesteps for the transformer batch.
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                # predict noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # CFG Guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred_cfg = noise_pred

                # Learned sigma strip
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred_cfg = noise_pred_cfg.chunk(2, dim=1)[0]
                    
                noise_pred_eval = noise_pred_cfg

                # Build DaTMe signals from the CFG-guided Tweedie estimate.
                if compression_ratio > 0 and importance_metric == "datme":
                    t_idx = int(t.item()) if torch.is_tensor(t) else int(t)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t_idx].to(latents.device)
                    beta_prod_t = 1 - alpha_prod_t
                    
                    # tweedie_estimate computed using final noise_pred_eval which has passed through CFG
                    tweedie_estimate = (latents - beta_prod_t ** 0.5 * noise_pred_eval) / (alpha_prod_t ** 0.5 + 1e-8)
                    score_fn = -noise_pred_eval.float() / (beta_prod_t ** 0.5 + 1e-8)

                    if self.previous_tweedie_estimate is not None and self.prev_score_fn is not None:
                        # Temporal signal: inter-step change of the Tweedie estimate.
                        if temporal_metric == "tweedie_diff":
                            temporal_delta = torch.abs(tweedie_estimate - self.previous_tweedie_estimate)
                        elif temporal_metric == "score_slope":
                            temporal_delta = torch.abs(score_fn - self.prev_score_fn)
                        else:
                            raise ValueError(f"Unsupported temporal_metric: {temporal_metric}")
                        temporal_signal = temporal_delta.norm(p=2, dim=1)  # (B, H, W)
                        
                        # Spatial signal: detail lost by coarse reconstruction.
                        tweedie_l2 = tweedie_estimate.norm(p=2, dim=1, keepdim=True) # (B, 1, H, W)
                        if spatial_signal_mode == "low_frequency":
                            spatial_kernel = lf_kernel
                            spatial_signal = F.avg_pool2d(
                                tweedie_l2.float(), kernel_size=spatial_kernel, stride=1, padding=spatial_kernel // 2
                            ).squeeze(1) # (B, H, W)
                        elif spatial_signal_mode == "local_residual":
                            spatial_kernel = avg_pool_kernel
                            local_mean = F.avg_pool2d(
                                tweedie_l2.float(), kernel_size=spatial_kernel, stride=1, padding=spatial_kernel // 2
                            )
                            spatial_signal = torch.abs(tweedie_l2.float() - local_mean).squeeze(1) # (B, H, W)
                        elif spatial_signal_mode == "local_variance":
                            spatial_kernel = avg_pool_kernel
                            tweedie_features = tweedie_estimate.float()
                            x2 = F.avg_pool2d(
                                tweedie_features * tweedie_features, kernel_size=spatial_kernel, stride=1, padding=spatial_kernel // 2
                            )
                            xmean = F.avg_pool2d(
                                tweedie_features, kernel_size=spatial_kernel, stride=1, padding=spatial_kernel // 2
                            )
                            spatial_signal = (x2 - xmean * xmean).clamp_min(0).mean(dim=1) # (B, H, W)
                        elif spatial_signal_mode == "coarse_residual":
                            scale = max(1, int(spatial_scale))
                            tweedie_features = tweedie_estimate.float()
                            if scale == 1:
                                spatial_signal = torch.zeros_like(tweedie_l2[:, 0])
                            else:
                                low_h = max(1, tweedie_features.shape[-2] // scale)
                                low_w = max(1, tweedie_features.shape[-1] // scale)
                                x_low = F.interpolate(tweedie_features, size=(low_h, low_w), mode="area")
                                x_low = F.interpolate(
                                    x_low,
                                    size=tweedie_features.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                spatial_signal = (tweedie_features - x_low).norm(p=2, dim=1) # (B, H, W)
                        elif spatial_signal_mode == "coarse_reconstruction":
                            scale = max(1, int(spatial_scale))
                            tweedie_features = tweedie_estimate.float()
                            if scale == 1:
                                spatial_signal = tweedie_l2[:, 0]
                            else:
                                low_h = max(1, tweedie_features.shape[-2] // scale)
                                low_w = max(1, tweedie_features.shape[-1] // scale)
                                x_low = F.interpolate(tweedie_features, size=(low_h, low_w), mode="area")
                                x_low = F.interpolate(
                                    x_low,
                                    size=tweedie_features.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                spatial_signal = x_low.norm(p=2, dim=1)
                        else:
                            raise ValueError(f"Unsupported spatial_signal_mode: {spatial_signal_mode}")

                        signal_maps = {
                            "temporal_signal": temporal_signal.detach(),
                            "spatial_signal": spatial_signal.detach(),
                        }
                    else:
                        signal_maps = None

                    self.previous_tweedie_estimate = tweedie_estimate.detach()
                    self.prev_score_fn = score_fn.detach()



                if num_inference_steps == 1:
                    latents = self.scheduler.step(noise_pred_cfg, t, latents, **extra_step_kwargs).pred_original_sample
                else:
                    latents = self.scheduler.step(noise_pred_cfg, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)


        if patch_installed:
            remove_patch(self.transformer)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)


        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
