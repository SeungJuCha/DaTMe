"""FLUX pipeline with the finalized DaTMe merge policy.

This module owns the flow-matching denoising loop and constructs the temporal
and spatial signals. ``datme.patch_mmdit`` injects the transformer-block hooks,
and ``datme.merge_flux`` performs image-token partitioning and matching.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import (
    calculate_shift,
    retrieve_timesteps,
    FluxPipelineOutput,
)

from datme.patch_mmdit import apply_patch, update_patch, remove_patch


class DaTMeFluxPipeline(FluxPipeline):
    """Merge FLUX image tokens while preserving text tokens."""

    # ── denoising loop ────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        # ── standard FluxPipeline args ──
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 35,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # DaTMe parameters
        compression_ratio: float = 0.3,
        merge_start_step: int = 1,
        num_spatial_partitions: int = 4,
        double_block_indices: Union[str, List[int]] = "all",
        single_block_indices: Union[str, List[int]] = "none",
        spatial_scale: int = 4,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width  = width  or self.default_sample_size * self.vae_scale_factor

        # 1. Validate inputs
        self.check_inputs(
            prompt, prompt_2, height, width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale         = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt              = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None else None
        )

        # 2. Encode prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 3. Prepare latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps_arr, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps_arr) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps_arr)

        N_img = latents.shape[1]
        img_h = height // (self.vae_scale_factor)
        img_w = width  // (self.vae_scale_factor)
        if img_h * img_w != N_img:
            raise ValueError(f"FLUX token grid {img_h}x{img_w} does not match {N_img} image tokens.")
        N_txt  = prompt_embeds.shape[1]

        # ── Reset per-call Tweedie state ──────────────────────────────────────
        self.previous_tweedie_estimate    = None
        signal_maps             = None
        patch_installed = False
        if compression_ratio > 0:
            self.transformer = apply_patch(
                self.transformer,
                ratio=compression_ratio,
                signal_maps=None,
                merge_mode="none",
                double_layer_indices=double_block_indices,
                single_layer_indices=single_block_indices,
            )
            # Store spatial dims and N_txt for patch blocks
            self.transformer._datme_info["img_h"] = img_h
            self.transformer._datme_info["img_w"] = img_w
            self.transformer._datme_info["N_txt"] = N_txt
            patch_installed = True

        # 6. Denoising loop
        try:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps_arr):
                    if self.interrupt:
                        continue

                    if patch_installed and compression_ratio > 0:
                        step_kwargs: Dict[str, Any] = {
                            "num_spatial_partitions": num_spatial_partitions,
                            "ind_mode": "ind_topm_toph",
                        }
                        if i >= merge_start_step:
                            update_patch(
                                self.transformer,
                                ratio=compression_ratio,
                                signal_maps=signal_maps,
                                merge_mode="datme",
                                step=i,
                                **step_kwargs,
                            )
                        else:
                            update_patch(
                                self.transformer,
                                ratio=compression_ratio,
                                signal_maps=None,
                                merge_mode="none",
                                step=i,
                            )

                    # ── Transformer forward ───────────────────────────────────────
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    guidance = None
                    if self.transformer.config.guidance_embeds:
                        guidance = torch.tensor([guidance_scale], device=device)
                        guidance = guidance.expand(latents.shape[0])

                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # Flow-matching Tweedie estimate: x0 = xt - sigma_t * v.
                    if compression_ratio > 0:
                        # tweedie_estimate = x_t - sigma_t * v_pred   (flow matching)
                        sigma_t  = t.float() / 1000.0
                        tweedie_estimate  = latents.float() - sigma_t * noise_pred.float()
                        # tweedie_estimate shape: (B, N_img, 64)

                        if self.previous_tweedie_estimate is None:
                            signal_maps = None
                        else:
                            # Temporal signal: inter-step change of the Tweedie estimate.
                            temporal_delta = torch.abs(tweedie_estimate - self.previous_tweedie_estimate)
                            temporal_signal = temporal_delta.norm(p=2, dim=-1)

                            # Spatial signal: residual from a coarse reconstruction.
                            B, _, C = tweedie_estimate.shape
                            tweedie_grid = tweedie_estimate.transpose(1, 2).reshape(B, C, img_h, img_w)
                            scale = max(1, int(spatial_scale))
                            if scale == 1:
                                spatial_signal = torch.zeros_like(temporal_signal)
                            else:
                                low_h = max(1, img_h // scale)
                                low_w = max(1, img_w // scale)
                                x_low = F.interpolate(tweedie_grid, size=(low_h, low_w), mode="area")
                                x_low = F.interpolate(x_low, size=(img_h, img_w), mode="bilinear", align_corners=False)
                                spatial_signal = (tweedie_grid - x_low).norm(p=2, dim=1).flatten(1)

                            signal_maps = {
                                "temporal_signal": temporal_signal.detach(),
                                "spatial_signal": spatial_signal.detach(),
                            }

                        self.previous_tweedie_estimate = tweedie_estimate.detach()

                    # ── Scheduler step ────────────────────────────────────────────
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    if latents.dtype != latents_dtype:
                        latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        cb_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            cb_kwargs[k] = locals()[k]
                        cb_out   = callback_on_step_end(self, i, t, cb_kwargs)
                        latents  = cb_out.pop("latents", latents)
                        prompt_embeds        = cb_out.pop("prompt_embeds", prompt_embeds)

                    if i == len(timesteps_arr) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()

        finally:
            if patch_installed:
                remove_patch(self.transformer)

        # ── Decode ────────────────────────────────────────────────────────────
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image   = self.vae.decode(latents, return_dict=False)[0]
            image   = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)
