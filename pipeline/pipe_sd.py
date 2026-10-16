"""Stable Diffusion 1.5 pipeline with DaTMe token merging.

This module owns the denoising loop and constructs the temporal and spatial
signals. ``datme.patch_unet`` injects the block hooks, and
``datme.merge_unet_datme`` performs token partitioning and matching.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import deprecate, replace_example_docstring

from .pipe_sd_base import (
    EXAMPLE_DOC_STRING,
    StableDiffusionPipelineOutput,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from .pipe_sd_base import StableDiffusionPipeline as StableDiffusionPipeline_Base
from datme.patch_unet import apply_patch, remove_patch


class DaTMeStableDiffusionPipeline(StableDiffusionPipeline_Base):

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 1)
        prune_start_step: Optional[int] = -1,
        merge_start_step: Optional[int] = -1,
        compression_ratio: Optional[float] = 0,  
        use_signal_guided_merging: Optional[bool] = False, 
        importance_metric: str = "cfg_diff", # e.g. "cfg_diff", "tweedie_triangle", "datme"
        avg_pool_kernel: int = 7,           # kernel size for local coherence avg pooling (padding = k//2)
        spatial_signal_mode: str = "coarse_residual",
        lf_kernel: int = 17,
        spatial_scale: int = 4,
        num_spatial_partitions: int = 4,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Prepare scheduler-specific step arguments.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        self.prev_score = None # Initialize for slope calculation
        self.previous_tweedie_estimate = None # Initialize for Tweedie slope calculation
        signal_maps = None # Initialize signal_maps
        
        patch_installed = False  # track whether apply_patch has been called once

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # 2)
                if compression_ratio > 0 and (merge_start_step >= 0 or prune_start_step >= 0):

                    assert compression_ratio > 0 and merge_start_step > prune_start_step

                    if use_signal_guided_merging:
                        if i >= merge_start_step:
                            should_merge = (
                                (importance_metric == "datme" and signal_maps is not None)
                                or (importance_metric in {"cfg_tweedie", "cfg_diff"} and signal_maps is not None)
                            )

                            if should_merge:
                                if not patch_installed:
                                    # First merge step: install patch (class replacement, hooks)
                                    merge_mode = "datme" if importance_metric == "datme" else "mean"
                                    self.unet = apply_patch(
                                        self.unet,
                                        ratio=compression_ratio,
                                        signal_maps=signal_maps,
                                        merge_mode=merge_mode,
                                        step=i,
                                        num_spatial_partitions=num_spatial_partitions,
                                        **kwargs,
                                    )
                                    patch_installed = True
                                else:
                                    # Subsequent steps: just update signal_maps and step in-place
                                    patch_args = self.unet._datme_info["args"]
                                    patch_args["signal_maps"] = signal_maps
                                    patch_args["step"] = i
                            else:
                                # No merge for this step
                                remove_patch(self.unet)
                                patch_installed = False
                        elif i == prune_start_step:
                            self.unet = apply_patch(self.unet, ratio=compression_ratio, merge_mode="none")
                            patch_installed = False
                    else:
                        if i >= merge_start_step:
                            if not patch_installed:
                                self.unet = apply_patch(
                                    self.unet,
                                    ratio=compression_ratio,
                                    step=i,
                                    **kwargs,
                                )
                                patch_installed = True
                            else:
                                patch_args = self.unet._datme_info["args"]
                                patch_args["step"] = i
                        elif i == prune_start_step:
                            self.unet = apply_patch(self.unet, ratio=compression_ratio, merge_mode="none")
                            patch_installed = False

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if use_signal_guided_merging:
                    if importance_metric == "score_norm" or importance_metric == "score_norm_fix":
                        # Score Function = - epsilon / sigma_t
                        # sigma_t = sqrt(1 - alpha_cumprod_t)
                        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device=latents.device)
                        sigma_t = (1 - alpha_prod_t) ** 0.5
                        
                        # Score Norm = || epsilon || / sigma_t
                        # noise_pred_text is (B, 4, H, W). Norm across channels (dim 1).
                        # We use a small epsilon for stability if sigma_t is 0 (though unlikely in intermediate steps)
                        score = -noise_pred_text / (sigma_t + 1e-8)
                        # signal_maps = torch.abs(torch.mean(score, dim=1)) # B, H, W
                        signal_maps = torch.mean(torch.abs(score), dim=1)
                    elif importance_metric == "inverse_score_norm":
                        # Inverted Score: High Score -> Low Value -> Merged
                        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device=latents.device)
                        sigma_t = (1 - alpha_prod_t) ** 0.5
                        score = -noise_pred_text / (sigma_t + 1e-8)
                        signal_maps = -torch.mean(torch.abs(score), dim=1) # Negative!
                    elif importance_metric == "score_slope" or importance_metric == "score_slope_fixed" or importance_metric == "score_slope_interval":
                        # Slope: abs(d(Score)/dt) ~ abs(Score_t - Score_{t-1})
                        
                        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device=latents.device)
                        sigma_t = (1 - alpha_prod_t) ** 0.5
                        score = -noise_pred_text / (sigma_t + 1e-8)
                        
                        if self.prev_score is None:
                            # Step 1: No Slope available
                            signal_maps = None
                        else:
                            # Calculate Slope
                            # score is (B, 4, H, W). Slope = Mean Abs Diff
                            diff = torch.abs(score - self.prev_score)
                            signal_maps = torch.mean(diff, dim=1) # (B, H, W)
                        
                        
                        self.prev_score = score
                    elif importance_metric == "score_hybrid":
                        # Hybrid: Max of (Norm Rank, Slope Rank)
                        # We want to prioritize tokens that are EITHER High Slope OR High Norm.
                        
                        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device=latents.device)
                        sigma_t = (1 - alpha_prod_t) ** 0.5
                        score = -noise_pred_text / (sigma_t + 1e-8)
                        
                        # 1. Norm Score
                        score_norm = torch.mean(torch.abs(score), dim=1) # (B, H, W)
                        
                        if self.prev_score is None:
                            # Step 1: Slope undefined, use Norm only
                            signal_maps = score_norm
                        else:
                            # 2. Slope Score
                            diff = torch.abs(score - self.prev_score)
                            score_slope = torch.mean(diff, dim=1) # (B, H, W)
                            
                            # 3. Compute Ranks (Per sample)
                            # Flatten: (B, N)
                            B_size, H, W = score_norm.shape
                            N = H * W
                            
                            flat_norm = score_norm.view(B_size, -1)
                            flat_slope = score_slope.view(B_size, -1)
                            
                            # Rank: 0 is Highest Score (Most Important)
                            rank_norm = torch.argsort(torch.argsort(flat_norm, dim=1, descending=True), dim=1)
                            rank_slope = torch.argsort(torch.argsort(flat_slope, dim=1, descending=True), dim=1)
                            
                            # 4. Combine: Take the WORST rank (Maximum value)
                            # Intersection Logic: You need to be good in BOTH to be top.
                            # If I am Rank 0 in Slope and Rank 100 in Norm -> Max is 100 -> Bad.
                            # If I am Rank 10 in Slope and Rank 10 in Norm -> Max is 10 -> Good.
                            worst_rank = torch.max(rank_norm, rank_slope)
                            
                            # 5. Convert back to Score (Higher is better)
                            signal_maps = -worst_rank.float().view(B_size, H, W)
                        
                        self.prev_score = score
                    elif importance_metric == "tweedie_slope" or importance_metric == "tweedie_slope_interval":
                        # Tweedie x̂₀(t) = (x_t - √β_t · ε_guided) / √ᾱ_t
                        # noise_pred here is the CFG-guided prediction (computed at line ~339)
                        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(device=latents.device)
                        beta_prod_t = 1 - alpha_prod_t
                        tweedie_estimate = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5 + 1e-8)

                        if self.previous_tweedie_estimate is None:
                            signal_maps = None  # First step: no slope yet
                        else:
                            temporal_delta = torch.abs(tweedie_estimate - self.previous_tweedie_estimate)
                            signal_maps = temporal_delta.mean(1)  # (B, H, W)

                        self.previous_tweedie_estimate = tweedie_estimate.detach()
                    elif importance_metric == "tweedie_triangle":
                        # Joint token roles from the temporal and spatial signals.
                        # IND = hi_m OR hi_h,  DST = lo_m AND lo_h
                        import torch.nn.functional as _F
                        t_idx = int(t.item()) if torch.is_tensor(t) else int(t)
                        alpha_prod_t = self.scheduler.alphas_cumprod[t_idx].to(latents.device)
                        beta_prod_t = 1 - alpha_prod_t
                        tweedie_estimate = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5 + 1e-8)

                        if self.previous_tweedie_estimate is None:
                            signal_maps = None
                        else:
                            temporal_delta = torch.abs(tweedie_estimate - self.previous_tweedie_estimate)
                            slopes_2d = temporal_delta.norm(p=2, dim=1)  # (B, H, W)

                            # Construct the selected spatial importance signal.
                            tweedie_l2 = tweedie_estimate[0].float().norm(p=2, dim=0)   # (H, W)
                            eps_log = 1e-8
                            if spatial_signal_mode == "none":
                                # Temporal-only ablation.
                                spatial_signal = torch.ones_like(tweedie_l2)
                            elif spatial_signal_mode == "global":
                                # Global deviation from the scene-wide mean.
                                spatial_signal = torch.abs(tweedie_l2 - tweedie_l2.mean())
                            elif spatial_signal_mode == "feature":
                                # Local feature-space deviation across latent channels.
                                local_mean_feat = _F.avg_pool2d(
                                    tweedie_estimate[0].float(),
                                    kernel_size=avg_pool_kernel, stride=1, padding=avg_pool_kernel // 2
                                )  # (4, H, W)
                                spatial_signal = (tweedie_estimate[0].float() - local_mean_feat).norm(p=2, dim=0)
                            elif spatial_signal_mode == "lowfreq":
                                # low-frequency structure: strong blur on feature map, then keep coarse scene layout
                                lf_kernel = max(avg_pool_kernel * 3, 9)
                                if lf_kernel % 2 == 0:
                                    lf_kernel += 1
                                lowfreq_feat = _F.avg_pool2d(
                                    tweedie_estimate[0].float(),
                                    kernel_size=lf_kernel,
                                    stride=1,
                                    padding=lf_kernel // 2,
                                )  # (4, H, W)
                                lowfreq_l2 = lowfreq_feat.norm(p=2, dim=0)
                                spatial_signal = torch.abs(lowfreq_l2 - lowfreq_l2.mean())
                            elif spatial_signal_mode == "region":
                                # coarse region structure: area-downsample to coarse cells, then upsample
                                region_div = max(2, avg_pool_kernel)
                                coarse_h = max(4, tweedie_estimate.shape[-2] // region_div)
                                coarse_w = max(4, tweedie_estimate.shape[-1] // region_div)
                                coarse_feat = _F.interpolate(
                                    tweedie_estimate[0].float().unsqueeze(0),
                                    size=(coarse_h, coarse_w),
                                    mode="area",
                                )
                                coarse_up = _F.interpolate(
                                    coarse_feat,
                                    size=tweedie_estimate.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze(0)
                                coarse_l2 = coarse_up.norm(p=2, dim=0)
                                spatial_residual = torch.abs(tweedie_l2 - coarse_l2)
                                # Spread boundary responses into nearby interior tokens.
                                _spread_k = 15
                                regional_spatial_signal = _F.max_pool2d(
                                    spatial_residual.unsqueeze(0).unsqueeze(0),
                                    kernel_size=_spread_k, stride=1, padding=_spread_k // 2
                                ).squeeze(0).squeeze(0)
                                # Combine regional boundaries with local details.
                                local_mean = _F.avg_pool2d(
                                    tweedie_l2.unsqueeze(0).unsqueeze(0),
                                    kernel_size=avg_pool_kernel, stride=1, padding=avg_pool_kernel // 2
                                ).squeeze(0).squeeze(0)
                                local_spatial_signal = torch.abs(tweedie_l2 - local_mean)
                                spatial_signal = torch.max(regional_spatial_signal, local_spatial_signal)
                            elif spatial_signal_mode == "coarse_contrast":
                                # Measure feature contrast between coarse spatial cells.
                                grid_h = min(coarse_grid, tweedie_estimate.shape[-2])
                                grid_w = min(coarse_grid, tweedie_estimate.shape[-1])
                                coarse_feat = _F.interpolate(
                                    tweedie_estimate[0].float().unsqueeze(0),
                                    size=(grid_h, grid_w),
                                    mode="area",
                                ).squeeze(0)  # (4, gh, gw)

                                contrast = torch.zeros(grid_h, grid_w, device=coarse_feat.device, dtype=coarse_feat.dtype)
                                neighbor_count = torch.zeros_like(contrast)

                                if grid_h > 1:
                                    diff_v = (coarse_feat[:, 1:, :] - coarse_feat[:, :-1, :]).norm(p=2, dim=0)
                                    contrast[1:, :] += diff_v
                                    contrast[:-1, :] += diff_v
                                    neighbor_count[1:, :] += 1
                                    neighbor_count[:-1, :] += 1
                                if grid_w > 1:
                                    diff_h = (coarse_feat[:, :, 1:] - coarse_feat[:, :, :-1]).norm(p=2, dim=0)
                                    contrast[:, 1:] += diff_h
                                    contrast[:, :-1] += diff_h
                                    neighbor_count[:, 1:] += 1
                                    neighbor_count[:, :-1] += 1

                                contrast = contrast / neighbor_count.clamp_min(1.0)
                                spatial_signal = _F.interpolate(
                                    contrast.unsqueeze(0).unsqueeze(0),
                                    size=tweedie_estimate.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze(0).squeeze(0)
                            elif spatial_signal_mode == "coarse_region_consistency":
                                # Emphasize coarse cells with contrast and internal consistency.
                                grid_h = min(coarse_grid, tweedie_estimate.shape[-2])
                                grid_w = min(coarse_grid, tweedie_estimate.shape[-1])
                                fine_feat = tweedie_estimate[0].float().unsqueeze(0)  # (1, 4, H, W)
                                coarse_feat = _F.interpolate(
                                    fine_feat,
                                    size=(grid_h, grid_w),
                                    mode="area",
                                ).squeeze(0)  # (4, gh, gw)

                                contrast = torch.zeros(grid_h, grid_w, device=coarse_feat.device, dtype=coarse_feat.dtype)
                                neighbor_count = torch.zeros_like(contrast)

                                if grid_h > 1:
                                    diff_v = (coarse_feat[:, 1:, :] - coarse_feat[:, :-1, :]).norm(p=2, dim=0)
                                    contrast[1:, :] += diff_v
                                    contrast[:-1, :] += diff_v
                                    neighbor_count[1:, :] += 1
                                    neighbor_count[:-1, :] += 1
                                if grid_w > 1:
                                    diff_h = (coarse_feat[:, :, 1:] - coarse_feat[:, :, :-1]).norm(p=2, dim=0)
                                    contrast[:, 1:] += diff_h
                                    contrast[:, :-1] += diff_h
                                    neighbor_count[:, 1:] += 1
                                    neighbor_count[:, :-1] += 1

                                contrast = contrast / neighbor_count.clamp_min(1.0)

                                coarse_up = _F.interpolate(
                                    coarse_feat.unsqueeze(0),
                                    size=tweedie_estimate.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                residual_map = (fine_feat - coarse_up).norm(p=2, dim=1, keepdim=True)  # (1, 1, H, W)
                                incoherence = _F.interpolate(
                                    residual_map,
                                    size=(grid_h, grid_w),
                                    mode="area",
                                ).squeeze(0).squeeze(0)
                                consistency = 1.0 / (incoherence + 1e-6)

                                contrast_norm = contrast / (contrast.mean() + 1e-6)
                                consistency_norm = consistency / (consistency.mean() + 1e-6)
                                region_score = contrast_norm * consistency_norm

                                spatial_signal = _F.interpolate(
                                    region_score.unsqueeze(0).unsqueeze(0),
                                    size=tweedie_estimate.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze(0).squeeze(0)
                            else:  # "local" (default)
                                # Local deviation from the neighboring L2 mean.
                                local_mean = _F.avg_pool2d(
                                    tweedie_l2.unsqueeze(0).unsqueeze(0),
                                    kernel_size=avg_pool_kernel, stride=1, padding=avg_pool_kernel // 2
                                ).squeeze(0).squeeze(0)
                                spatial_signal = torch.abs(tweedie_l2 - local_mean)

                            # Log-space mean±std thresholding (O(n), GPU-native)
                            log_temporal = torch.log(slopes_2d[0].float() + eps_log).flatten()
                            log_spatial = torch.log(spatial_signal.float() + eps_log).flatten()
                            N_tok = float(len(log_temporal))

                            mu_m, sig_m = log_temporal.mean(), log_temporal.std()
                            mu_h, sig_h = log_spatial.mean(), log_spatial.std()

                            if sigma_k is not None:
                                lo_m = log_temporal < (mu_m - sigma_k * sig_m)
                                lo_h = log_spatial < (mu_h - sigma_k * sig_h)
                            else:
                                lo_m = lo_h = torch.zeros(int(N_tok), dtype=torch.bool, device=log_temporal.device)

                            if joint_1d:
                                log_j = log_temporal + log_spatial
                                if sigma_k is None:
                                    # Non-heuristic: "high" = top (1-cr) per axis; IND = m∩h intersection
                                    _top_cnt = int(N_tok * (1.0 - compression_ratio))
                                    _hi_m = torch.zeros(int(N_tok), dtype=torch.bool, device=log_temporal.device)
                                    _hi_m.scatter_(0, log_temporal.topk(_top_cnt).indices, True)
                                    _hi_h = torch.zeros(int(N_tok), dtype=torch.bool, device=log_spatial.device)
                                    _hi_h.scatter_(0, log_spatial.topk(_top_cnt).indices, True)
                                    ind_nat = (_hi_m & _hi_h).float().mean().item()
                                else:
                                    if per_axis_inter:
                                        # Per-axis threshold: mu - sigma_k*sigma on each axis, then intersection
                                        # Larger sigma_k broadens the constraint set.
                                        _hi_m = log_temporal >= (mu_m - sigma_k * sig_m)
                                        _hi_h = log_spatial >= (mu_h - sigma_k * sig_h)
                                        ind_nat = (_hi_m & _hi_h).float().mean().item()
                                    else:
                                        # Threshold the joint score distribution.
                                        # sigma_k < 0 → upper threshold (e.g. sigma_k=-1 → mu_j+sig_j → ~16%)
                                        mu_j, sig_j = log_j.mean(), log_j.std()
                                        ind_nat = (log_j >= (mu_j - sigma_k * sig_j)).float().mean().item()
                                dst_nat = max(0.0, 1.0 - compression_ratio - ind_nat)
                            elif single_threshold:
                                # μ-kσ only: DST = lo_m AND lo_h, IND = complement (NOT AND)
                                # Constraint tokens are high on either signal; DST is low on both.
                                dst_nat = (lo_m & lo_h).sum().item() / N_tok
                                ind_nat = 1.0 - dst_nat
                            else:
                                hi_m = log_temporal > (mu_m + sigma_k * sig_m)
                                hi_h = log_spatial > (mu_h + sigma_k * sig_h)
                                ind_nat = (hi_m | hi_h).sum().item() / N_tok   # OR → IND
                                dst_nat = (lo_m & lo_h).sum().item() / N_tok   # AND → DST

                            non_src = 1.0 - compression_ratio
                            MIN_RATE = 0.05
                            if auto_ind:
                                ind_rate = non_src
                                dst_frac = 0.0
                                # m∩h intersection → IND ratio (non-heuristic)
                                _top_cnt = int(N_tok * non_src)
                                _high_m = torch.zeros(int(N_tok), dtype=torch.bool, device=log_temporal.device)
                                _high_m.scatter_(0, log_temporal.topk(_top_cnt).indices, True)
                                _high_h = torch.zeros(int(N_tok), dtype=torch.bool, device=log_spatial.device)
                                _high_h.scatter_(0, log_spatial.topk(_top_cnt).indices, True)
                                self._cr_ind_ratio = (_high_m & _high_h).float().mean().item()
                            elif joint_1d:
                                ind_rate = min(max(ind_nat, MIN_RATE), non_src - MIN_RATE)
                                dst_frac = max(0.0, non_src - ind_rate)
                            else:
                                total_nat = ind_nat + dst_nat
                                if total_nat > 1e-6:
                                    ind_rate = non_src * ind_nat / total_nat
                                    dst_frac = non_src * dst_nat / total_nat
                                else:
                                    ind_rate = non_src / 2.0
                                    dst_frac = non_src / 2.0
                                ind_rate = max(ind_rate, MIN_RATE)
                                dst_frac = max(dst_frac, MIN_RATE)
                                _total = ind_rate + dst_frac
                                if _total > non_src:
                                    ind_rate = non_src * ind_rate / _total
                                    dst_frac = non_src * dst_frac / _total

                            # joint score → token ranking
                            if spatial_signal_mode == "or_gate":
                                # Promote tokens whose spatial signal exceeds its threshold.
                                spatial_threshold = log_spatial.mean()
                                spatial_gate = (log_spatial > spatial_threshold).float()
                                # Raise promoted tokens to the temporal maximum.
                                log_temporal_max = log_temporal.max()
                                boosted = log_temporal + spatial_gate * (log_temporal_max - log_temporal + 1.0)
                                joint_score = torch.exp(boosted).reshape(slopes_2d[0].shape)
                            elif max_agg:
                                joint_score = torch.exp(torch.max(log_temporal, log_spatial)).reshape(slopes_2d[0].shape)
                            else:
                                joint_score = torch.exp(log_temporal + log_spatial).reshape(slopes_2d[0].shape)
                            signal_maps = joint_score.unsqueeze(0).to(dtype=slopes_2d.dtype)  # (1, H, W)


                            self._tri_top_k_rate = ind_rate
                            self._tri_dst_m_rate = dst_frac


                            self.previous_tweedie_estimate = tweedie_estimate.detach()
                    elif importance_metric == "datme":
                        import torch.nn.functional as _F

                        t_idx = int(t.item()) if torch.is_tensor(t) else int(t)
                        alpha_prod_t = self.scheduler.alphas_cumprod[t_idx].to(latents.device)
                        beta_prod_t = 1 - alpha_prod_t
                        tweedie_estimate = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5 + 1e-8)

                        if self.previous_tweedie_estimate is None:
                            signal_maps = None
                        else:
                            # Temporal signal: inter-step change of the Tweedie estimate.
                            temporal_delta = torch.abs(tweedie_estimate - self.previous_tweedie_estimate)
                            temporal_signal = temporal_delta.norm(p=2, dim=1)  # (B, H, W)

                            # Spatial signal: residual from a coarse reconstruction.
                            tweedie_l2 = tweedie_estimate[0].float().norm(p=2, dim=0)  # (H, W)
                            if spatial_signal_mode == "low_frequency":
                                # Low-frequency ablation uses the smoothed estimate directly.
                                spatial_signal = _F.avg_pool2d(
                                    tweedie_l2.unsqueeze(0).unsqueeze(0),
                                    kernel_size=lf_kernel,
                                    stride=1,
                                    padding=lf_kernel // 2,
                                ).squeeze(0).squeeze(0)
                            elif spatial_signal_mode == "local_variance":
                                tweedie_features = tweedie_estimate[0:1].float()
                                x2 = _F.avg_pool2d(
                                    tweedie_features * tweedie_features,
                                    kernel_size=avg_pool_kernel,
                                    stride=1,
                                    padding=avg_pool_kernel // 2,
                                )
                                xmean = _F.avg_pool2d(
                                    tweedie_features,
                                    kernel_size=avg_pool_kernel,
                                    stride=1,
                                    padding=avg_pool_kernel // 2,
                                )
                                spatial_signal = (x2 - xmean * xmean).clamp_min(0).mean(dim=1).squeeze(0)
                            elif spatial_signal_mode == "coarse_residual":
                                scale = max(1, int(spatial_scale))
                                tweedie_features = tweedie_estimate[0:1].float()
                                if scale == 1:
                                    spatial_signal = torch.zeros_like(tweedie_l2)
                                else:
                                    low_h = max(1, tweedie_features.shape[-2] // scale)
                                    low_w = max(1, tweedie_features.shape[-1] // scale)
                                    x_low = _F.interpolate(tweedie_features, size=(low_h, low_w), mode="area")
                                    x_low = _F.interpolate(
                                        x_low,
                                        size=tweedie_features.shape[-2:],
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                                    spatial_signal = (tweedie_features - x_low).norm(p=2, dim=1).squeeze(0)
                            elif spatial_signal_mode == "multiscale_residual":
                                # Normalize each scale before aggregation to reduce magnitude bias.
                                tweedie_features = tweedie_estimate[0:1].float()
                                H_lat, W_lat = tweedie_features.shape[-2], tweedie_features.shape[-1]
                                ms_residuals = []
                                for ms in [4, 6]:
                                    low_h = max(1, H_lat // ms)
                                    low_w = max(1, W_lat // ms)
                                    x_low = _F.interpolate(tweedie_features, size=(low_h, low_w), mode="area")
                                    x_low = _F.interpolate(x_low, size=(H_lat, W_lat), mode="bilinear", align_corners=False)
                                    diff = tweedie_features - x_low  # (1, C, H, W)
                                    diff_n = diff / (diff.norm(p=2, dim=[2, 3], keepdim=True) + 1e-8)
                                    residual = diff_n.norm(p=2, dim=1).squeeze(0)  # (H, W)
                                    ms_residuals.append(residual)
                                spatial_signal = torch.stack(ms_residuals, dim=0).max(dim=0).values  # (H, W)
                            else:  # local-residual spatial signal
                                local_mean = _F.avg_pool2d(
                                    tweedie_l2.unsqueeze(0).unsqueeze(0),
                                    kernel_size=avg_pool_kernel,
                                    stride=1,
                                    padding=avg_pool_kernel // 2,
                                ).squeeze(0).squeeze(0)
                                spatial_signal = torch.abs(tweedie_l2 - local_mean)
                            spatial_signal = spatial_signal.unsqueeze(0).expand(temporal_signal.shape[0], -1, -1)

                            signal_maps = {
                                "temporal_signal": temporal_signal.detach(),
                                "spatial_signal": spatial_signal.detach(),
                            }


                        self.previous_tweedie_estimate = tweedie_estimate.detach()
                    elif importance_metric == "cfg_tweedie":
                        t_idx = int(t.item()) if torch.is_tensor(t) else int(t)
                        alpha_prod_t = self.scheduler.alphas_cumprod[t_idx].to(latents.device)
                        beta_prod_t = 1 - alpha_prod_t
                        tweedie_estimate = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5 + 1e-8)

                        if self.previous_tweedie_estimate is None:
                            signal_maps = None
                        else:
                            temporal_delta = torch.abs(tweedie_estimate - self.previous_tweedie_estimate)
                            signal_maps = temporal_delta.mean(1)  # (B, H, W)

                        self.previous_tweedie_estimate = tweedie_estimate.detach()
                    else:

                        # Default: CFG Difference
                        signal_maps = torch.abs(noise_pred_text - noise_pred_uncond) 
                        signal_maps = signal_maps.mean(1) # B,h,w 
                    

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)


                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)


        # 4)
        if merge_start_step >= 0 or prune_start_step >= 0:
            remove_patch(self.unet)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
        else:
            image = latents
        
        has_nsfw_concept = None
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)


        # Offload all models
        self.maybe_free_model_hooks()


        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
