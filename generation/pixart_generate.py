"""Generate images with the fixed DaTMe method on PixArt-alpha."""

import argparse
import os
import sys
from pathlib import Path

if "--deterministic" in sys.argv:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipe_pixart import DaTMePixArtAlphaPipeline
from utils.output_paths import config_fingerprint, image_filename
from utils.runtime import (
    configure_runtime_backend,
    load_entries,
    parse_compression_ratio,
    seed_everything,
)


OUTDIR_ROOT = PROJECT_ROOT / "outputs" / "pixart"
MODEL_ID = "PixArt-alpha/PixArt-XL-2-1024-MS"
def build_call_kwargs(
    compression_ratio: float,
    merge_layer_start: int = 9,
    merge_layer_end: int = 19,
    merge_start_step: int = 2,
    prune_start_step: int = 1,
    num_spatial_partitions: int = 4,
    spatial_scale: int = 4,
) -> dict:
    """Build pipeline arguments for the finalized PixArt DaTMe configuration."""
    return dict(
        compression_ratio=float(compression_ratio),
        max_downsample=2,
        importance_metric="datme",
        merge_layer_start=merge_layer_start,
        merge_layer_end=merge_layer_end,
        merge_start_step=merge_start_step,
        prune_start_step=prune_start_step,
        num_spatial_partitions=num_spatial_partitions,
        spatial_signal_mode="coarse_residual",
        spatial_scale=spatial_scale,
        ind_mode="ind_topm_toph",
        src_fill_mode="random",
        ind_ratio=0.3,
        feature_based=False,
        selection_axis="joint",
        temporal_metric="tweedie_diff",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images with DaTMe on PixArt")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="JSONL with ann_index and caption fields")
    source.add_argument("--prompt", type=str, help="Generate one image from this prompt")
    parser.add_argument(
        "--compression_ratio",
        type=parse_compression_ratio,
        required=True,
        help="Fraction of image tokens removed by DaTMe, in [0, 1)",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--output_dir", type=Path, default=OUTDIR_ROOT)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic CUDA kernels for exact repeatability",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge_start_step", type=int, default=2, help="First denoising step that performs token merging")
    parser.add_argument("--prune_start_step", type=int, default=1, help="First denoising step that performs token pruning")
    parser.add_argument("--merge_layer_start", type=int, default=9)
    parser.add_argument("--merge_layer_end", type=int, default=19)
    parser.add_argument("--spatial_scale", type=int, default=4, help="Downsampling factor used to compute the spatial signal")
    parser.add_argument(
        "--num_spatial_partitions", type=int, default=4, choices=[1, 4, 16, 64],
        help="Number of spatial regions used to balance source-token selection",
    )
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--scheduler", choices=["dpm", "ddim", "sa-solver", "unipc"], default="dpm")
    parser.add_argument("--disable_resolution_binning", action="store_true")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[+] image output: {outdir}")

    configure_runtime_backend(args.deterministic)
    print(f"[+] loading pipeline: {args.model_id}")
    pipe = DaTMePixArtAlphaPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to(args.device)
    if args.scheduler == "ddim":
        from diffusers import DDIMScheduler

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "sa-solver":
        from diffusers import SASolverScheduler

        pipe.scheduler = SASolverScheduler.from_config(
            pipe.scheduler.config, algorithm_type="data_prediction"
        )
    elif args.scheduler == "unipc":
        from diffusers import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print(f"[+] scheduler: {pipe.scheduler.__class__.__name__}")

    entries = load_entries(args.data, args.prompt, args.start, args.end)

    call_kwargs = build_call_kwargs(
        args.compression_ratio,
        merge_layer_start=args.merge_layer_start,
        merge_layer_end=args.merge_layer_end,
        merge_start_step=args.merge_start_step,
        prune_start_step=args.prune_start_step,
        num_spatial_partitions=args.num_spatial_partitions,
        spatial_scale=args.spatial_scale,
    )
    call_kwargs.update(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        use_resolution_binning=not args.disable_resolution_binning,
    )
    if args.height is not None:
        call_kwargs["height"] = args.height
    if args.width is not None:
        call_kwargs["width"] = args.width
    print(f"[+] pipeline parameters: {call_kwargs}")
    fingerprint = config_fingerprint(
        {
            "model_id": args.model_id,
            "scheduler": args.scheduler,
            "device": args.device,
            "seed": args.seed,
            "deterministic": args.deterministic,
            **call_kwargs,
        }
    )

    n_skip = 0
    n_done = 0
    for entry in entries:
        ann_index = entry["ann_index"]
        caption = str(entry["caption"])
        save_path = outdir / image_filename(
            ann_index, caption, args.compression_ratio, fingerprint
        )
        if save_path.exists():
            n_skip += 1
            continue

        result = pipe(
            prompt=caption,
            generator=seed_everything(args.seed, args.deterministic),
            **call_kwargs,
        )
        result.images[0].save(save_path)
        n_done += 1
        print(f"  [{n_done:>5}] ann={str(ann_index):>6} saved -> {save_path.name}")

    print(f"\n[done] generated: {n_done} / skipped: {n_skip} / total: {n_done + n_skip}")


if __name__ == "__main__":
    main()
