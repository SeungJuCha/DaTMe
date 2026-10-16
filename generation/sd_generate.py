"""Generate images with the fixed DaTMe method on Stable Diffusion 1.5."""

import argparse
import os
import sys
from pathlib import Path

if "--deterministic" in sys.argv:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from diffusers import DDIMScheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipe_sd import DaTMeStableDiffusionPipeline
from utils.output_paths import config_fingerprint, image_filename
from utils.runtime import (
    configure_runtime_backend,
    load_entries,
    parse_compression_ratio,
    seed_everything,
)


OUTDIR_ROOT = PROJECT_ROOT / "outputs" / "sd"
MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
def build_call_kwargs(
    compression_ratio: float,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    merge_start_step: int = 2,
    prune_start_step: int = 1,
    num_spatial_partitions: int = 4,
    spatial_scale: int = 2,
) -> dict:
    """Build pipeline arguments for the finalized SD DaTMe configuration."""
    return dict(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        compression_ratio=float(compression_ratio),
        use_signal_guided_merging=True,
        importance_metric="datme",
        max_downsample=1,
        merge_start_step=merge_start_step,
        prune_start_step=prune_start_step,
        num_spatial_partitions=num_spatial_partitions,
        ind_mode="no_ind",
        spatial_signal_mode="coarse_residual",
        spatial_scale=spatial_scale,
        src_fill_mode="random",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images with DaTMe on SD 1.5")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="JSONL with ann_index and caption fields")
    source.add_argument("--prompt", type=str, help="Generate one image from this prompt")
    parser.add_argument(
        "--compression_ratio",
        type=parse_compression_ratio,
        required=True,
        help="Fraction of image tokens removed by DaTMe, in [0, 1)",
    )
    parser.add_argument("--start", type=int, default=0, help="First JSONL row (inclusive)")
    parser.add_argument("--end", type=int, default=10000, help="Last JSONL row (exclusive)")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic CUDA kernels for exact repeatability",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num_spatial_partitions", type=int, default=4, choices=[1, 4, 16, 64],
        help="Number of spatial regions used to balance source-token selection",
    )
    parser.add_argument("--merge_start_step", type=int, default=2, help="First denoising step that performs token merging")
    parser.add_argument("--prune_start_step", type=int, default=1, help="First denoising step that performs token pruning")
    parser.add_argument("--spatial_scale", type=int, default=2, help="Downsampling factor used to compute the spatial signal")
    parser.add_argument("--output_dir", "--outdir", dest="output_dir", type=Path, default=OUTDIR_ROOT)
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[+] image output: {outdir}")

    configure_runtime_backend(args.deterministic)
    print(f"[+] loading pipeline: {MODEL_ID}")
    ddim = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe = DaTMeStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        scheduler=ddim,
    ).to(args.device)

    entries = load_entries(args.data, args.prompt, args.start, args.end)

    call_kwargs = build_call_kwargs(
        args.compression_ratio,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        merge_start_step=args.merge_start_step,
        prune_start_step=args.prune_start_step,
        num_spatial_partitions=args.num_spatial_partitions,
        spatial_scale=args.spatial_scale,
    )
    print(f"[+] pipeline parameters: {call_kwargs}")
    fingerprint = config_fingerprint(
        {
            "model_id": MODEL_ID,
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

        image = pipe(
            prompt=caption,
            generator=seed_everything(args.seed, args.deterministic),
            **call_kwargs,
        ).images[0]
        image.save(save_path)
        n_done += 1
        print(f"  [{n_done:>5}] ann={str(ann_index):>6} saved -> {save_path.name}")

    print(f"\n[done] generated: {n_done} / skipped: {n_skip} / total: {n_done + n_skip}")


if __name__ == "__main__":
    main()
