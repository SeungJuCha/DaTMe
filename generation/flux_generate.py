"""Generate images with the fixed DaTMe method on FLUX.1-dev."""

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

from pipeline.pipe_flux import DaTMeFluxPipeline
from utils.output_paths import config_fingerprint, image_filename
from utils.runtime import (
    configure_runtime_backend,
    load_entries,
    parse_compression_ratio,
    seed_everything,
)


OUTDIR_ROOT = PROJECT_ROOT / "outputs" / "flux"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
N_DOUBLE_BLOCKS = 19
N_SINGLE_BLOCKS = 38


def parse_block_indices(value: str, n_total: int):
    value = value.strip().lower()
    if value in {"all", "none"}:
        return value
    if ":" in value:
        start_text, end_text = value.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else n_total
        return list(range(start, end))
    return [int(index) for index in value.split(",") if index.strip()]


def build_call_kwargs(
    compression_ratio: float,
    double_block_indices,
    single_block_indices,
    *,
    num_inference_steps: int = 35,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    merge_start_step: int = 2,
    num_spatial_partitions: int = 4,
    spatial_scale: int = 4,
) -> dict:
    """Build pipeline arguments for the finalized FLUX DaTMe configuration."""
    return dict(
        compression_ratio=float(compression_ratio),
        double_block_indices=double_block_indices,
        single_block_indices=single_block_indices,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        merge_start_step=merge_start_step,
        num_spatial_partitions=num_spatial_partitions,
        spatial_scale=spatial_scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images with DaTMe on FLUX")
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
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic CUDA kernels for exact repeatability",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Offload inactive pipeline components to CPU to reduce GPU memory use",
    )
    parser.add_argument(
        "--double_blocks",
        default="9:19",
        help="Double-stream block indices: all, none, a range, or comma-separated values",
    )
    parser.add_argument(
        "--single_blocks",
        default="7:22",
        help="Single-stream block indices: all, none, a range, or comma-separated values",
    )
    parser.add_argument("--merge_start_step", type=int, default=2, help="First denoising step that performs token merging")
    parser.add_argument(
        "--num_spatial_partitions", type=int, default=4, choices=[1, 4, 16, 64],
        help="Number of spatial regions used to balance source-token selection",
    )
    parser.add_argument("--spatial_scale", type=int, default=4, help="Downsampling factor used to compute the spatial signal")
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--output_dir", type=Path, default=OUTDIR_ROOT)
    args = parser.parse_args()

    double_blocks = parse_block_indices(args.double_blocks, N_DOUBLE_BLOCKS)
    single_blocks = parse_block_indices(args.single_blocks, N_SINGLE_BLOCKS)
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[+] image output: {outdir}")

    configure_runtime_backend(args.deterministic, allow_tf32=True)
    print(f"[+] loading pipeline: {MODEL_ID}")
    pipe = DaTMeFluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=args.device)
    else:
        pipe = pipe.to(args.device)

    entries = load_entries(args.data, args.prompt, args.start, args.end)

    call_kwargs = build_call_kwargs(
        args.compression_ratio,
        double_blocks,
        single_blocks,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        merge_start_step=args.merge_start_step,
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

        with torch.inference_mode():
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
