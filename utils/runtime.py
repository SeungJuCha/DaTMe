"""Shared runtime helpers for the model generation entry points."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def parse_compression_ratio(value: str) -> float:
    """Parse the fraction of image tokens removed by DaTMe."""
    ratio = float(value)
    if not 0.0 <= ratio < 1.0:
        raise argparse.ArgumentTypeError("compression ratio must be in [0, 1)")
    return ratio


def configure_runtime_backend(deterministic: bool, *, allow_tf32: bool = False) -> None:
    """Configure PyTorch kernels consistently before loading a pipeline."""
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(not deterministic)
        torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32 and not deterministic
        torch.backends.cudnn.allow_tf32 = allow_tf32 and not deterministic

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_everything(seed: int, deterministic: bool) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch and return the pipeline generator."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    return torch.manual_seed(seed)


def load_entries(
    data_path: Optional[Union[str, Path]],
    prompt: Optional[str],
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    """Load the requested JSONL slice or create one custom-prompt entry."""
    if prompt is not None:
        return [{"ann_index": "custom", "caption": prompt}]
    if data_path is None:
        raise ValueError("Provide exactly one input source: --prompt or --data.")

    with open(data_path, "r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle]
    return entries[start:end]
