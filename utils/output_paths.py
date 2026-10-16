"""Flat output filenames shared by the model generation entry points."""

import hashlib
import json
from typing import Any, Mapping


def config_fingerprint(config: Mapping[str, Any], length: int = 10) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def image_filename(
    ann_index: Any,
    caption: str,
    compression_ratio: float,
    fingerprint: str,
) -> str:
    image_id = "custom" if str(ann_index) == "custom" else str(ann_index)
    prompt_id = hashlib.sha256(caption.encode("utf-8")).hexdigest()[:10]
    return f"{image_id}_p{prompt_id}_r{compression_ratio}_{fingerprint}.png"
