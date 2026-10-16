"""Core token selection, merge, and unmerge operations for DaTMe."""

from typing import Callable, Tuple

import torch
import torch.nn.functional as F

from .selection import sample_balanced_extra_src_flux
from .utils import chunked_score_max, do_nothing, mps_gather_workaround


def _flatten_map(
    score_map: torch.Tensor, h: int, w: int, device: torch.device
) -> torch.Tensor:
    """Convert BHW, B1HW, or BN score maps to B,N,1."""
    score_map = score_map.to(device=device, dtype=torch.float32)
    if score_map.ndim == 4:
        if score_map.shape[1] == 1:
            score_map = score_map[:, 0]
        elif score_map.shape[-1] == 1:
            score_map = score_map[..., 0]
        else:
            score_map = score_map.mean(dim=1)
    if score_map.ndim == 3:
        if score_map.shape[1:] != (h, w):
            score_map = F.interpolate(
                score_map.unsqueeze(1), size=(h, w), mode="area"
            ).squeeze(1)
        return score_map.reshape(score_map.shape[0], h * w, 1)
    if score_map.ndim == 2:
        if score_map.shape[1] != h * w:
            raise ValueError(
                f"Expected {h * w} flattened scores, got {score_map.shape[1]}."
            )
        return score_map.unsqueeze(-1)
    raise ValueError(f"Unsupported score-map shape: {tuple(score_map.shape)}")






def matching_datme(
    metric: torch.Tensor,
    w: int,
    h: int,
    r: int,
    generator: torch.Generator = None,
    num_spatial_partitions: int = 4,
    ind_mode: str = "no_ind",
    signal_maps: dict = None,
    **kwargs,
) -> Tuple[Callable, Callable]:
    """Build the fixed DaTMe matching used by FLUX."""
    batch_size, token_count, _ = metric.shape
    if r <= 0:
        return do_nothing, do_nothing
    if ind_mode not in {"no_ind", "ind_topm_toph"}:
        raise ValueError(f"Unsupported finalized IND mode: {ind_mode}")
    if not isinstance(signal_maps, dict):
        return do_nothing, do_nothing
    temporal_signal = signal_maps.get("temporal_signal")
    spatial_signal = signal_maps.get("spatial_signal")
    if temporal_signal is None or spatial_signal is None:
        return do_nothing, do_nothing
    if generator is None:
        generator = torch.Generator(device=metric.device)
        generator.manual_seed(torch.initial_seed())

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    remove_count = min(int(r), token_count - 1)
    cached = kwargs.get("_cached_indices")
    cache_store = kwargs.get("_index_cache_store")

    with torch.no_grad():
        if cached is None:
            temporal = _flatten_map(temporal_signal, h, w, metric.device)[0, :, 0]
            spatial = _flatten_map(spatial_signal, h, w, metric.device)[0, :, 0]
            low_temporal = torch.zeros(token_count, dtype=torch.bool, device=metric.device)
            low_spatial = torch.zeros_like(low_temporal)
            low_temporal[torch.topk(temporal, remove_count, largest=False).indices] = True
            low_spatial[torch.topk(spatial, remove_count, largest=False).indices] = True
            src_core = low_temporal & low_spatial

            ind_mask = torch.zeros_like(src_core)
            if ind_mode == "ind_topm_toph":
                high_count = min(max(1, token_count - remove_count), token_count)
                high_temporal = torch.zeros_like(src_core)
                high_spatial = torch.zeros_like(src_core)
                high_temporal[
                    torch.topk(temporal, high_count, largest=True).indices
                ] = True
                high_spatial[
                    torch.topk(spatial, high_count, largest=True).indices
                ] = True
                ind_mask = high_temporal & high_spatial & ~src_core
                max_ind = max(0, token_count - remove_count - 1)
                if int(ind_mask.sum()) > max_ind:
                    candidates = ind_mask.nonzero(as_tuple=False).squeeze(-1)
                    joint = temporal[candidates] + spatial[candidates]
                    keep = candidates[torch.topk(joint, max_ind, largest=True).indices]
                    ind_mask.zero_()
                    ind_mask[keep] = True

            extra_src = sample_balanced_extra_src_flux(
                ~(src_core | ind_mask),
                remove_count - int(src_core.sum()),
                h,
                w,
                num_spatial_partitions,
                generator,
            )
            src_mask = src_core | extra_src
            dst_mask = ~(src_mask | ind_mask)
            if not dst_mask.any():
                return do_nothing, do_nothing
            indep_idx = ind_mask.nonzero(as_tuple=False).view(1, -1, 1)
            src_idx_all = src_mask.nonzero(as_tuple=False).view(1, -1, 1)
            dst_idx_all = dst_mask.nonzero(as_tuple=False).view(1, -1, 1)
            indep_idx = indep_idx.expand(batch_size, -1, -1)
            src_idx_all = src_idx_all.expand(batch_size, -1, -1)
            dst_idx_all = dst_idx_all.expand(batch_size, -1, -1)
            if cache_store is not None:
                cache_store.update(
                    indep_idx=indep_idx,
                    a_idx=src_idx_all,
                    b_idx=dst_idx_all,
                    r_target=remove_count,
                )
        else:
            indep_idx = cached["indep_idx"]
            src_idx_all = cached["a_idx"]
            dst_idx_all = cached["b_idx"]
            remove_count = cached["r_target"]

        num_dst = dst_idx_all.shape[1]

        def split(tensor: torch.Tensor):
            channels = tensor.shape[-1]
            indep = gather(tensor, 1, indep_idx.expand(-1, -1, channels))
            src = gather(tensor, 1, src_idx_all.expand(-1, -1, channels))
            dst = gather(tensor, 1, dst_idx_all.expand(-1, -1, channels))
            return indep, src, dst

        _, src_metric, dst_metric = split(metric)
        if metric.shape[-1] == 2:
            scores = -torch.cdist(src_metric.float(), dst_metric.float(), p=2)
        else:
            src_norm = src_metric.float() / src_metric.float().norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            dst_norm = dst_metric.float() / dst_metric.float().norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            scores = src_norm @ dst_norm.transpose(-1, -2)
        dst_for_src = scores.argmax(dim=-1, keepdim=True)

    def merge(tensor: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        indep, src, dst = split(tensor)
        if mode != "none":
            dst = dst.scatter_reduce(
                -2,
                dst_for_src.expand(tensor.shape[0], remove_count, tensor.shape[-1]),
                src,
                reduce="mean",
            )
        return torch.cat([indep, dst], dim=1)

    def unmerge(tensor: torch.Tensor) -> torch.Tensor:
        indep_len = indep_idx.shape[1]
        indep, dst = tensor[:, :indep_len], tensor[:, indep_len:]
        channels = tensor.shape[-1]
        src = gather(
            dst, -2, dst_for_src.expand(batch_size, remove_count, channels)
        )
        output = torch.zeros(
            batch_size, token_count, channels, device=tensor.device, dtype=tensor.dtype
        )
        output.scatter_(1, indep_idx.expand(-1, -1, channels), indep)
        output.scatter_(1, dst_idx_all.expand(-1, num_dst, channels), dst)
        output.scatter_(1, src_idx_all.expand(-1, remove_count, channels), src)
        return output

    return merge, unmerge


def bipartite_soft_matching_random2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    r: int,
    sx: int = 2,
    sy: int = 2,
    no_rand: bool = False,
    generator: torch.Generator = None,
    **_kwargs,
) -> Tuple[Callable, Callable]:
    """Prune SD/SDXL tokens with feature based bipartite matching."""
    batch_size, token_count, _ = metric.shape
    if r <= 0:
        return do_nothing, do_nothing
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        grid_h, grid_w = h // sy, w // sx
        if no_rand:
            random_index = torch.zeros(
                grid_h, grid_w, 1, device=metric.device, dtype=torch.int64
            )
        else:
            random_index = torch.randint(
                sy * sx,
                (grid_h, grid_w, 1),
                device=generator.device,
                generator=generator,
            ).to(metric.device)
        index_view = torch.zeros(
            grid_h, grid_w, sy * sx, device=metric.device, dtype=torch.int64
        )
        index_view.scatter_(2, random_index, -torch.ones_like(random_index))
        index_view = (
            index_view.view(grid_h, grid_w, sy, sx)
            .transpose(1, 2)
            .reshape(grid_h * sy, grid_w * sx)
        )
        if grid_h * sy < h or grid_w * sx < w:
            index_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            index_buffer[: grid_h * sy, : grid_w * sx] = index_view
        else:
            index_buffer = index_view
        order = index_buffer.reshape(1, -1, 1).argsort(dim=1)
        num_dst = grid_h * grid_w
        src_idx_all = order[:, num_dst:].expand(batch_size, -1, -1)
        dst_idx_all = order[:, :num_dst].expand(batch_size, -1, -1)

        def split(tensor: torch.Tensor):
            channels = tensor.shape[-1]
            src = gather(tensor, 1, src_idx_all.expand(-1, -1, channels))
            dst = gather(tensor, 1, dst_idx_all.expand(-1, -1, channels))
            return src, dst

        normalized = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        src_metric, dst_metric = split(normalized)
        remove_count = min(src_metric.shape[1], int(r))
        if src_metric.shape[1] <= 9216:
            scores = src_metric @ dst_metric.transpose(-1, -2)
            node_max, node_idx = scores.max(dim=-1)
        else:
            node_max, node_idx = chunked_score_max(
                src_metric, dst_metric, chunk_size=max(1, src_metric.shape[1] // 8)
            )
        edge_order = node_max.argsort(dim=-1, descending=True).unsqueeze(-1)
        unmerged_idx = edge_order[:, remove_count:]
        merged_idx = edge_order[:, :remove_count]
        dst_for_src = gather(node_idx.unsqueeze(-1), -2, merged_idx)

    def merge(tensor: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = split(tensor)
        channels = tensor.shape[-1]
        unmerged = gather(
            src, -2, unmerged_idx.expand(batch_size, src.shape[1] - remove_count, channels)
        )
        if mode != "none":
            merged = gather(src, -2, merged_idx.expand(batch_size, remove_count, channels))
            dst = dst.scatter_reduce(
                -2,
                dst_for_src.expand(batch_size, remove_count, channels),
                merged,
                reduce="mean",
            )
        return torch.cat([unmerged, dst], dim=1)

    def unmerge(tensor: torch.Tensor) -> torch.Tensor:
        unmerged_len = unmerged_idx.shape[1]
        unmerged, dst = tensor[:, :unmerged_len], tensor[:, unmerged_len:]
        channels = tensor.shape[-1]
        reconstructed = gather(
            dst, -2, dst_for_src.expand(batch_size, remove_count, channels)
        )
        output = torch.zeros(
            batch_size, token_count, channels, device=tensor.device, dtype=tensor.dtype
        )
        output.scatter_(1, dst_idx_all.expand(-1, -1, channels), dst)
        output.scatter_(
            1,
            gather(src_idx_all, 1, unmerged_idx).expand(-1, -1, channels),
            unmerged,
        )
        output.scatter_(
            1,
            gather(src_idx_all, 1, merged_idx).expand(-1, -1, channels),
            reconstructed,
        )
        return output

    return merge, unmerge
