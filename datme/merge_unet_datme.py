"""DaTMe token partitioning and matching for SD and SDXL U-Nets."""

from typing import Callable, Tuple

import torch

from .selection import grid_shape, sample_balanced_extra_src, sample_low_joint_extra_src
from .utils import do_nothing, mps_gather_workaround


def _flatten_map(score_map: torch.Tensor, h: int, w: int, device: torch.device) -> torch.Tensor:
    score_map = score_map.to(device=device, dtype=torch.float32)

    if score_map.dim() == 4:
        if score_map.shape[1] == 1:
            score_map = score_map[:, 0]
        elif score_map.shape[-1] == 1:
            score_map = score_map[..., 0]
        else:
            score_map = score_map.mean(dim=1)

    if score_map.dim() == 3:
        if score_map.shape[1:3] != (h, w):
            score_map = torch.nn.functional.interpolate(
                score_map.unsqueeze(1),
                size=(h, w),
                mode="area",
            ).squeeze(1)
        return score_map.reshape(score_map.shape[0], h * w, 1)

    if score_map.dim() == 2:
        if score_map.shape[1] != h * w:
            raise ValueError(f"Expected flattened score map with {h*w} tokens, got {score_map.shape}")
        return score_map.unsqueeze(-1)

    raise ValueError(f"Unsupported score map shape: {tuple(score_map.shape)}")












def matching_datme_unet(
    metric: torch.Tensor,
    w: int,
    h: int,
    r: int,
    generator: torch.Generator = None,
    local_match: bool = False,
    num_spatial_partitions: int = 4,
    ind_mode: str = "ind_topm_toph",
    signal_maps: dict = None,
    **kwargs,
) -> Tuple[Callable, Callable]:
    """Partition tokens with DaTMe's temporal and spatial signals.

    Low values on both signals form the safe-to-merge SRC core. Depending on
    ``ind_mode``, high values form the protected constraint set. Spatially
    balanced sampling fills any remaining SRC budget; all other tokens are DST.
    """
    B, N, _ = metric.shape
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    if r <= 0:
        return do_nothing, do_nothing

    if not isinstance(signal_maps, dict):
        return do_nothing, do_nothing

    temporal_signal = signal_maps.get("temporal_signal")
    spatial_signal = signal_maps.get("spatial_signal")
    if temporal_signal is None or spatial_signal is None:
        return do_nothing, do_nothing

    with torch.no_grad():
        r_target = min(int(r), N - 1)
        if r_target <= 0:
            return do_nothing, do_nothing


        # ── Index cache: reuse SRC/DST/IND selection across blocks in same step ──
        _cached = kwargs.get("_cached_indices")  # dict or None
        _cache_store = kwargs.get("_index_cache_store")  # dict to write into, or None

        if _cached is not None:
            # Subsequent block in same step: load pre-computed indices
            indep_idx     = _cached["indep_idx"]
            a_idx         = _cached["a_idx"]
            b_idx         = _cached["b_idx"]
            same_cell     = _cached["same_cell"]
            neighbor_cell = _cached.get("neighbor_cell")
            r_target      = _cached["r_target"]
            num_dst       = b_idx.shape[1]
        else:
            # First block: compute SRC/DST/IND selection from scratch
            temporal_flat = _flatten_map(temporal_signal, h, w, metric.device)
            spatial_flat = _flatten_map(spatial_signal, h, w, metric.device)

            temporal_score = temporal_flat[0, :, 0]
            spatial_score = spatial_flat[0, :, 0]

            low_k = min(r_target, N)
            if ind_mode == "ind_adaptive":
                # ind_ratio controls only the top-M/top-H IND candidate pool.
                # SRC core remains the compression-ratio-sized low-M/low-H set.
                ind_ratio = float(kwargs.get("ind_ratio", 0.3))
                ind_ratio = max(0.0, min(ind_ratio, 1.0))
                high_k = min(max(1, int(round(N * ind_ratio))), N)
            else:
                high_k = min(max(1, N - r_target), N)

            low_temporal_idx = torch.topk(
                temporal_score, k=low_k, largest=False
            ).indices
            low_spatial_idx = torch.topk(
                spatial_score, k=low_k, largest=False
            ).indices

            base_mask = torch.zeros(N, dtype=torch.bool, device=metric.device)
            low_temporal_mask = torch.zeros_like(base_mask)
            low_spatial_mask = torch.zeros_like(base_mask)
            low_temporal_mask[low_temporal_idx] = True
            low_spatial_mask[low_spatial_idx] = True

            ind_mask = torch.zeros(N, dtype=torch.bool, device=metric.device)
            use_protected_dst = ind_mode == "protected_dst"
            high_temporal_mask = torch.zeros(N, dtype=torch.bool, device=metric.device)
            high_spatial_mask = torch.zeros(N, dtype=torch.bool, device=metric.device)
            top_select_k = min(high_k, N)
            if top_select_k > 0:
                high_temporal_mask[
                    torch.topk(temporal_score, k=top_select_k, largest=True).indices
                ] = True
                high_spatial_mask[
                    torch.topk(spatial_score, k=top_select_k, largest=True).indices
                ] = True

            protected_mask = (
                high_temporal_mask & high_spatial_mask
                if use_protected_dst
                else torch.zeros_like(ind_mask)
            )
            src_core_mask = (low_temporal_mask & low_spatial_mask) & ~protected_mask

            remaining_mask = ~src_core_mask

            if ind_mode != "no_ind" and not use_protected_dst:
                if ind_mode in ("ind_topm_toph", "ind_adaptive"):
                    ind_mask = high_temporal_mask & high_spatial_mask & remaining_mask
                elif ind_mode == "ind_lowm_toph":
                    ind_mask = low_temporal_mask & high_spatial_mask & remaining_mask
                elif ind_mode == "ind_topm_lowh":
                    ind_mask = high_temporal_mask & low_spatial_mask & remaining_mask
                else:
                    raise ValueError(f"Unsupported ind_mode: {ind_mode}")

            # Always leave at least one DST token.
            if ind_mode != "no_ind" and not use_protected_dst:
                max_ind = max(0, N - r_target - 1)
                ind_count = int(ind_mask.sum().item())
                if ind_count > max_ind:
                    ind_candidates = ind_mask.nonzero(as_tuple=False).squeeze(-1)
                    constraint_score = (
                        temporal_score[ind_candidates] + spatial_score[ind_candidates]
                    )
                    keep_idx = ind_candidates[
                        torch.topk(constraint_score, k=max_ind, largest=True).indices
                    ]
                    ind_mask.zero_()
                    if keep_idx.numel() > 0:
                        ind_mask[keep_idx] = True

            src_count = int(src_core_mask.sum().item())
            extra_src_needed = r_target - src_count
            middle_mask = remaining_mask & ~ind_mask & ~protected_mask

            extra_src_mask = torch.zeros(N, dtype=torch.bool, device=metric.device)
            if extra_src_needed > 0:
                src_fill_mode = kwargs.get("src_fill_mode", "random")
                if src_fill_mode == "random":
                    extra_src_mask = sample_balanced_extra_src(
                        candidate_mask=middle_mask,
                        extra_needed=extra_src_needed,
                        h=h,
                        w=w,
                        num_spatial_partitions=num_spatial_partitions,
                        generator=generator,
                    )
                elif src_fill_mode == "low_joint":
                    extra_src_mask = sample_low_joint_extra_src(
                        candidate_mask=middle_mask,
                        extra_needed=extra_src_needed,
                        h=h,
                        w=w,
                        num_spatial_partitions=num_spatial_partitions,
                        temporal_score=temporal_score,
                        spatial_score=spatial_score,
                    )
                else:
                    raise ValueError(f"Unsupported src_fill_mode: {src_fill_mode}")

            src_mask = src_core_mask | extra_src_mask
            dst_mask = ~(src_mask | ind_mask)

            if int(dst_mask.sum().item()) == 0:
                move_candidates = ind_mask.nonzero(as_tuple=False).squeeze(-1)
                if move_candidates.numel() == 0:
                    return do_nothing, do_nothing
                move_score = (
                    temporal_score[move_candidates] + spatial_score[move_candidates]
                )
                move_idx = move_candidates[torch.argmin(move_score)]
                ind_mask[move_idx] = False
                dst_mask[move_idx] = True

            indep_idx = ind_mask.nonzero(as_tuple=False).view(1, -1, 1)
            a_idx = src_mask.nonzero(as_tuple=False).view(1, -1, 1)
            b_idx = dst_mask.nonzero(as_tuple=False).view(1, -1, 1)

            if a_idx.shape[1] != r_target:
                raise RuntimeError(f"Expected SRC count {r_target}, got {a_idx.shape[1]}")

            if indep_idx.shape[0] < B:
                repeat_factor = B // indep_idx.shape[0]
                indep_idx = torch.cat([indep_idx] * repeat_factor, dim=0)
                a_idx = torch.cat([a_idx] * repeat_factor, dim=0)
                b_idx = torch.cat([b_idx] * repeat_factor, dim=0)

            same_cell = None
            neighbor_cell = None
            if local_match and num_spatial_partitions > 1:
                grid_h, grid_w = grid_shape(num_spatial_partitions)
                token_idx = torch.arange(N, device=metric.device)
                t_row = token_idx // w
                t_col = token_idx % w
                cell_row = (t_row * grid_h) // h
                cell_col = (t_col * grid_w) // w

                # Map SRC and DST tokens to spatial partition coordinates.
                src_grid_row = cell_row[a_idx[:, :, 0]]   # (B, r)
                src_grid_col = cell_col[a_idx[:, :, 0]]
                dst_grid_row = cell_row[b_idx[:, :, 0]]   # (B, num_dst)
                dst_grid_col = cell_col[b_idx[:, :, 0]]

                # manhattan distance between SRC and DST cells
                manhattan = (
                    (src_grid_row.unsqueeze(-1) - dst_grid_row.unsqueeze(-2)).abs() +
                    (src_grid_col.unsqueeze(-1) - dst_grid_col.unsqueeze(-2)).abs()
                )  # (B, r, num_dst)
                same_cell     = manhattan == 0
                neighbor_cell = manhattan == 1

            num_dst = b_idx.shape[1]

            # Write to cache for subsequent blocks in this step
            if _cache_store is not None:
                _cache_store["indep_idx"]    = indep_idx
                _cache_store["a_idx"]        = a_idx
                _cache_store["b_idx"]        = b_idx
                _cache_store["same_cell"]    = same_cell
                _cache_store["neighbor_cell"] = neighbor_cell
                _cache_store["r_target"]     = r_target

        perm_idx = torch.cat([indep_idx, a_idx, b_idx], dim=1)

        def split(x: torch.Tensor):
            c = x.shape[-1]
            permuted = gather(x, dim=1, index=perm_idx.expand(-1, -1, c))
            indep_len = indep_idx.shape[1]
            src_len = a_idx.shape[1]
            src_end = indep_len + src_len

            indep = permuted[:, :indep_len]
            src = permuted[:, indep_len:src_end]
            dst = permuted[:, src_end:]
            return indep, src, dst

        is_coord = (metric.shape[-1] == 2)
        if not is_coord:
            metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        indep_feat, src_feat, dst_feat = split(metric)
        real_r = src_feat.shape[1]

        if is_coord:
            # Coordinate metric: nearby destinations receive higher scores.
            raw_scores = -torch.cdist(src_feat.float(), dst_feat.float(), p=2)
        else:
            raw_scores = src_feat @ dst_feat.transpose(-1, -2)
        scores = raw_scores
        if same_cell is not None:
            # Prefer the same partition, then adjacent partitions, then global matching.
            effective_mask = same_cell.clone()

            no_same = ~same_cell.any(dim=-1, keepdim=True)       # (B, r, 1)
            if no_same.any() and neighbor_cell is not None:
                effective_mask = effective_mask | (no_same & neighbor_cell)

            no_neighbor = ~effective_mask.any(dim=-1, keepdim=True)  # (B, r, 1)
            if no_neighbor.any():
                effective_mask = effective_mask | no_neighbor

            scores = raw_scores.masked_fill(~effective_mask, float("-inf"))
        node_max, node_idx = scores.max(dim=-1)


        dst_idx_in_b = node_idx.unsqueeze(-1)


    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        indep, src, dst = split(x)
        n, _, c = src.shape
        unm_candidates = torch.zeros(n, 0, c, device=x.device, dtype=x.dtype)

        if mode != "none":
            dst = dst.scatter_reduce(
                -2,
                dst_idx_in_b.expand(n, real_r, c),
                src,
                reduce="mean",
            )

        return torch.cat([indep, unm_candidates, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        indep_len = indep_idx.shape[1]
        indep = x[..., :indep_len, :]
        dst = x[..., indep_len:, :]

        _, _, c = dst.shape
        src_reconstructed = gather(dst, dim=-2, index=dst_idx_in_b.expand(B, real_r, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=indep_idx.expand(B, indep_len, c), src=indep)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=a_idx.expand(B, real_r, c), src=src_reconstructed)
        return out

    return merge, unmerge
