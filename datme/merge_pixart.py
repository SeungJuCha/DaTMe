"""DaTMe token partitioning and coordinate matching for PixArt."""

from typing import Callable, Optional, Tuple

import torch

from .selection import sample_balanced_extra_src, sample_low_joint_extra_src
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

def _select_extreme_mask(
    score: torch.Tensor,
    count: int,
    *,
    largest: bool,
    blocked_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select exactly ``count`` extrema, optionally excluding blocked tokens."""
    mask = torch.zeros_like(score, dtype=torch.bool)
    if count <= 0:
        return mask

    if blocked_mask is None:
        candidates = torch.arange(score.numel(), device=score.device)
    else:
        candidates = (~blocked_mask).nonzero(as_tuple=False).squeeze(-1)

    if candidates.numel() < count:
        raise RuntimeError(
            f"Cannot select {count} tokens from only {candidates.numel()} available candidates."
        )

    selected = torch.topk(score[candidates], k=count, largest=largest).indices
    mask[candidates[selected]] = True
    return mask



def matching_datme_pixart(
    metric: torch.Tensor, 
    w: int, h: int, r: int,
    generator: torch.Generator = None,
    signal_maps: dict = None,
    add_rand_small: bool = False,
    **kwargs
) -> Tuple[Callable, Callable]:
    """Partition PixArt tokens using DaTMe's temporal and spatial signals.

    The selected indices are shared between unconditional and conditional CFG
    batches. Source tokens are matched to nearby destination coordinates.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        temporal_signal = signal_maps["temporal_signal"]
        spatial_signal = signal_maps["spatial_signal"]

        signal_batch = temporal_signal.shape[0]
        assert signal_batch == B // 2 or signal_batch == B

        # Flatten both signals to the PixArt image-token sequence.
        temporal_flat = _flatten_map(temporal_signal, h, w, metric.device)
        spatial_flat = _flatten_map(spatial_signal, h, w, metric.device)

        temporal_score = temporal_flat[0, :, 0]
        spatial_score = spatial_flat[0, :, 0]

        r_target = min(int(r), N - 1)
        if r_target <= 0:
            return do_nothing, do_nothing
        
        # 2. Build the native two-axis partition first. The one-axis ablations
        # use these actual core/IND counts, rather than the compression ratio,
        # so their selected low/high sets are cardinality-matched to the joint set.
        ind_mode = kwargs.get("ind_mode", "no_ind")
        selection_axis = kwargs.get("selection_axis", "joint")
        if selection_axis not in (
            "joint",
            "temporal_only_matched_counts",
            "spatial_only_matched_counts",
            "temporal_only_direct",
            "spatial_only_direct",
            "rank_sum",
        ):
            raise ValueError(f"Unsupported selection_axis: {selection_axis}")

        low_k = min(r_target, N)
        if ind_mode == "ind_adaptive":
            # Match SD semantics: ind_ratio controls only top-M/top-H IND candidates.
            ind_ratio = float(kwargs.get("ind_ratio", 0.3))
            ind_ratio = max(0.0, min(ind_ratio, 1.0))
            high_k = min(max(1, int(round(N * ind_ratio))), N)
        else:
            high_k = min(max(1, N - r_target), N)

        low_temporal_mask = _select_extreme_mask(
            temporal_score, low_k, largest=False
        )
        low_spatial_mask = _select_extreme_mask(
            spatial_score, low_k, largest=False
        )
        high_temporal_mask = _select_extreme_mask(
            temporal_score, high_k, largest=True
        )
        high_spatial_mask = _select_extreme_mask(
            spatial_score, high_k, largest=True
        )

        use_protected_dst = ind_mode == "protected_dst"
        joint_high_mask = high_temporal_mask & high_spatial_mask
        joint_protected_mask = (
            joint_high_mask.clone()
            if use_protected_dst
            else torch.zeros_like(low_temporal_mask)
        )
        joint_src_core_mask = (
            low_temporal_mask & low_spatial_mask
        ) & ~joint_protected_mask
        joint_remaining_mask = ~joint_src_core_mask
        joint_ind_mask = torch.zeros(N, dtype=torch.bool, device=metric.device)

        if ind_mode != "no_ind" and not use_protected_dst:
            if ind_mode in ("ind_topm_toph", "ind_adaptive"):
                joint_ind_mask = joint_high_mask & joint_remaining_mask
            elif ind_mode == "ind_lowm_toph":
                joint_ind_mask = (
                    low_temporal_mask & high_spatial_mask & joint_remaining_mask
                )
            elif ind_mode == "ind_topm_lowh":
                joint_ind_mask = (
                    high_temporal_mask & low_spatial_mask & joint_remaining_mask
                )
            else:
                raise ValueError(f"Unsupported ind_mode: {ind_mode}")

        if ind_mode != "no_ind" and not use_protected_dst:
            max_ind = max(0, N - r_target - 1)
            joint_ind_count = int(joint_ind_mask.sum().item())
            if joint_ind_count > max_ind:
                ind_candidates = joint_ind_mask.nonzero(as_tuple=False).squeeze(-1)
                constraint_score = (
                    temporal_score[ind_candidates] + spatial_score[ind_candidates]
                )
                keep_idx = ind_candidates[
                    torch.topk(constraint_score, k=max_ind, largest=True).indices
                ]
                joint_ind_mask.zero_()
                if keep_idx.numel() > 0:
                    joint_ind_mask[keep_idx] = True

        joint_src_core_count = int(joint_src_core_mask.sum().item())
        joint_ind_count = int(joint_ind_mask.sum().item())
        joint_protected_count = int(joint_protected_mask.sum().item())


        if selection_axis == "joint":
            src_core_mask = joint_src_core_mask
            ind_mask = joint_ind_mask
            protected_mask = joint_protected_mask
            axis_score = None
        elif selection_axis == "rank_sum":
            # Rank-sum ablation (Reviewer SJAg Q1): collapse the two maps into a
            # single ordering (sum of per-axis ranks). The lowest r tokens become
            # SRC and the highest r become IND (constraint); the middle is the DST
            # pool. This hits the SRC budget exactly, so no grid completion runs.
            temporal_rank = torch.argsort(torch.argsort(temporal_score))
            spatial_rank = torch.argsort(torch.argsort(spatial_score))
            rank_sum_score = (temporal_rank + spatial_rank).to(temporal_score.dtype)
            protected_mask = torch.zeros_like(joint_protected_mask)
            src_core_mask = _select_extreme_mask(rank_sum_score, r_target, largest=False)
            ind_mask = torch.zeros_like(joint_ind_mask)
            if ind_mode != "no_ind":
                ind_mask = _select_extreme_mask(
                    rank_sum_score, r_target, largest=True, blocked_mask=src_core_mask
                )
            axis_score = None
        elif selection_axis in (
            "temporal_only_matched_counts",
            "spatial_only_matched_counts",
        ):
            axis_score = (
                temporal_score
                if selection_axis == "temporal_only_matched_counts"
                else spatial_score
            )
            protected_mask = torch.zeros_like(joint_protected_mask)
            if use_protected_dst:
                protected_mask = _select_extreme_mask(axis_score, joint_protected_count, largest=True)

            src_core_mask = _select_extreme_mask(
                axis_score,
                joint_src_core_count,
                largest=False,
                blocked_mask=protected_mask,
            )
            blocked_for_ind = src_core_mask | protected_mask
            ind_mask = torch.zeros_like(joint_ind_mask)
            if ind_mode != "no_ind" and not use_protected_dst:
                ind_largest = True
                if ind_mode == "ind_lowm_toph" and selection_axis == "temporal_only_matched_counts":
                    ind_largest = False
                elif ind_mode == "ind_topm_lowh" and selection_axis == "spatial_only_matched_counts":
                    ind_largest = False
                ind_mask = _select_extreme_mask(
                    axis_score,
                    joint_ind_count,
                    largest=ind_largest,
                    blocked_mask=blocked_for_ind,
                )
        else:
            # Direct one-axis ablation: do not inherit either the joint
            # intersection cardinalities or its grid-completion behavior.
            # The one score alone defines the full partition:
            #   bottom-r -> SRC, top-r -> IND, remaining tokens -> DST.
            # This intentionally forces a visibly distinct one-axis selection.
            if 2 * r_target >= N:
                raise ValueError(
                    "Direct one-axis selection requires compression_ratio < 0.5 "
                    "so that at least one DST token remains."
                )

            axis_score = (
                temporal_score
                if selection_axis == "temporal_only_direct"
                else spatial_score
            )
            protected_mask = torch.zeros_like(joint_protected_mask)
            src_core_mask = _select_extreme_mask(axis_score, r_target, largest=False)
            ind_mask = _select_extreme_mask(
                axis_score,
                r_target,
                largest=True,
                blocked_mask=src_core_mask,
            )
        src_count = int(src_core_mask.sum().item())
        extra_src_needed = r_target - src_count
        middle_mask = ~(src_core_mask | ind_mask | protected_mask)

        num_spatial_partitions = kwargs.get("num_spatial_partitions", 1)
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
                # A one-axis ablation must not reintroduce the other axis while
                # filling the remaining source budget.
                fill_temporal_score = (
                    temporal_score if axis_score is None else axis_score
                )
                fill_spatial_score = (
                    spatial_score if axis_score is None else axis_score
                )
                extra_src_mask = sample_low_joint_extra_src(
                    candidate_mask=middle_mask,
                    extra_needed=extra_src_needed,
                    h=h,
                    w=w,
                    num_spatial_partitions=num_spatial_partitions,
                    temporal_score=fill_temporal_score,
                    spatial_score=fill_spatial_score,
                )
            else:
                raise ValueError(f"Unsupported src_fill_mode: {src_fill_mode}")

        src_mask = src_core_mask | extra_src_mask
        dst_mask = ~(src_mask | ind_mask)

        num_dst = int(dst_mask.sum().item())
        
        indep_idx = ind_mask.nonzero(as_tuple=False).view(1, -1, 1)
        a_idx = src_mask.nonzero(as_tuple=False).view(1, -1, 1)
        b_idx = dst_mask.nonzero(as_tuple=False).view(1, -1, 1)

        if a_idx.shape[1] != r_target:
            raise RuntimeError(f"Expected SRC count {r_target}, got {a_idx.shape[1]}")

        # 4. Copy -> unconditional & conditional (CFG)
        if indep_idx.shape[0] < B:
            repeat_factor = B // indep_idx.shape[0]
            indep_idx = torch.cat([indep_idx] * repeat_factor, dim=0)
            a_idx = torch.cat([a_idx] * repeat_factor, dim=0)
            b_idx = torch.cat([b_idx] * repeat_factor, dim=0)
	            
        assert a_idx.shape == (B, r_target, 1) and b_idx.shape == (B, num_dst, 1)

        def split(x):
            C = x.shape[-1]
            indep = gather(x, dim=1, index=indep_idx.expand(-1, -1, C))
            src = gather(x, dim=1, index=a_idx.expand(-1, -1, C))
            dst = gather(x, dim=1, index=b_idx.expand(-1, -1, C))
            return indep, src, dst

        # 5. Coordinate metric between A and B
        _, a, b = split(metric)
        assert a.shape[1] >= r_target

        # raw_scores shape: (B, r, num_dst)
        feature_based = kwargs.get("feature_based", False)
        if feature_based or metric.shape[-1] != 2:
            a_norm = a.float() / a.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
            b_norm = b.float() / b.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scores = a_norm @ b_norm.transpose(-1, -2)  # per-batch cosine sim (uncond/cond separate)
        else:
            scores = - torch.cdist(a.float(), b.float(), p=2) # coordinate distance
        
        # for each src token, find its most similar dst token
        node_max, node_idx = scores.max(dim=-1) 

        # Add a tiny jitter when coordinate distances produce equal edge scores.
        original_tome = kwargs.get("original_tome", False)
        if add_rand_small and not original_tome:
            small_randomness = torch.rand_like(node_max[:max(1, B//2)]) * 1e-5
            small_randomness = torch.cat([small_randomness] * (B // small_randomness.shape[0]), dim=0)
            node_max = node_max + small_randomness

        # All selected SRC tokens are merged; the unmerged SRC slice is empty.
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B, r, 1
        
        unm_idx = edge_idx[..., r_target:, :]  # B, 0, 1  (Empty since we separated exactly r)
        src_idx = edge_idx[..., :r_target, :]  # B, r, 1
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx) # B, r, 1


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        indep, src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r_target, c))
        if mode != "none":
            src = gather(src, dim=-2, index=src_idx.expand(n, r_target, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r_target, c), src, reduce=mode)

        return torch.cat([indep, unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        indep_len = indep_idx.shape[1]
        unm_len = unm_idx.shape[1] # 0
        indep = x[..., :indep_len, :]
        unm = x[..., indep_len:indep_len + unm_len, :]
        dst = x[..., indep_len + unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r_target, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=indep_idx.expand(B, indep_len, c), src=indep)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r_target, c), src=src)

        return out

    return merge, unmerge
