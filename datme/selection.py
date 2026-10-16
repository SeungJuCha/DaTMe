"""Spatial SRC budget completion shared by DaTMe model implementations."""

from typing import Tuple

import torch


def grid_shape(num_spatial_partitions: int) -> Tuple[int, int]:
    grid_dim = int(num_spatial_partitions ** 0.5)
    if grid_dim * grid_dim == num_spatial_partitions:
        return grid_dim, grid_dim
    return num_spatial_partitions, 1


def sample_balanced_extra_src_flux(
    candidate_mask: torch.Tensor,
    extra_needed: int,
    h: int,
    w: int,
    num_spatial_partitions: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Randomly complete the SRC budget while balancing spatial regions."""
    selected = torch.zeros_like(candidate_mask)
    if extra_needed <= 0:
        return selected
    candidates = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
    if candidates.numel() < extra_needed:
        raise RuntimeError("Not enough candidate tokens to fill the SRC budget.")
    if num_spatial_partitions <= 1:
        order = torch.randperm(candidates.numel(), generator=generator, device=candidates.device)
        selected[candidates[order[:extra_needed]]] = True
        return selected

    grid_h, grid_w = grid_shape(num_spatial_partitions)
    token_index = torch.arange(h * w, device=candidate_mask.device)
    cell_id = ((token_index // w) * grid_h // h) * grid_w + (
        (token_index % w) * grid_w // w
    )
    per_cell = []
    cursors = []
    base_quota, remainder = divmod(extra_needed, num_spatial_partitions)
    for cell in range(num_spatial_partitions):
        local = candidates[cell_id[candidates] == cell]
        if local.numel():
            local = local[
                torch.randperm(local.numel(), generator=generator, device=local.device)
            ]
        per_cell.append(local)
        quota = min(base_quota + (cell < remainder), local.numel())
        selected[local[:quota]] = True
        cursors.append(int(quota))

    left = extra_needed - int(selected.sum())
    while left > 0:
        active = [i for i, local in enumerate(per_cell) if cursors[i] < local.numel()]
        if not active:
            break
        active_tensor = torch.tensor(active, device=candidate_mask.device)
        order = torch.randperm(
            active_tensor.numel(), generator=generator, device=candidate_mask.device
        )
        for cell in active_tensor[order].tolist():
            selected[per_cell[cell][cursors[cell]]] = True
            cursors[cell] += 1
            left -= 1
            if left == 0:
                break
    if left:
        raise RuntimeError("Spatial balancing could not fill the SRC budget.")
    return selected


def sample_balanced_extra_src(
    candidate_mask: torch.Tensor,
    extra_needed: int,
    h: int,
    w: int,
    num_spatial_partitions: int,
    generator: torch.Generator,
) -> torch.Tensor:
    selected_mask = torch.zeros_like(candidate_mask)
    if extra_needed <= 0:
        return selected_mask

    candidate_idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
    if candidate_idx.numel() < extra_needed:
        raise RuntimeError("Not enough middle-pool tokens to fill SRC target.")

    if num_spatial_partitions <= 1:
        rand_perm = torch.randperm(candidate_idx.numel(), generator=generator, device=candidate_idx.device)
        selected_mask[candidate_idx[rand_perm[:extra_needed]]] = True
        return selected_mask

    grid_h, grid_w = grid_shape(num_spatial_partitions)
    token_idx = torch.arange(h * w, device=candidate_mask.device)
    t_row = token_idx // w
    t_col = token_idx % w
    cell_row = (t_row * grid_h) // h
    cell_col = (t_col * grid_w) // w
    cell_id = cell_row * grid_w + cell_col

    interval_candidates = []
    interval_cursor = []
    base_quota = extra_needed // num_spatial_partitions
    remainder = extra_needed % num_spatial_partitions

    for interval_id in range(num_spatial_partitions):
        local_idx = candidate_idx[cell_id[candidate_idx] == interval_id]
        if local_idx.numel() > 0:
            perm = torch.randperm(local_idx.numel(), generator=generator, device=local_idx.device)
            local_idx = local_idx[perm]
        interval_candidates.append(local_idx)
        interval_cursor.append(0)

        quota = base_quota + (1 if interval_id < remainder else 0)
        take = min(quota, local_idx.numel())
        if take > 0:
            selected_mask[local_idx[:take]] = True
            interval_cursor[-1] = take

    selected_count = int(selected_mask.sum().item())
    leftover = extra_needed - selected_count

    while leftover > 0:
        active = [
            interval_id
            for interval_id in range(num_spatial_partitions)
            if interval_cursor[interval_id] < interval_candidates[interval_id].numel()
        ]
        if not active:
            break

        active_tensor = torch.tensor(active, device=candidate_mask.device, dtype=torch.long)
        active_perm = torch.randperm(active_tensor.numel(), generator=generator, device=candidate_mask.device)
        active_order = active_tensor[active_perm].tolist()

        took_any = False
        for interval_id in active_order:
            cursor = interval_cursor[interval_id]
            candidates = interval_candidates[interval_id]
            if cursor >= candidates.numel():
                continue
            selected_mask[candidates[cursor]] = True
            interval_cursor[interval_id] += 1
            leftover -= 1
            took_any = True
            if leftover == 0:
                break

        if not took_any:
            break

    if leftover > 0:
        remaining_idx = (candidate_mask & ~selected_mask).nonzero(as_tuple=False).squeeze(-1)
        if remaining_idx.numel() < leftover:
            raise RuntimeError("Grid redistribution failed to fill SRC target.")
        perm = torch.randperm(remaining_idx.numel(), generator=generator, device=remaining_idx.device)
        selected_mask[remaining_idx[perm[:leftover]]] = True

    return selected_mask


def _rank01(score: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(score)
    rank = torch.empty_like(score, dtype=torch.float32)
    denom = max(score.numel() - 1, 1)
    rank[order] = torch.arange(score.numel(), device=score.device, dtype=torch.float32) / denom
    return rank


def sample_low_joint_extra_src(
    candidate_mask: torch.Tensor,
    extra_needed: int,
    h: int,
    w: int,
    num_spatial_partitions: int,
    temporal_score: torch.Tensor,
    spatial_score: torch.Tensor,
) -> torch.Tensor:
    """Fill the SRC budget with tokens low on both DaTMe signals."""
    selected_mask = torch.zeros_like(candidate_mask)
    if extra_needed <= 0:
        return selected_mask

    candidate_idx = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
    if candidate_idx.numel() < extra_needed:
        raise RuntimeError("Not enough middle-pool tokens to fill SRC target.")

    joint_score = _rank01(temporal_score.float()) + _rank01(spatial_score.float())

    def take_lowest(indices: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0 or indices.numel() == 0:
            return indices[:0]
        order = torch.argsort(joint_score[indices])
        return indices[order[: min(count, indices.numel())]]

    if num_spatial_partitions <= 1:
        selected_mask[take_lowest(candidate_idx, extra_needed)] = True
        return selected_mask

    grid_h, grid_w = grid_shape(num_spatial_partitions)
    token_idx = torch.arange(h * w, device=candidate_mask.device)
    t_row = token_idx // w
    t_col = token_idx % w
    cell_row = (t_row * grid_h) // h
    cell_col = (t_col * grid_w) // w
    cell_id = cell_row * grid_w + cell_col

    base_quota = extra_needed // num_spatial_partitions
    remainder = extra_needed % num_spatial_partitions
    for interval_id in range(num_spatial_partitions):
        local_idx = candidate_idx[cell_id[candidate_idx] == interval_id]
        quota = base_quota + (1 if interval_id < remainder else 0)
        selected_mask[take_lowest(local_idx, quota)] = True

    selected_count = int(selected_mask.sum().item())
    leftover = extra_needed - selected_count
    if leftover > 0:
        remaining_idx = (candidate_mask & ~selected_mask).nonzero(as_tuple=False).squeeze(-1)
        selected_mask[take_lowest(remaining_idx, leftover)] = True

    if int(selected_mask.sum().item()) != extra_needed:
        raise RuntimeError("Low-joint grid fill failed to fill SRC target.")

    return selected_mask
