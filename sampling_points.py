import torch
import torch.nn.functional as F


def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)

    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


@torch.no_grad()
def sampling_points(coarse, N, k=7, beta=0.75, training=True):
    assert coarse.dim() == 4, "Dim must be N(Batch)CHW"
    device = coarse.device
    B, _, H, W = coarse.shape
    mask, _ = coarse.sort(1, descending=True)

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + torch.div(idx, W, rounding_mode='trunc').to(torch.float) * H_step
        return idx, points

    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)

    uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    certain_map = over_generation_map[:, 0] - over_generation_map[:, 1]
    _, idx1 = uncertainty_map.topk(int(2 * N), -1)
    _, idx2 = certain_map.topk(int((k - 1) * N), -1)
    random_indices = torch.randperm(idx2.size(1))[:int((1 - beta) * N)]
    idx2_2 = idx2[:, random_indices]

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)
    idx1 += shift[:, None]
    idx2_2 += shift[:, None]

    importance = over_generation.view(-1, 2)[idx1.view(-1), :].view(B, int(beta * N), 2)
    coverage = over_generation.view(-1, 2)[idx2_2.view(-1), :].view(B, N - int(beta * N), 2)
    return torch.cat([importance, coverage], 1).to(device)
