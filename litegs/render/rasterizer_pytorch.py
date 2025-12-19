import torch
import math


@torch.no_grad()
def gaussian_radius_from_inv_cov2d(inv_cov2d: torch.Tensor) -> torch.Tensor:
    """
    inv_cov2d: [P, 2, 2]
    radius ≈ 3 * sqrt(1 / min_eigenvalue(inv_cov2d))
    """
    a = inv_cov2d[:, 0, 0]
    b = inv_cov2d[:, 0, 1]
    c = inv_cov2d[:, 1, 1]

    trace = a + c
    det = a * c - b * b
    disc = (trace * trace - 4 * det).clamp(min=0.0)

    lambda_min = 0.5 * (trace - torch.sqrt(disc))
    lambda_min = lambda_min.clamp(min=1e-6)

    return 3.0 * torch.sqrt(1.0 / lambda_min).ceil()


def rasterize_pytorch_view(
    means2D: torch.Tensor,    # [P, 2]
    depths: torch.Tensor,     # [P]
    inv_cov2d: torch.Tensor,  # [P, 2, 2]
    color: torch.Tensor,      # [P, C]
    opacity: torch.Tensor,    # [P, 1] or [P]
    H: int,
    W: int,
    white_background: bool = True,
):
    device = means2D.device
    P = means2D.shape[0]

    if P == 0:
        bg = 1.0 if white_background else 0.0
        img = torch.full((H, W, color.shape[-1]), bg, device=device)
        depth = torch.zeros((H, W, 1), device=device)
        alpha = torch.zeros((H, W, 1), device=device)
        return img, depth, alpha

    # Ensure opacity shape
    if opacity.dim() == 1:
        opacity = opacity.unsqueeze(-1)

    # ---------- sort near → far ----------
    sort_idx = torch.argsort(depths, descending=False)
    depths = depths[sort_idx]
    means2D = means2D[sort_idx]
    inv_cov2d = inv_cov2d[sort_idx]
    color = color[sort_idx]
    opacity = opacity[sort_idx]

    # ---------- radius estimation ----------
    radii = gaussian_radius_from_inv_cov2d(inv_cov2d)

    MAX_RADIUS = 64
    radii = radii.clamp(max=MAX_RADIUS)

    # ---------- output buffers ----------
    bg = 1.0 if white_background else 0.0
    img = torch.full((H, W, color.shape[-1]), bg, device=device)
    depth_img = torch.zeros((H, W, 1), device=device)
    alpha_img = torch.zeros((H, W, 1), device=device)

    C_flat = img.view(-1, color.shape[-1])
    D_flat = depth_img.view(-1, 1)
    A_flat = alpha_img.view(-1, 1)

    MAX_PATCH_PIXELS = 4096  # hard safety cap

    # ---------- raster loop ----------
    for i in range(P):
        r = int(radii[i].item())
        if r < 1:
            continue

        mean = means2D[i]
        inv_cov = inv_cov2d[i]
        col = color[i]
        op = opacity[i]
        z = depths[i]

        xmin = max(0, int(mean[0] - r))
        xmax = min(W, int(mean[0] + r))
        ymin = max(0, int(mean[1] - r))
        ymax = min(H, int(mean[1] + r))

        if xmax <= xmin or ymax <= ymin:
            continue

        patch_w = xmax - xmin
        patch_h = ymax - ymin
        if patch_w * patch_h > MAX_PATCH_PIXELS:
            continue

        # local pixel coordinates
        xs = torch.arange(xmin, xmax, device=device)
        ys = torch.arange(ymin, ymax, device=device)
        coords = torch.stack(
            torch.meshgrid(xs, ys, indexing="xy"),
            dim=-1
        ).reshape(-1, 2)

        d = coords - mean[None, :]

        gx = (
            d[:, 0] * (inv_cov[0, 0] * d[:, 0] + inv_cov[0, 1] * d[:, 1]) +
            d[:, 1] * (inv_cov[1, 0] * d[:, 0] + inv_cov[1, 1] * d[:, 1])
        )

        weight = torch.exp(-0.5 * gx).unsqueeze(-1)
        alpha = weight * op

        flat_idx = (coords[:, 1] * W + coords[:, 0]).long()

        # early termination
        T = 1.0 - A_flat[flat_idx]
        if (T <= 1e-4).all():
            continue

        contrib = T * alpha

        C_flat[flat_idx] = C_flat[flat_idx] * (1.0 - contrib) + contrib * col
        D_flat[flat_idx] += contrib * z
        A_flat[flat_idx] += contrib

    return img, depth_img, alpha_img
