import torch
import torch.nn.functional as F

from utils.utils import normalize_deformation

# Loss functions
## similarity - NCC
def NCC_loss(x, y, eps=1e-8):
    """
    Calculate the normalized cross correlation (NCC) between two tensors.
    Args:
    - x (torch.Tensor): Deformed image tensor (B, 1, D, H, W)
    - y (torch.Tensor): Template image tensor (B, 1, D, H, W)
    - eps (float): Small constant to avoid division by zero.

    Returns:
    - NCC loss value (torch.Tensor)
    """
    x_mean = torch.mean(x, dim=[2,3,4], keepdim=True).to(x.device)
    y_mean = torch.mean(y, dim=[2,3,4], keepdim=True).to(y.device)

    x = x - x_mean
    y = y - y_mean

    numerator = torch.sum(x * y, dim=[2,3,4])
    denominator = torch.sqrt(torch.sum(x ** 2, dim=[2,3,4]) * torch.sum(y ** 2, dim=[2,3,4]) + eps)

    ncc = numerator / denominator
    return -torch.mean(ncc)  # Maximize NCC by minimizing the negative value

## similarity - Local NCC
def local_NCC_loss(x, y, win_size=9, eps=1e-8):
    """
    Calculate the normalized cross correlation (NCC) between two tensors using Local Patch.
    
    Args:
    - x (torch.Tensor): Deformed image tensor (B, 1, D, H, W)
    - y (torch.Tensor): Template image tensor (B, 1, D, H, W)
    - win_size (int): Size of local window.
    - eps (float): Small constant to avoid division by zero.

    Returns:
    - NCC loss value (torch.Tensor)
    """
    ndims = len(x.shape) - 2  # Check if 2D/3D
    sum_filt = torch.ones([1, 1] + [win_size] * ndims).to(x.device)  # Kernel for conv

    # Compute local mean
    pad = win_size // 2
    x_mean = F.conv3d(x, sum_filt, padding=pad, stride=1) / (win_size ** ndims)
    y_mean = F.conv3d(y, sum_filt, padding=pad, stride=1) / (win_size ** ndims)

    # Subtract local mean
    x = x - x_mean
    y = y - y_mean

    # Compute numerator (cross-correlation)
    numerator = F.conv3d(x * y, sum_filt, padding=pad, stride=1)

    # Compute denominator (variance normalization)
    x_var = F.conv3d(x ** 2, sum_filt, padding=pad, stride=1)
    y_var = F.conv3d(y ** 2, sum_filt, padding=pad, stride=1)
    denominator = torch.sqrt(x_var * y_var + eps)

    # Compute Local Patch NCC
    ncc = numerator / denominator
    return -torch.mean(ncc)  # Minimize negative NCC for loss

## similarity - MSE
def MSE_loss(x, y):
    mse = torch.mean(torch.norm(x - y, dim=1, p=2) ** 2)
    return mse

def gaussian_window_3d(window_size, sigma, channel):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None, None] * g[None, :, None] * g[None, None, :]
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    return window.repeat(channel, 1, 1, 1, 1)

def SSIM_loss(x, y, window_size=7, sigma=1.5, eps=1e-8):
    """
    Structural Similarity Index (SSIM) for 3D images
    Args:
    - x, y: (B, 1, D, H, W)
    """
    C = x.shape[1]
    window = gaussian_window_3d(window_size, sigma, C).to(x.device)

    mu_x = F.conv3d(x, window, padding=window_size//2, groups=C)
    mu_y = F.conv3d(y, window, padding=window_size//2, groups=C)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x = F.conv3d(x * x, window, padding=window_size//2, groups=C) - mu_x2
    sigma_y = F.conv3d(y * y, window, padding=window_size//2, groups=C) - mu_y2
    sigma_xy = F.conv3d(x * y, window, padding=window_size//2, groups=C) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2) + eps)

    return ssim_map.mean()

## regularization - l1 norm
def l1_loss(displace_field):
    return torch.norm(displace_field, p=1)

## regularization - l2 norm
def l2_loss(displace_field):
    return torch.mean(displace_field ** 2)

## regularization - Total Variation
def tv_loss(displace):
    """
    displace: Tensor of shape [B, 3, D, H, W]
    TV loss는 인접 voxel 간의 L1 차이의 평균을 구하는 방식입니다.
    """
    displace = normalize_deformation(displace)
    # Depth 방향 차이 (D axis)
    dz = torch.abs(displace[:, :, 1:, :, :] - displace[:, :, :-1, :, :])
    # Height 방향 차이 (H axis)
    dy = torch.abs(displace[:, :, :, 1:, :] - displace[:, :, :, :-1, :])
    # Width 방향 차이 (W axis)
    dx = torch.abs(displace[:, :, :, :, 1:] - displace[:, :, :, :, :-1])
    
    loss_z = torch.mean(dz)
    loss_y = torch.mean(dy)
    loss_x = torch.mean(dx)
    
    loss = (loss_z + loss_y + loss_x)/3
    return loss

## regularization - Jacobian Determiant
def jac_det_loss(displace_field):
    # displace -> deformation grid
    B, _, D, H, W = displace_field.shape

    # denormalize (scaling)
    disp = displace_field.clone()
    disp[:, 0, ...] *= (W - 1) / 2
    disp[:, 1, ...] *= (H - 1) / 2
    disp[:, 2, ...] *= (D - 1) / 2
    
    d_range = torch.arange(D, device=displace_field.device)
    h_range = torch.arange(H, device=displace_field.device)
    w_range = torch.arange(W, device=displace_field.device)
    d_grid, h_grid, w_grid = torch.meshgrid(d_range, h_range, w_range, indexing='ij')
    grid = torch.stack((w_grid, h_grid, d_grid), dim=0).float()  # (D, H, W, 3)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)

    deformation_field = grid + disp

    dz = deformation_field[:, :, 1:, :-1, :-1] - deformation_field[:, :, :-1, :-1, :-1]
    dy = deformation_field[:, :, :-1, 1:, :-1] - deformation_field[:, :, :-1, :-1, :-1]
    dx = deformation_field[:, :, :-1, :-1, 1:] - deformation_field[:, :, :-1, :-1, :-1]

    # (B, 3, D, H, W)
    J = torch.stack((dx, dy, dz), dim=-1)  # (B, 3, D, H, W, 3)

    # --- 4. Jacobian Determinant 계산 ---
    # For each voxel, compute 3x3 determinant
    J = J.permute(0, 2, 3, 4, 1, 5)  # (B, D, H, W, 3, 3)
    detJ = (J[..., 0, 0] * (J[..., 1, 1] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 1]) -
            J[..., 0, 1] * (J[..., 1, 0] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 0]) +
            J[..., 0, 2] * (J[..., 1, 0] * J[..., 2, 1] - J[..., 1, 1] * J[..., 2, 0]))

    jacobian_loss = torch.mean(F.relu(-detJ))  # Penalize negative determinants

    return jacobian_loss

## regularization - Total Variation
def tv_loss_l2(displace, std=None, p=None):
    """
    displace: Tensor of shape [B, 3, D, H, W]
    TV loss는 인접 voxel 간의 L2 차이의 평균을 구하는 방식입니다.
    """
    # Depth 방향 차이 (D axis)
    dz = torch.mean((displace[:, :, 1:, :, :] - displace[:, :, :-1, :, :])**2)
    # Height 방향 차이 (H axis)
    dy = torch.mean((displace[:, :, :, 1:, :] - displace[:, :, :, :-1, :])**2)
    # Width 방향 차이 (W axis)
    dx = torch.mean((displace[:, :, :, :, 1:] - displace[:, :, :, :, :-1])**2)

    loss = (dx + dy + dz)/3.
    return loss.mean()

## regularization - Adaptive Total Variation (using uncertainty)
def adaptive_tv_loss_l2(phi, std, eps=1e-6):
    """
    Adaptive total variation loss based on inverse σ².
    phi: [B, 3, D, H, W]
    std: [B, 3, D, H, W]  (std = exp(0.5 * log_sigma))
    """
    dz = (phi[:, :, 1:, :, :] - phi[:, :, :-1, :, :])**2
    dy = (phi[:, :, :, 1:, :] - phi[:, :, :, :-1, :])**2
    dx = (phi[:, :, :, :, 1:] - phi[:, :, :, :, :-1])**2

    sigma_z = std[:, :, 1:, :, :]
    sigma_y = std[:, :, :, 1:, :]
    sigma_x = std[:, :, :, :, 1:]

    weight_z = 1.0 / (sigma_z**2 + eps)
    weight_y = 1.0 / (sigma_y**2 + eps)
    weight_x = 1.0 / (sigma_x**2 + eps)

    loss_z = (weight_z * dz).mean()
    loss_y = (weight_y * dy).mean()
    loss_x = (weight_x * dx).mean()

    return (loss_z + loss_y + loss_x) / 3.0
