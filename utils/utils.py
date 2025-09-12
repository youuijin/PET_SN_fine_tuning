import torch, random
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def set_seed(seed=0):
    """
    Set seed before training

    Parameters:
    - seed: selected seed (int)
    """
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Apply deformation into image
def normalize_deformation(disp_field):
    """
    Normalize deformation field: [-Size, Size] -> [-1, 1]

    Parameters:
    - disp_field: 3D displacement field (torch.Tensor) (shape: B, 3, D, H, W)
    """
    D, H, W = disp_field.shape[2:]

    norm_disp = torch.zeros_like(disp_field)
    norm_disp[:, 0] = disp_field[:, 0] / (W / 2)
    norm_disp[:, 1] = disp_field[:, 1] / (H / 2)
    norm_disp[:, 2] = disp_field[:, 2] / (D / 2)

    return norm_disp

def apply_deformation_using_disp(img, displace_field, mode='bilinear'):
    """
    Deform image using displacement field (need to add identity matrix)

    Parameters:
    - img: to be transformed (torch.Tensor) (shape: B, 1, D, H, W)
    - displace_field: 3D displacement field (torch.Tensor) (shape: B, 3, D, H, W)
        x : W, y : H, z : D
    - mode: interpolation mode, ['bilinear', 'nearest']
    """
    assert mode in ['bilinear', 'nearest']

    B, _, D, H, W = img.shape
    
    # Generate normalized grid (-1, 1)
    d = torch.linspace(-1, 1, D)
    h = torch.linspace(-1, 1, H)
    w = torch.linspace(-1, 1, W)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    grid = torch.stack((grid_w, grid_h, grid_d), dim=0) # (3, D, H, W)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1).to(img.device) # (B, 3, D, H, W)

    # Apply deformation
    normalized_disp = normalize_deformation(displace_field)

    deformed_grid = grid + normalized_disp

    # Reshape grid to match F.grid_sample requirements (B, D, H, W, 3)
    deformed_grid = deformed_grid.permute(0, 2, 3, 4, 1)

    # Perform deformation
    deformed_img = F.grid_sample(img, deformed_grid, mode=mode, padding_mode='border', align_corners=True)

    return deformed_img

def apply_deformation(img, deformation_field, mode='bilinear'):
    """
    Deform image using deformation field

    Parameters:
    - img: to be transformed (torch.Tensor) (shape: B, 1, D, H, W)
    - deformation_field: 3D deformation field (torch.Tensor) (shape: B, 3, D, H, W)
        x : W, y : H, z : D
    - mode: interpolation mode, ['bilinear', 'nearest']
    """
    assert mode in ['bilinear', 'nearest']

    B, _, D, H, W = img.shape

    # Using another normalization
    deformed_grid = torch.empty_like(deformation_field)
    # deformed_grid[:, 0] = 2 * (deformation_field[:, 0] / (W - 1) - 0.5)
    # deformed_grid[:, 1] = 2 * (deformation_field[:, 1] / (H - 1) - 0.5)
    # deformed_grid[:, 2] = 2 * (deformation_field[:, 2] / (D - 1) - 0.5)

    # Reshape grid to match F.grid_sample requirements (B, D, H, W, 3)
    deformed_grid = deformed_grid.permute(0, 2, 3, 4, 1)

    # Perform deformation
    deformed_img = F.grid_sample(img, deformed_grid, mode=mode, padding_mode='border', align_corners=True)

    return deformed_img

# Save 3D or 2D image
def save_disp_RGB(disp_field, output_path, affine):
    """
    Save displacement field into RGB colored image

    Parameters:
    - disp_field: 3D displacement field (torch.Tensor) (shape: 1, 3, D, H, W)
    - output_path: save nifti file path (str)
    - affine: affine matrix to save nii format (shape: 4, 4)
    """
    disp_field = disp_field[0] # select first one of batch
    
    if not isinstance(disp_field, np.ndarray):
        if isinstance(disp_field, torch.Tensor):
            disp_field = disp_field.detach().cpu().numpy()
        else:
            disp_field = np.array(disp_field)
    disp_field = disp_field.squeeze()
    if disp_field.shape[-1] != 3:
        disp_field = np.transpose(disp_field, (1, 2, 3, 0))

    # displacement field: (X, Y, Z, 3)

    # abs (only magnitude)
    disp_field = np.abs(disp_field)

    # 정규화 및 uint8 변환
    disp_field_min = disp_field.min(axis=(0,1,2), keepdims=True)
    disp_field_max = disp_field.max(axis=(0,1,2), keepdims=True)
    disp_field_norm = ((disp_field - disp_field_min) / (disp_field_max - disp_field_min + 1e-8))

    rgb_data = disp_field_norm[..., np.newaxis, :]  # shape: (X, Y, Z, 1, 3)
    rgb_data = rgb_data.astype(np.float32)

    hdr = nib.nifti1.Nifti1Header()
    hdr.set_data_dtype(np.float32)       # header datatype
    hdr['dim'] = [5, 192, 224, 192, 1, 3, 1, 1]  # 정확히 맞추기
    hdr['intent_code'] = 1007            # NIFTI_INTENT_VECTOR

    nii_img = nib.Nifti1Image(rgb_data, affine=affine, header=hdr)
    nib.save(nii_img, output_path)

def denormalize_image(normalized_img, img_min, img_max):
    """
    Denormalize: [0, 1] -> [img_min, img_max]

    Parameters:
    - normalized_img: 3D tensor in [0, 1] (torch.Tensor) (shape: B, 1, D, H, W)
    - img_min, img_max: original min, max to denormalize (shape: B, 1)
    """
    img_min, img_max = img_min.to(normalized_img.device), img_max.to(normalized_img.device)
    return normalized_img * (img_max - img_min) + img_min

def save_deformed_image_nii(deformed_tensor, output_path, img_min, img_max, affine):
    """
    Save deformed tensor image to .nii.gz format

    Parameters:
    - deformed_tensor: 3D tensor (torch.Tensor) (shape: 1, 1, D, H, W)
    - output_path: save nifti file path (str)
    - img_min, img_max: original min, max to denormalize (shape: B, 1)
    - affine: affine matrix to save nii format (shape: 4, 4)
    """
    # 1. convert tensor to numpy
    deformed_tensor = denormalize_image(deformed_tensor, img_min, img_max)
    deformed_img = deformed_tensor.squeeze().cpu().detach().numpy()

    # 4. 새로운 NIfTI 객체 생성 및 저장
    deformed_nifti = nib.Nifti1Image(deformed_img, affine=affine)
    nib.save(deformed_nifti, output_path)
    

# save images
def transform_slice(img):
    # apply 90-degree CCW rotation + horizontal flip
    return np.fliplr(np.rot90(img, k=1))
    
def save_middle_slices(img_3d, epoch, idx):
    """
    img_3d: [D, H, W] or [1, D, H, W] or [B, 1, D, H, W] (e.g., torch.Tensor)
    Returns: matplotlib Figure with x, y, z middle slices side-by-side
    """
    img_3d = img_3d.squeeze().detach().cpu().numpy()

    D, H, W = img_3d.shape

    slice_x = transform_slice(img_3d[D // 2, :, :])
    slice_y = transform_slice(img_3d[:, H // 2, :])
    slice_z = transform_slice(img_3d[:, :, W // 2])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(slice_z, cmap='gray')
    axes[0].set_title('Axial (X)')
    axes[1].imshow(slice_y, cmap='gray')
    axes[1].set_title('Coronal (Y)')
    axes[2].imshow(slice_x, cmap='gray')
    axes[2].set_title('Sagittal (Z)')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    return fig

def save_middle_slices_mfm(moving, fixed, moved, epoch, idx):
    """
    img_3d: [D, H, W] or [1, D, H, W] or [B, 1, D, H, W] (e.g., torch.Tensor)
    Returns: matplotlib Figure with x, y, z middle slices side-by-side
    """
    def get_slices(img):
        img = img.squeeze().detach().cpu().numpy()
        D, H, W = img.shape
        return [
            transform_slice(img[D // 2, :, :]),  # axial (x)
            transform_slice(img[:, H // 2, :]),  # coronal (y)
            transform_slice(img[:, :, W // 2]),  # sagittal (z)
        ]
    
    slices_moving = get_slices(moving)
    slices_fixed = get_slices(fixed)
    slices_moved = get_slices(moved)
    
    titles = ['Axial (X)', 'Coronal (Y)', 'Sagittal (Z)']
    labels = ['Moving', 'Fixed', 'Moved']

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for row in range(3):
        for col, slices in enumerate([slices_moving, slices_fixed, slices_moved]):
            axes[row, col].imshow(slices[row], cmap='gray')
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(labels[col], fontsize=12)
        # 왼쪽 row label
        axes[row, 0].text(-0.2, 0.5, titles[row], va='center', ha='right',
                          rotation=90, transform=axes[row, 0].transAxes, fontsize=12)

    plt.suptitle(f'Middle slices at Epoch {epoch}, Sample {idx}', fontsize=14)
    plt.tight_layout()
    return fig

def print_with_timestamp(string=""):
    timestamp = (datetime.now() + timedelta(hours=9)).strftime("[%Y-%m-%d %H:%M:%S]")
    print(timestamp, string)