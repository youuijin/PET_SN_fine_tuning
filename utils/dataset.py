import torch, os
import nibabel as nib
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv, random

def set_datapath(dataset, numpy):
    if numpy:
        return './data/FDG_PET_percent_numpy', None
    else:
        return './data/FDG_PET_percent', None


def set_dataloader_usingcsv(dataset, csv_dir, template_path, batch_size, numpy=False, return_path=False):
    if numpy:
        train_file = f'{csv_dir}/{dataset}/{dataset}_train_numpy.csv'
        valid_file = f'{csv_dir}/{dataset}/{dataset}_valid_numpy.csv'
    else:
        train_file = f'{csv_dir}/{dataset}/{dataset}_train.csv'
        valid_file = f'{csv_dir}/{dataset}/{dataset}_valid.csv'
    train_dataset = MedicalImageDatasetCSV(train_file, template_path, numpy=numpy, return_path=return_path)
    val_dataset = MedicalImageDatasetCSV(valid_file, template_path, numpy=numpy, return_path=return_path)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader

def set_paired_dataloader_usingcsv(dataset, csv_dir, batch_size=1, numpy=True, return_path=False, return_mask=False, mask_path=None):
    # No paired (random pairing)
    if numpy:
        train_file = f'{csv_dir}/{dataset}/{dataset}_train_numpy.csv'
    else:
        train_file = f'{csv_dir}/{dataset}/{dataset}_train.csv'
    with open(train_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    train_image_paths = [s[0] for s in rows]
    train_dataset = RandomInterPatientDataset(train_image_paths, numpy=numpy, return_path=return_path)
    
    # Predefined paired (fixed pairing)
    if numpy:
        valid_file = f'{csv_dir}/{dataset}/{dataset}_valid_pair_numpy.csv'
    else:
        valid_file = f'{csv_dir}/{dataset}/{dataset}_valid_pair.csv'
    with open(valid_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        valid_image_paths = list(reader)

    val_dataset = FixedPairDataset(valid_image_paths, numpy=numpy, return_path=return_path)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader

# Define dataset class
class MedicalImageDatasetCSV(Dataset):
    def __init__(self, csv_path, template_path, numpy=False, transform=None, return_path=False):
        # if return_mask and mask_path is None:
        #     return ValueError('If you want to use brain mask, Enter mask path.')
        
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        self.image_paths = [s[0] for s in rows]
        self.numpy = numpy
        if numpy:
            template = np.load(template_path)
        else:
            template = nib.load(template_path).get_fdata().astype(np.float32)
        
        # Template normalize - percentile
        t_data = template.flatten()
        p1_temp = np.percentile(t_data, 1)
        p99_temp = np.percentile(t_data, 99)
        template = np.clip(template, p1_temp, p99_temp)
        
        template_min, template_max = template.min(), template.max()
        self.template = (template - template_min) / (template_max - template_min)

        self.transform = transform
        self.return_path = return_path
        self.seg_path = 'data/FDG_label_cortex_mask'

        # load segments
        self.temp_seg = []
        for i in range(6):
            seg_path = f"{self.seg_path}/template_T1w_MRI/mask{i+1}.nii.gz"
            seg = nib.load(seg_path).get_fdata().astype(np.float32)
            self.temp_seg.append(torch.from_numpy(seg))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.numpy:
            img = np.load(self.image_paths[idx])
            affine = np.load('data/affine.npy')
        else:
            img = nib.load(self.image_paths[idx])
            affine = img.affine
            img = img.get_fdata().astype(np.float32)

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#

        # Load segments
        # load segments
        img_seg = []
        for i in range(6):
            seg_path = f"{self.seg_path}/{self.image_paths[idx].split('/')[-1].split('_')[1]}_T1w_MRI/mask{i+1}.nii.gz"
            seg = nib.load(seg_path).get_fdata().astype(np.float32)
            img_seg.append(torch.from_numpy(seg))

        if self.transform is not None:
            img = self.transform(img)

        # return format
        ## img, template, img_min, img_max, affine matrix, img_segs, temp_segs
        if self.return_path:
            return torch.from_numpy(img), torch.from_numpy(self.template), img_min, img_max, affine, self.image_paths[idx]
 
        return torch.from_numpy(img), torch.from_numpy(self.template), img_min, img_max, affine, img_seg, self.temp_seg

class RandomInterPatientDataset(Dataset):
    def __init__(self, image_paths, numpy=True, return_path=False):
        """
        image_dir: directory of all training images (e.g., NIfTI)
        num_pairs_per_epoch: how many random pairs to draw per epoch
        """
        self.image_paths = image_paths
        self.num_pairs = len(self.image_paths)
        self.numpy = numpy
        self.return_path = return_path
        self.seg_path = 'data/FDG_label_cortex_mask'

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        idx1, idx2 = random.sample(range(len(self.image_paths)), 2)
        img1, affine = self.load_image(self.image_paths[idx1])
        img2, affine = self.load_image(self.image_paths[idx2])

        img1_seg, img2_seg = [], []
        for i in range(6):
            seg1_path = f"{self.seg_path}/{self.image_paths[idx1].split('/')[-1].split('_')[1]}_T1w_MRI/mask{i+1}.nii.gz"
            seg1 = nib.load(seg1_path).get_fdata().astype(np.float32)
            img1_seg.append(torch.from_numpy(seg1))
            seg2_path = f"{self.seg_path}/{self.image_paths[idx2].split('/')[-1].split('_')[1]}_T1w_MRI/mask{i+1}.nii.gz"
            seg2 = nib.load(seg2_path).get_fdata().astype(np.float32)
            img2_seg.append(torch.from_numpy(seg2))

        if self.return_path:
            return img1, img2, 0, 0, affine, [self.image_paths[idx1], self.image_paths[idx2]], img1_seg, img2_seg
        return img1, img2, 0, 0, affine, img1_seg, img2_seg

    def load_image(self, path):
        if self.numpy:
            img = np.load(path)
            affine = np.load('data/affine.npy')
        else:
            nifti = nib.load(path)
            img = nifti.get_fdata().astype(np.float32)
            affine = nifti.affine

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        return torch.from_numpy(img), affine

class FixedPairDataset(Dataset):
    def __init__(self, image_paths, numpy=True, return_path=False):
        self.pairs = image_paths # paired list [[a,b], [c,d], ...]
        self.return_path = return_path
        self.numpy = numpy
        self.seg_path = 'data/FDG_label_cortex_mask'

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        moving_path, fixed_path = self.pairs[idx]
        moving, affine = self.load_image(moving_path)
        fixed, affine = self.load_image(fixed_path)

        img1_seg, img2_seg = [], []
        for i in range(6):
            seg1_path = f"{self.seg_path}/{moving_path.split('/')[-1].split('_')[1]}_T1w_MRI/mask{i+1}.nii.gz"
            seg1 = nib.load(seg1_path).get_fdata().astype(np.float32)
            img1_seg.append(torch.from_numpy(seg1))
            seg2_path = f"{self.seg_path}/{fixed_path.split('/')[-1].split('_')[1]}_T1w_MRI/mask{i+1}.nii.gz"
            seg2 = nib.load(seg2_path).get_fdata().astype(np.float32)
            img2_seg.append(torch.from_numpy(seg2))


        if self.return_path:
            return moving, fixed, 0, 0, affine, [moving_path, fixed_path], img1_seg, img2_seg
        return moving, fixed, 0, 0, affine, img1_seg, img2_seg

    def load_image(self, path):
        if self.numpy:
            img = np.load(path)
            affine = np.load('data/affine.npy')
        else:
            nifti = nib.load(path)
            img = nifti.get_fdata().astype(np.float32)
            affine = nifti.affine

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#
        return torch.from_numpy(img), affine
