# Helper functions placeholder
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score

def load_nifti(path):
    """Load a NIfTI file and return the image data as a NumPy array."""
    img = nib.load(path)
    return img.get_fdata(), img.affine

def save_nifti(data, affine, path):
    """Save a NumPy array as a NIfTI file."""
    nifti_img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(nifti_img, path)

def dice_score(pred, target, threshold=0.5):
    """Compute Dice Similarity Coefficient."""
    pred = (pred > threshold).float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def jaccard_index(pred, target, threshold=0.5):
    """Compute Jaccard Index (IoU)."""
    pred = (pred > threshold).view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    return jaccard_score(target, pred, average='binary')

def preprocess_volume(volume, target_shape=(96, 512, 512)):
    """Resize or crop/pad volume to match target shape."""
    from skimage.transform import resize
    vol = resize(volume, target_shape, mode='constant', anti_aliasing=True)
    return vol

def normalize(volume):
    """Normalize image volume to [0, 1] range."""
    return (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)

def to_tensor(volume):
    """Convert 3D volume to 5D PyTorch tensor: (B, C, D, H, W)"""
    return torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
