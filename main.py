# Entry point script
import argparse
import os
import torch
from torch.utils.data import DataLoader
from models.model import CoAtUNet
from train import StrokeDataset, train_model
from utils.utils import load_nifti, preprocess_volume, normalize, to_tensor


def load_dataset(image_paths, mask_paths):
    images, masks = [], []
    for img_path, msk_path in zip(image_paths, mask_paths):
        img_np, _ = load_nifti(img_path)
        msk_np, _ = load_nifti(msk_path)
        img_np = normalize(preprocess_volume(img_np))
        msk_np = preprocess_volume(msk_np)

        images.append(to_tensor(img_np))
        masks.append(to_tensor(msk_np))
    return images, masks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_imgs', type=str, required=True, help='Path to training image folder')
    parser.add_argument('--train_masks', type=str, required=True, help='Path to training mask folder')
    parser.add_argument('--val_imgs', type=str, required=True, help='Path to validation image folder')
    parser.add_argument('--val_masks', type=str, required=True, help='Path to validation mask folder')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='results/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # Load data
    train_img_paths = sorted([os.path.join(args.train_imgs, f) for f in os.listdir(args.train_imgs) if f.endswith('.nii')])
    train_mask_paths = sorted([os.path.join(args.train_masks, f) for f in os.listdir(args.train_masks) if f.endswith('.nii')])
    val_img_paths = sorted([os.path.join(args.val_imgs, f) for f in os.listdir(args.val_imgs) if f.endswith('.nii')])
    val_mask_paths = sorted([os.path.join(args.val_masks, f) for f in os.listdir(args.val_masks) if f.endswith('.nii')])

    train_images, train_masks = load_dataset(train_img_paths, train_mask_paths)
    val_images, val_masks = load_dataset(val_img_paths, val_mask_paths)

    train_loader = DataLoader(StrokeDataset(train_images, train_masks), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(StrokeDataset(val_images, val_masks), batch_size=1, shuffle=False)

    # Initialize model
    model = CoAtUNet()
    model.to(args.device)

    # Train model
    train_model(model, train_loader, val_loader, args.epochs, args.lr, args.device, args.save_path)


if __name__ == '__main__':
    main()