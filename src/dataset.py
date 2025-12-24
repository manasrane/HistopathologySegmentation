import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class GLaSDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir: path to Warwick_QU_Dataset
        split: 'train' or 'test'
        """
        self.root_dir = root_dir
        self.split = split

        all_files = os.listdir(root_dir)

        if split == "train":
            self.images = sorted([
                f for f in all_files
                if f.startswith("train_") and f.endswith(".bmp") and "_anno" not in f
            ])
        else:
            self.images = sorted([
                f for f in all_files
                if f.startswith("testA_") and f.endswith(".bmp") and "_anno" not in f
            ])

        self.transform_img = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        self.transform_mask = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".bmp", "_anno.bmp")

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform_img(image)
        mask = self.transform_mask(mask)

        # Binary mask
        mask = (mask > 0).float()

        return image, mask
