import os
from torchvision.datasets import VisionDataset
from PIL import Image
from sklearn.model_selection import train_test_split


import os
import shutil
import pandas as pd
from torchvision.datasets import VisionDataset
from PIL import Image
from sklearn.model_selection import train_test_split
import kagglehub

class CustomDataset(VisionDataset):
    def __init__(self, root_path, subset="train", transform=None, target_transform=None, 
                 split_ratios=(0.7, 0.2, 0.1), seed=42, samples_per_class=500):
        super().__init__(root_path, transform=transform, target_transform=target_transform)
        
        self.root = root_path
        self.subset = subset
        self.split_ratios = split_ratios
        self.seed = seed
        self.samples_per_class = samples_per_class
        self.classes = ['mel', 'bcc']  # Fixed for this specific use case

        # Create dataset if it doesn't exist
        if not os.path.exists(self.root):
            self._download_and_prepare()
            
        self.class_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _download_and_prepare(self):
        """Download and prepare the dataset structure"""
        # Download original dataset
        print("Downloading dataset...")
        downloaded_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        
        # Load metadata
        metadata = pd.read_csv(os.path.join(downloaded_path, 'HAM10000_metadata.csv'))
        
        # Map image IDs to paths
        image_id_to_path = {}
        for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            part_dir = os.path.join(downloaded_path, part)
            if os.path.exists(part_dir):
                for fname in os.listdir(part_dir):
                    if fname.endswith('.jpg'):
                        image_id_to_path[fname.split('.')[0]] = os.path.join(part_dir, fname)

        # Process each class
        os.makedirs(self.root, exist_ok=True)
        for cls in self.classes:
            # Get all image IDs for this class
            cls_ids = metadata[metadata['dx'] == cls]['image_id'].tolist()

            # Randomly select `samples_per_class`
            if len(cls_ids) > self.samples_per_class:
                cls_ids = train_test_split(
                    cls_ids, train_size=self.samples_per_class,
                    random_state=self.seed, shuffle=True
                )[0]  # Only take `samples_per_class` images

            # Compute exact split sizes
            train_size = int(len(cls_ids) * self.split_ratios[0])
            val_size = int(len(cls_ids) * self.split_ratios[1])
            test_size = len(cls_ids) - train_size - val_size  # Ensure remaining images go to test

            # Perform the splits
            train, remaining = train_test_split(cls_ids, train_size=train_size, random_state=self.seed, shuffle=True)
            val, test = train_test_split(remaining, train_size=val_size, test_size=test_size, random_state=self.seed, shuffle=True)

            # Copy files to respective directories
            for split_name, split_ids in zip(['train', 'val', 'test'], [train, val, test]):
                split_dir = os.path.join(self.root, split_name, cls)
                os.makedirs(split_dir, exist_ok=True)

                for img_id in split_ids:
                    src = image_id_to_path.get(img_id)
                    if src:
                        shutil.copy(src, os.path.join(split_dir, os.path.basename(src)))

    def _make_dataset(self):
        """Create dataset from prepared folder structure"""
        samples = []
        target_dir = os.path.join(self.root, self.subset)
        
        for cls in self.classes:
            cls_dir = os.path.join(target_dir, cls)
            cls_idx = self.class_idx[cls]
            
            for fname in os.listdir(cls_dir):
                if fname.endswith('.jpg'):
                    path = os.path.join(cls_dir, fname)
                    samples.append((path, cls_idx))
        
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

    def __len__(self):
        return len(self.samples)