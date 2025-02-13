import os
from torchvision.datasets import VisionDataset
from PIL import Image
from sklearn.model_selection import train_test_split


class CustomDataset(VisionDataset):
    def __init__(self, root_path, subset="train", transform=None, target_transform=None, split_ratios=(0.7, 0.15, 0.15), seed=42):
        super(CustomDataset, self).__init__(root_path, transform=transform, target_transform=target_transform)
        self.root = root_path
        self.subset = subset  # Can be "train", "val", or "test"
        self.split_ratios = split_ratios
        self.seed = seed

        self.classes, self.class_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_idx

    def _make_dataset(self):
        samples = []
        for target_class in sorted(self.class_idx.keys()):
            class_index = self.class_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    samples.append((path, class_index))

        # Split into train, val, and test sets
        train_samples, test_samples = train_test_split(
            samples, test_size=1 - self.split_ratios[0], random_state=self.seed, stratify=[s[1] for s in samples]
        )
        val_samples, test_samples = train_test_split(
            test_samples, test_size=self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2]),
            random_state=self.seed, stratify=[s[1] for s in test_samples]
        )

        if self.subset == "train":
            return train_samples
        elif self.subset == "val":
            return val_samples
        elif self.subset == "test":
            return test_samples
        else:
            raise ValueError(f"Unknown subset: {self.subset}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target
