import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class MultiTaskFireDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to 'fire_multitask_data'.
            split (str): 'train' or 'val'.
            transform (callable, optional): PyTorch or Albumentations transforms.
        """
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []  # List of tuples: (label_path, environment_class)

        # Indoor = 0, Outdoor = 1
        self.env_mapping = {"indoor": 0, "outdoor": 1}

        # Walk through our folders and pair up images with labels
        for env_name, env_class in self.env_mapping.items():
            images_dir = os.path.join(self.split_dir, env_name, "images")
            labels_dir = os.path.join(self.split_dir, env_name, "labels")
            
            if not os.path.exists(images_dir):
                continue
                
            for img_name in os.listdir(images_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(images_dir, img_name)
                    
                    # Map to the corresponding .txt label file
                    base_name = os.path.splitext(img_name)[0]
                    lbl_path = os.path.join(labels_dir, base_name + ".txt")
                    
                    self.image_paths.append(img_path)
                    self.labels.append((lbl_path, env_class))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # 2. Get environment classification target (0 or 1)
        lbl_path, env_label = self.labels[idx]
        
        # 3. Read YOLO Bounding Boxes
        bboxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])  # Fire or smoke class index
                        coords = [float(x) for x in parts[1:]] # x_center, y_center, width, height
                        bboxes.append([class_id] + coords)
        
        bboxes = np.array(bboxes, dtype=np.float32) if len(bboxes) > 0 else np.zeros((0, 5), dtype=np.float32)

        # 4. Standard Preprocessing (Resize to 640x640, Normalization, Tensor conversion)
        # We can implement a default transform here if none is provided
        if self.transform:
            augmented = self.transform(image=np.array(image), bboxes=bboxes)
            image = augmented['image']
            bboxes = augmented['bboxes']
        else:
            # Resize image to standard YOLOv8 size (640x640)
            image = image.resize((640, 640))
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0  # [3, 640, 640]
        
        return {
            "image": image,
            "bbox_targets": torch.tensor(bboxes, dtype=torch.float32),
            "env_label": torch.tensor(env_label, dtype=torch.float32)
        }

def collate_fn(batch):
    """
    Custom collate to stack batch items.
    Since different images have different numbers of bounding boxes,
    we prepend the image's index in the batch to keep track of coordinates.
    """
    images = []
    env_labels = []
    targets = []
    
    for i, sample in enumerate(batch):
        images.append(sample["image"])
        env_labels.append(sample["env_label"])
        
        bboxes = sample["bbox_targets"]
        if len(bboxes) > 0:
            # Append batch index as column 0 -> [batch_idx, class, x, y, w, h]
            batch_idx = torch.full((bboxes.size(0), 1), i)
            bboxes_with_idx = torch.cat((batch_idx, bboxes), dim=1)
            targets.append(bboxes_with_idx)
            
    images = torch.stack(images, dim=0)
    env_labels = torch.stack(env_labels, dim=0)
    
    if len(targets) > 0:
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.empty((0, 6))
        
    return {
        "images": images,          # [Batch_Size, 3, 640, 640]
        "env_labels": env_labels,  # [Batch_Size]
        "bbox_targets": targets    # [Total_Boxes_In_Batch, 6]
    }

# ==========================================================
# 🧪 TEST BLOCK
# ==========================================================
if __name__ == "__main__":
    print("Testing Custom Multi-Task Dataset Loader...")
    
    # Initialize the training dataset
    dataset = MultiTaskFireDataset(root_dir="./fire_multitask_data", split="train")
    print(f"Total training images found: {len(dataset)}")
    
    # Create DataLoader with our custom collate function
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Pull one batch to test
    for batch in dataloader:
        print("\n--- Successful Batch Loaded ---")
        print("Images Tensor Shape:", batch["images"].shape)       # Should be [4, 3, 640, 640]
        print("Environment Labels:", batch["env_labels"])          # Should be tensor with 4 values (0s and 1s)
        print("Bounding Boxes in Batch:")
        print(batch["bbox_targets"])                                # [Num_Boxes, 6] (batch_idx, class, x, y, w, h)
        break