"""
Runway Detection Model - Combined Task 1 (Segmentation) + Task 2 (Anchor Points)
Optimized for your dataset structure:
- Task 1: Images + PNG mask labels
- Task 2: JSON files with anchor points
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# U-NET++ MODEL ARCHITECTURE
# ============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        return x

class AttentionGate(nn.Module):
    """Attention Gate for U-Net++"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class CombinedUNetPlusPlus(nn.Module):
    """
    Combined U-Net++ for both segmentation and anchor point detection
    """
    def __init__(self, num_classes=3, input_channels=3, num_anchor_points=6, deep_supervision=True, attention=True):
        super(CombinedUNetPlusPlus, self).__init__()

        # Filter sizes
        nb_filter = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision
        self.attention = attention
        self.num_anchor_points = num_anchor_points

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder blocks
        self.conv0_0 = ConvBlock(input_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        # Nested connections
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # Attention gates
        if self.attention:
            self.att1 = AttentionGate(F_g=nb_filter[1], F_l=nb_filter[0], F_int=nb_filter[0]//2)
            self.att2 = AttentionGate(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[1]//2)
            self.att3 = AttentionGate(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[2]//2)
            self.att4 = AttentionGate(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[3]//2)

        # Task 1: Segmentation outputs
        if self.deep_supervision:
            self.seg_final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.seg_final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.seg_final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.seg_final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.seg_final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        # Task 2: Anchor point regression head
        self.anchor_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.anchor_fc = nn.Sequential(
            nn.Linear(nb_filter[4], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_anchor_points * 2)  # x, y coordinates for each point
        )

    def forward(self, input):
        # Encoder path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Nested connections with attention
        if self.attention:
            x0_0_att = self.att1(self.up(x1_0), x0_0)
            x0_1 = self.conv0_1(torch.cat([x0_0_att, self.up(x1_0)], 1))

            x1_0_att = self.att2(self.up(x2_0), x1_0)
            x1_1 = self.conv1_1(torch.cat([x1_0_att, self.up(x2_0)], 1))

            x2_0_att = self.att3(self.up(x3_0), x2_0)
            x2_1 = self.conv2_1(torch.cat([x2_0_att, self.up(x3_0)], 1))

            x3_0_att = self.att4(self.up(x4_0), x3_0)
            x3_1 = self.conv3_1(torch.cat([x3_0_att, self.up(x4_0)], 1))
        else:
            x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Task 1: Segmentation outputs
        if self.deep_supervision:
            seg_output1 = self.seg_final1(x0_1)
            seg_output2 = self.seg_final2(x0_2)
            seg_output3 = self.seg_final3(x0_3)
            seg_output4 = self.seg_final4(x0_4)
            seg_outputs = [seg_output1, seg_output2, seg_output3, seg_output4]
        else:
            seg_outputs = self.seg_final(x0_4)

        # Task 2: Anchor point prediction
        anchor_features = self.anchor_pool(x4_0)
        anchor_features = anchor_features.view(anchor_features.size(0), -1)
        anchor_coords = self.anchor_fc(anchor_features)

        # Reshape anchor coordinates to (batch_size, num_points, 2)
        anchor_coords = anchor_coords.view(-1, self.num_anchor_points, 2)

        return seg_outputs, anchor_coords

# ============================================================================
# DATASET CLASS
# ============================================================================

class RunwayDataset(Dataset):
    """
    Dataset for your structure:
    - Images: RGB images
    - Labels: PNG segmentation masks
    - Anchor Points: JSON with coordinates
    """
    def __init__(self, image_folder, label_folder, json_file, transform=None, input_size=(512, 512)):
        self.image_folder = Path(image_folder)
        self.label_folder = Path(label_folder)
        self.transform = transform
        self.input_size = input_size

        # Load JSON annotations for anchor points
        with open(json_file, 'r') as f:
            self.anchor_data = json.load(f)

        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_files.extend(list(self.image_folder.glob(ext)))

        self.image_files = sorted(self.image_files)

        print(f"Found {len(self.image_files)} images")
        print(f"Loaded {len(self.anchor_data)} anchor annotations")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]

        # Load corresponding mask
        img_name = img_path.name
        # Try different mask extensions
        mask_name_png = img_name.rsplit('.', 1)[0] + '.png'
        mask_path = self.label_folder / mask_name_png

        if not mask_path.exists():
            # Try with same extension
            mask_path = self.label_folder / img_name

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Warning: No mask found for {img_name}")
            mask = np.zeros(original_shape, dtype=np.uint8)

        # Load anchor points
        anchor_coords = self.get_anchor_points(img_name, original_shape)

        # Resize image and mask
        image = cv2.resize(image, self.input_size)
        mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)

        # Resize anchor coordinates proportionally
        scale_x = self.input_size[0] / original_shape[1]
        scale_y = self.input_size[1] / original_shape[0]

        if anchor_coords is not None:
            anchor_coords[:, 0] *= scale_x  # x coordinates
            anchor_coords[:, 1] *= scale_y  # y coordinates

            # Normalize to [0, 1]
            anchor_coords[:, 0] /= self.input_size[0]
            anchor_coords[:, 1] /= self.input_size[1]
        else:
            # Create dummy anchor points if not found
            anchor_coords = np.zeros((6, 2), dtype=np.float32)

        if self.transform:
            # Apply augmentation to image and mask
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        anchor_coords = torch.from_numpy(anchor_coords).float()

        return {
            'image': image,
            'mask': mask,
            'anchor_coords': anchor_coords,
            'filename': img_name
        }

    def get_anchor_points(self, img_name, img_shape):
        """Extract anchor points from JSON"""

        # Try different possible keys for the image
        possible_keys = [
            img_name,
            img_name.rsplit('.', 1)[0],  # without extension
            img_name.rsplit('.', 1)[0] + '.jpg',
            img_name.rsplit('.', 1)[0] + '.png'
        ]

        for key in possible_keys:
            if key in self.anchor_data:
                annotation = self.anchor_data[key]

                # Extract anchor points - adjust this based on your JSON format
                anchor_points = []

                # Common JSON structures - adapt to your format
                if isinstance(annotation, dict):
                    # Option 1: Points stored as separate keys
                    if 'left_edge' in annotation:
                        left_edge = annotation['left_edge']
                        if isinstance(left_edge, list) and len(left_edge) == 4:  # [x1, y1, x2, y2]
                            anchor_points.extend([[left_edge[0], left_edge[1]], [left_edge[2], left_edge[3]]])

                    if 'right_edge' in annotation:
                        right_edge = annotation['right_edge']
                        if isinstance(right_edge, list) and len(right_edge) == 4:
                            anchor_points.extend([[right_edge[0], right_edge[1]], [right_edge[2], right_edge[3]]])

                    if 'center_line' in annotation:
                        center_line = annotation['center_line']
                        if isinstance(center_line, list) and len(center_line) == 4:
                            anchor_points.extend([[center_line[0], center_line[1]], [center_line[2], center_line[3]]])

                    # Option 2: Points stored as 'points' or 'anchor_points'
                    if 'points' in annotation:
                        points = annotation['points']
                        if isinstance(points, list):
                            anchor_points.extend(points)

                    if 'anchor_points' in annotation:
                        points = annotation['anchor_points']
                        if isinstance(points, list):
                            anchor_points.extend(points)

                elif isinstance(annotation, list):
                    # Option 3: Direct list of points
                    anchor_points = annotation

                # Convert to numpy array and ensure we have 6 points
                if len(anchor_points) >= 6:
                    anchor_coords = np.array(anchor_points[:6], dtype=np.float32)
                elif len(anchor_points) > 0:
                    # Pad or repeat points to get 6 points
                    anchor_coords = np.array(anchor_points, dtype=np.float32)
                    while len(anchor_coords) < 6:
                        anchor_coords = np.vstack([anchor_coords, anchor_coords[-1]])
                    anchor_coords = anchor_coords[:6]
                else:
                    anchor_coords = None

                return anchor_coords

        return None

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.softmax(predictions, dim=1)

        dice_score = 0
        for i in range(predictions.shape[1]):
            pred_i = predictions[:, i, :, :]
            target_i = (targets == i).float()

            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()

            dice_score += (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_score / predictions.shape[1]

class CombinedLoss(nn.Module):
    """Combined loss for segmentation and anchor point regression"""
    def __init__(self, seg_weight=1.0, anchor_weight=1.0, ce_weight=0.6, dice_weight=0.4):
        super(CombinedLoss, self).__init__()
        self.seg_weight = seg_weight
        self.anchor_weight = anchor_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, seg_pred, anchor_pred, seg_target, anchor_target):
        # Segmentation loss
        if isinstance(seg_pred, list):  # Deep supervision
            seg_loss = 0
            for pred in seg_pred:
                ce = self.ce_loss(pred, seg_target)
                dice = self.dice_loss(pred, seg_target)
                seg_loss += self.ce_weight * ce + self.dice_weight * dice
            seg_loss /= len(seg_pred)
        else:
            ce = self.ce_loss(seg_pred, seg_target)
            dice = self.dice_loss(seg_pred, seg_target)
            seg_loss = self.ce_weight * ce + self.dice_weight * dice

        # Anchor point regression loss
        anchor_loss = self.mse_loss(anchor_pred, anchor_target)

        # Combined loss
        total_loss = self.seg_weight * seg_loss + self.anchor_weight * anchor_loss

        return total_loss, seg_loss, anchor_loss

# ============================================================================
# DATA LOADING
# ============================================================================

def get_transforms(input_size=(512, 512)):
    """Data augmentation transforms"""

    train_transform = A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
        ], p=0.4),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(input_size[0], input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform

def setup_data_loaders(train_images, train_labels, train_json, val_split=0.2, batch_size=8):
    """Setup data loaders"""

    print("Setting up data loaders...")

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Create full dataset
    full_dataset = RunwayDataset(train_images, train_labels, train_json, transform=None)

    # Split dataset
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)

    indices = list(range(total_size))
    random.shuffle(indices)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    print(f"Total samples: {total_size}")
    print(f"Training: {len(train_indices)}, Validation: {len(val_indices)}")

    # Create datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(train_loader, val_loader, num_epochs=75, device='cuda', save_path='models/runway_model.pth'):
    """Train the combined model"""

    print(f"Starting training on {device}...")

    # Create model
    model = CombinedUNetPlusPlus(
        num_classes=3,  # Adjust based on your segmentation classes
        num_anchor_points=6,  # Adjust based on your anchor points
        deep_supervision=True,
        attention=True
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = CombinedLoss(seg_weight=1.0, anchor_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Create save directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_anchor_loss = 0.0

        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(train_pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            anchor_coords = batch['anchor_coords'].to(device)

            optimizer.zero_grad()

            seg_outputs, anchor_pred = model(images)

            total_loss, seg_loss, anchor_loss = criterion(seg_outputs, anchor_pred, masks, anchor_coords)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_seg_loss += seg_loss.item()
            train_anchor_loss += anchor_loss.item()

            train_pbar.set_postfix({
                'Total Loss': f'{total_loss.item():.4f}',
                'Seg Loss': f'{seg_loss.item():.4f}',
                'Anchor Loss': f'{anchor_loss.item():.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_seg_loss = train_seg_loss / len(train_loader)
        avg_train_anchor_loss = train_anchor_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_anchor_loss = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                anchor_coords = batch['anchor_coords'].to(device)

                seg_outputs, anchor_pred = model(images)

                total_loss, seg_loss, anchor_loss = criterion(seg_outputs, anchor_pred, masks, anchor_coords)

                val_loss += total_loss.item()
                val_seg_loss += seg_loss.item()
                val_anchor_loss += anchor_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_seg_loss = val_seg_loss / len(val_loader)
        avg_val_anchor_loss = val_anchor_loss / len(val_loader)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Record metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, Anchor: {avg_train_anchor_loss:.4f})")
        print(f"Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, Anchor: {avg_val_anchor_loss:.4f})")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"✅ New best model saved! Val Loss: {best_val_loss:.4f}")

    print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("🛬 Combined Runway Detection Training")
    print("=" * 50)

    # Configuration - MODIFY THESE PATHS
    CONFIG = {
        'train_images': 'E:\train\images',        # Your 4000 training images
        'train_labels': 'E:\train\masks',        # Your 4000 PNG mask labels
        'train_json': 'E:\train\train_labels_640x360.json', # Anchor points JSON
        'model_save_path': 'models/combined_runway_model.pth',
        'batch_size': 6,        # Adjust based on GPU memory
        'num_epochs': 75,       # Training epochs
        'val_split': 0.2,       # 20% for validation
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Device: {CONFIG['device']}")
    print(f"Training images: {CONFIG['train_images']}")
    print(f"Training labels: {CONFIG['train_labels']}")
    print(f"Training JSON: {CONFIG['train_json']}")

    # Setup data loaders
    train_loader, val_loader = setup_data_loaders(
        train_images=CONFIG['train_images'],
        train_labels=CONFIG['train_labels'],
        train_json=CONFIG['train_json'],
        val_split=CONFIG['val_split'],
        batch_size=CONFIG['batch_size']
    )

    # Start training
    print(f"\n🚀 Starting training...")
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG['num_epochs'],
        device=CONFIG['device'],
        save_path=CONFIG['model_save_path']
    )

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Training completed!")
    print(f"Model saved to: {CONFIG['model_save_path']}")
    print(f"Training curves saved to: training_curves.png")
