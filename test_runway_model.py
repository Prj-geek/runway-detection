
"""
Runway Detection Testing - Combined Task 1 (Segmentation) + Task 2 (Anchor Points)
For your dataset structure:
- Test images (2000) + PNG label masks (2000)
- JSON file with anchor points for evaluation
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import jaccard_score
from train_runway_model import CombinedUNetPlusPlus, RunwayDataset

# ============================================================================
# INFERENCE SYSTEM
# ============================================================================

class RunwayInferenceSystem:
    """Complete inference system for both tasks"""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained model"""
        model = CombinedUNetPlusPlus(
            num_classes=3,  # Adjust based on your classes
            num_anchor_points=6,  # Adjust based on your anchor points
            deep_supervision=True,
            attention=True
        )

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded from checkpoint")
            else:
                model.load_state_dict(checkpoint)
                print("✅ Model loaded from state dict")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_path, target_size=(512, 512)):
        """Preprocess image for inference"""
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]

        # Resize
        image_resized = cv2.resize(image, target_size)

        # Normalize
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor.unsqueeze(0), original_shape, image

    def predict(self, image_path):
        """Run inference on single image"""
        image_tensor, original_shape, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            seg_outputs, anchor_pred = self.model(image_tensor)

            # Get final segmentation
            if isinstance(seg_outputs, list):
                seg_prediction = seg_outputs[-1]
            else:
                seg_prediction = seg_outputs

            # Get segmentation mask
            pred_mask = torch.argmax(seg_prediction, dim=1).cpu().numpy()[0]

            # Get anchor coordinates
            anchor_coords = anchor_pred.cpu().numpy()[0]  # Shape: (6, 2)

            # Denormalize anchor coordinates to original image size
            anchor_coords[:, 0] *= original_shape[1]  # x coordinates
            anchor_coords[:, 1] *= original_shape[0]  # y coordinates

        # Resize mask back to original size
        pred_mask_resized = cv2.resize(
            pred_mask.astype(np.uint8), 
            (original_shape[1], original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )

        return pred_mask_resized, anchor_coords, original_image

    def detect_runway_lines_from_anchors(self, anchor_coords):
        """
        Convert anchor points to runway lines
        Assumes anchor points represent: [left_start, left_end, right_start, right_end, center_start, center_end]
        """
        if len(anchor_coords) >= 6:
            left_edge = [int(anchor_coords[0][0]), int(anchor_coords[0][1]), 
                        int(anchor_coords[1][0]), int(anchor_coords[1][1])]

            right_edge = [int(anchor_coords[2][0]), int(anchor_coords[2][1]), 
                         int(anchor_coords[3][0]), int(anchor_coords[3][1])]

            center_line = [int(anchor_coords[4][0]), int(anchor_coords[4][1]), 
                          int(anchor_coords[5][0]), int(anchor_coords[5][1])]

            return left_edge, right_edge, center_line

        return None, None, None

    def create_evaluation_polygon(self, left_edge, right_edge, center_line, image_shape):
        """Create evaluation polygon from detected lines"""
        if left_edge is None or right_edge is None:
            return None

        h, w = image_shape[:2]

        # Extract coordinates
        x1_l, y1_l, x2_l, y2_l = left_edge
        x1_r, y1_r, x2_r, y2_r = right_edge

        # Create polygon coordinates
        polygon_coords = [
            (max(0, min(w-1, x1_l)), max(0, min(h-1, y1_l))),  # Left edge start
            (max(0, min(w-1, x2_l)), max(0, min(h-1, y2_l))),  # Left edge end  
            (max(0, min(w-1, x2_r)), max(0, min(h-1, y2_r))),  # Right edge end
            (max(0, min(w-1, x1_r)), max(0, min(h-1, y1_r))),  # Right edge start
        ]

        return polygon_coords

    def calculate_scores(self, pred_mask, gt_mask, anchor_coords, gt_anchor_coords, 
                        left_edge, right_edge, center_line):
        """Calculate evaluation scores"""
        scores = {}

        # Task 1: Segmentation IoU
        scores['iou_score'] = self.calculate_iou(pred_mask, gt_mask)

        # Task 2: Anchor point accuracy
        if gt_anchor_coords is not None and anchor_coords is not None:
            scores['anchor_mae'] = self.calculate_anchor_mae(anchor_coords, gt_anchor_coords)
            scores['anchor_accuracy'] = self.calculate_anchor_accuracy(anchor_coords, gt_anchor_coords)
        else:
            scores['anchor_mae'] = float('inf')
            scores['anchor_accuracy'] = 0.0

        # Combined runway evaluation
        polygon_coords = self.create_evaluation_polygon(left_edge, right_edge, center_line, pred_mask.shape)
        scores['anchor_score'] = self.calculate_anchor_score(polygon_coords, pred_mask)
        scores['boolean_score'] = self.calculate_boolean_score(center_line, left_edge, right_edge)

        return scores

    def calculate_iou(self, pred_mask, gt_mask, num_classes=3):
        """Calculate mean IoU across all classes"""
        ious = []

        for c in range(num_classes):
            pred_c = (pred_mask == c)
            gt_c = (gt_mask == c)

            intersection = (pred_c & gt_c).sum()
            union = (pred_c | gt_c).sum()

            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union

            ious.append(iou)

        return np.mean(ious)

    def calculate_anchor_mae(self, pred_anchors, gt_anchors):
        """Calculate Mean Absolute Error for anchor points"""
        if pred_anchors is None or gt_anchors is None:
            return float('inf')

        # Ensure same shape
        min_points = min(len(pred_anchors), len(gt_anchors))
        pred_points = pred_anchors[:min_points]
        gt_points = gt_anchors[:min_points]

        mae = np.mean(np.abs(pred_points - gt_points))
        return mae

    def calculate_anchor_accuracy(self, pred_anchors, gt_anchors, threshold=20.0):
        """Calculate accuracy for anchor points within threshold"""
        if pred_anchors is None or gt_anchors is None:
            return 0.0

        min_points = min(len(pred_anchors), len(gt_anchors))
        pred_points = pred_anchors[:min_points]
        gt_points = gt_anchors[:min_points]

        distances = np.sqrt(np.sum((pred_points - gt_points)**2, axis=1))
        accuracy = np.mean(distances <= threshold)

        return accuracy

    def calculate_anchor_score(self, polygon_coords, segmentation_mask):
        """Calculate anchor score - percentage of runway pixels within polygon"""
        if polygon_coords is None:
            return 0.0

        h, w = segmentation_mask.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Create polygon mask
        pts = np.array(polygon_coords, np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # Count runway pixels (class 1 or 2) within polygon
        runway_pixels = (segmentation_mask > 0).sum()
        if runway_pixels == 0:
            return 0.0

        polygon_runway_pixels = ((segmentation_mask > 0) & (mask > 0)).sum()

        return polygon_runway_pixels / runway_pixels

    def calculate_boolean_score(self, center_line, left_edge, right_edge):
        """Check if center line is between left and right edges"""
        if center_line is None or left_edge is None or right_edge is None:
            return False

        # Get average y-coordinates for each line
        center_y = (center_line[1] + center_line[3]) / 2
        left_y = (left_edge[1] + left_edge[3]) / 2
        right_y = (right_edge[1] + right_edge[3]) / 2

        # Check if center is between edges (with tolerance)
        min_edge_y = min(left_y, right_y)
        max_edge_y = max(left_y, right_y)
        tolerance = abs(max_edge_y - min_edge_y) * 0.1

        return (min_edge_y - tolerance) <= center_y <= (max_edge_y + tolerance)

# ============================================================================
# TEST DATASET EVALUATION
# ============================================================================

def evaluate_test_dataset(model_path, test_images, test_labels, test_json, output_folder='results/test_evaluation'):
    """
    Comprehensive evaluation on test dataset
    """

    print(f"🔍 Starting evaluation on test dataset...")
    print(f"Model: {model_path}")
    print(f"Test images: {test_images}")
    print(f"Test labels: {test_labels}")
    print(f"Test JSON: {test_json}")

    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Initialize inference system
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_system = RunwayInferenceSystem(model_path, device)

    # Load test annotations
    with open(test_json, 'r') as f:
        test_annotations = json.load(f)

    print(f"Loaded {len(test_annotations)} test annotations")

    # Get all test images
    test_images_path = Path(test_images)
    test_labels_path = Path(test_labels)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(test_images_path.glob(ext)))

    image_files = sorted(image_files)
    print(f"Found {len(image_files)} test images")

    # Results storage
    results = []
    successful_predictions = 0

    # Process each test image
    for i, img_path in enumerate(tqdm(image_files, desc="Processing test images")):
        try:
            img_name = img_path.name

            # Load ground truth mask
            mask_name = img_name.rsplit('.', 1)[0] + '.png'
            gt_mask_path = test_labels_path / mask_name

            if not gt_mask_path.exists():
                gt_mask_path = test_labels_path / img_name

            if gt_mask_path.exists():
                gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                print(f"Warning: No ground truth mask for {img_name}")
                continue

            # Run inference
            pred_mask, anchor_coords, original_img = inference_system.predict(img_path)

            # Resize ground truth to match prediction
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)

            # Get ground truth anchor points
            gt_anchor_coords = get_ground_truth_anchors(img_name, test_annotations, original_img.shape)

            # Convert anchor points to lines
            left_edge, right_edge, center_line = inference_system.detect_runway_lines_from_anchors(anchor_coords)

            # Calculate scores
            scores = inference_system.calculate_scores(
                pred_mask, gt_mask, anchor_coords, gt_anchor_coords,
                left_edge, right_edge, center_line
            )

            # Store results
            result = {
                'filename': img_name,
                'segmentation_iou': scores['iou_score'],
                'anchor_mae': scores['anchor_mae'],
                'anchor_accuracy': scores['anchor_accuracy'],
                'anchor_score': scores['anchor_score'],
                'boolean_score': scores['boolean_score'],
                'lines_detected': {
                    'left_edge': left_edge is not None,
                    'right_edge': right_edge is not None,
                    'center_line': center_line is not None
                }
            }

            results.append(result)
            successful_predictions += 1

            # Save visualization for first 20 images
            if i < 20:
                save_test_visualization(
                    original_img, pred_mask, gt_mask, anchor_coords, gt_anchor_coords,
                    left_edge, right_edge, center_line, scores,
                    save_path=f"{output_folder}/visualizations/{img_name}_result.png"
                )

        except Exception as e:
            print(f"❌ Error processing {img_path}: {str(e)}")
            results.append({
                'filename': img_path.name,
                'error': str(e)
            })

    print(f"\n✅ Evaluation completed!")
    print(f"Successfully processed: {successful_predictions}/{len(image_files)} images")

    # Generate summary
    generate_test_summary(results, output_folder)

    return results

def get_ground_truth_anchors(img_name, annotations, img_shape):
    """Extract ground truth anchor points from JSON"""

    # Try different possible keys
    possible_keys = [
        img_name,
        img_name.rsplit('.', 1)[0],
        img_name.rsplit('.', 1)[0] + '.jpg',
        img_name.rsplit('.', 1)[0] + '.png'
    ]

    for key in possible_keys:
        if key in annotations:
            annotation = annotations[key]

            anchor_points = []

            # Extract points based on your JSON format
            if isinstance(annotation, dict):
                if 'left_edge' in annotation:
                    left_edge = annotation['left_edge']
                    if isinstance(left_edge, list) and len(left_edge) == 4:
                        anchor_points.extend([[left_edge[0], left_edge[1]], [left_edge[2], left_edge[3]]])

                if 'right_edge' in annotation:
                    right_edge = annotation['right_edge']
                    if isinstance(right_edge, list) and len(right_edge) == 4:
                        anchor_points.extend([[right_edge[0], right_edge[1]], [right_edge[2], right_edge[3]]])

                if 'center_line' in annotation:
                    center_line = annotation['center_line']
                    if isinstance(center_line, list) and len(center_line) == 4:
                        anchor_points.extend([[center_line[0], center_line[1]], [center_line[2], center_line[3]]])

            elif isinstance(annotation, list):
                anchor_points = annotation

            # Ensure we have 6 points
            if len(anchor_points) >= 6:
                return np.array(anchor_points[:6], dtype=np.float32)
            elif len(anchor_points) > 0:
                # Pad to 6 points
                anchor_coords = np.array(anchor_points, dtype=np.float32)
                while len(anchor_coords) < 6:
                    anchor_coords = np.vstack([anchor_coords, anchor_coords[-1]])
                return anchor_coords[:6]

    return None

def save_test_visualization(original_img, pred_mask, gt_mask, pred_anchors, gt_anchors,
                           left_edge, right_edge, center_line, scores, save_path):
    """Save comprehensive test visualization"""

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Test Results Visualization', fontsize=16, fontweight='bold')

    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Ground truth mask
    gt_colored = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    gt_colored[gt_mask == 0] = [0, 0, 0]
    gt_colored[gt_mask == 1] = [0, 255, 0]
    gt_colored[gt_mask == 2] = [255, 0, 0]

    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')

    # Predicted mask
    pred_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    pred_colored[pred_mask == 0] = [0, 0, 0]
    pred_colored[pred_mask == 1] = [0, 255, 0]
    pred_colored[pred_mask == 2] = [255, 0, 0]

    axes[0, 2].imshow(pred_colored)
    axes[0, 2].set_title(f'Predicted Mask (IoU: {scores["iou_score"]:.3f})')
    axes[0, 2].axis('off')

    # Ground truth anchor points
    anchor_viz_gt = original_img.copy()
    if gt_anchors is not None:
        for i, point in enumerate(gt_anchors):
            cv2.circle(anchor_viz_gt, (int(point[0]), int(point[1])), 8, (0, 255, 0), -1)
            cv2.putText(anchor_viz_gt, str(i), (int(point[0])+10, int(point[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    axes[1, 0].imshow(anchor_viz_gt)
    axes[1, 0].set_title('Ground Truth Anchors')
    axes[1, 0].axis('off')

    # Predicted anchor points
    anchor_viz_pred = original_img.copy()
    if pred_anchors is not None:
        for i, point in enumerate(pred_anchors):
            cv2.circle(anchor_viz_pred, (int(point[0]), int(point[1])), 8, (255, 0, 0), -1)
            cv2.putText(anchor_viz_pred, str(i), (int(point[0])+10, int(point[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    axes[1, 1].imshow(anchor_viz_pred)
    axes[1, 1].set_title(f'Predicted Anchors (MAE: {scores["anchor_mae"]:.2f})')
    axes[1, 1].axis('off')

    # Lines visualization
    lines_viz = original_img.copy()
    if left_edge:
        cv2.line(lines_viz, (left_edge[0], left_edge[1]), (left_edge[2], left_edge[3]), (255, 0, 0), 3)
    if right_edge:
        cv2.line(lines_viz, (right_edge[0], right_edge[1]), (right_edge[2], right_edge[3]), (0, 255, 0), 3)
    if center_line:
        cv2.line(lines_viz, (center_line[0], center_line[1]), (center_line[2], center_line[3]), (0, 0, 255), 3)

    axes[1, 2].imshow(lines_viz)
    axes[1, 2].set_title(f'Detected Lines (Boolean: {scores["boolean_score"]})')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def generate_test_summary(results, output_folder):
    """Generate comprehensive test summary"""

    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]

    if not successful_results:
        print("❌ No successful predictions to summarize")
        return

    print(f"\n📊 Generating test summary...")

    # Calculate statistics
    iou_scores = [r['segmentation_iou'] for r in successful_results]
    anchor_maes = [r['anchor_mae'] for r in successful_results if r['anchor_mae'] != float('inf')]
    anchor_accuracies = [r['anchor_accuracy'] for r in successful_results]
    anchor_scores = [r['anchor_score'] for r in successful_results]
    boolean_scores = [r['boolean_score'] for r in successful_results]

    # Summary statistics
    summary = {
        'total_images': len(results),
        'successful_predictions': len(successful_results),
        'success_rate': len(successful_results) / len(results),
        'segmentation_iou': {
            'mean': np.mean(iou_scores),
            'std': np.std(iou_scores),
            'min': np.min(iou_scores),
            'max': np.max(iou_scores)
        },
        'anchor_mae': {
            'mean': np.mean(anchor_maes) if anchor_maes else float('inf'),
            'std': np.std(anchor_maes) if anchor_maes else 0,
            'min': np.min(anchor_maes) if anchor_maes else float('inf'),
            'max': np.max(anchor_maes) if anchor_maes else float('inf')
        },
        'anchor_accuracy': {
            'mean': np.mean(anchor_accuracies),
            'std': np.std(anchor_accuracies)
        },
        'anchor_score': {
            'mean': np.mean(anchor_scores),
            'std': np.std(anchor_scores)
        },
        'boolean_pass_rate': np.mean(boolean_scores)
    }

    # Save summary
    with open(f"{output_folder}/test_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Create CSV
    df = pd.DataFrame(successful_results)
    df.to_csv(f"{output_folder}/test_results.csv", index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {summary['total_images']}")
    print(f"Successful predictions: {summary['successful_predictions']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"\nSegmentation IoU:")
    print(f"  Mean: {summary['segmentation_iou']['mean']:.4f} ± {summary['segmentation_iou']['std']:.4f}")
    print(f"  Range: {summary['segmentation_iou']['min']:.4f} - {summary['segmentation_iou']['max']:.4f}")
    print(f"\nAnchor Point MAE:")
    if summary['anchor_mae']['mean'] != float('inf'):
        print(f"  Mean: {summary['anchor_mae']['mean']:.2f} ± {summary['anchor_mae']['std']:.2f}")
    else:
        print(f"  No valid anchor predictions")
    print(f"\nAnchor Point Accuracy (within 20px):")
    print(f"  Mean: {summary['anchor_accuracy']['mean']:.3f}")
    print(f"\nAnchor Score:")
    print(f"  Mean: {summary['anchor_score']['mean']:.3f}")
    print(f"\nBoolean Score Pass Rate: {summary['boolean_pass_rate']:.1%}")

# ============================================================================
# MAIN TESTING SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("🔍 Combined Runway Detection Testing")
    print("=" * 50)

    # Configuration - MODIFY THESE PATHS
    CONFIG = {
        'model_path': 'models/combined_runway_model.pth',  # Your trained model
        'test_images': 'data/test/images',                # Your 2000 test images
        'test_labels': 'data/test/labels',                # Your 2000 PNG labels
        'test_json': 'data/test/annotations.json',        # Test anchor points JSON
        'output_folder': 'results/test_evaluation'        # Where to save results
    }

    print(f"Model: {CONFIG['model_path']}")
    print(f"Test images: {CONFIG['test_images']}")
    print(f"Test labels: {CONFIG['test_labels']}")
    print(f"Test JSON: {CONFIG['test_json']}")

    # Check if paths exist
    for key, path in CONFIG.items():
        if key != 'output_folder' and not os.path.exists(path):
            print(f"❌ Path not found: {path}")
            exit(1)

    # Run evaluation
    print(f"\n🚀 Starting evaluation on test dataset...")
    results = evaluate_test_dataset(
        model_path=CONFIG['model_path'],
        test_images=CONFIG['test_images'],
        test_labels=CONFIG['test_labels'],
        test_json=CONFIG['test_json'],
        output_folder=CONFIG['output_folder']
    )

    print(f"\n✅ Testing completed!")
    print(f"Results saved in: {CONFIG['output_folder']}")
