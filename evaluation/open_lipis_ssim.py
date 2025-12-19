"""
Image Editing Quality Evaluation Script

This script evaluates the quality of image editing by comparing original and edited images.
It calculates SSIM and LPIPS metrics specifically for the background (regions outside the masks)
to assess background preservation.

Key Features:
- Strict dimension validation (no resizing allowed).
- Mask-based evaluation (ignores edited regions).
- Batch processing with multi-threading.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate

# TorchMetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Wrapper for calculating perceptual metrics (SSIM, LPIPS) using TorchMetrics.
    """
    def __init__(self, device: str) -> None:
        self.device = device
        # Initialize metrics on the specified device
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def _prepare_tensor(self, img, mask=None):
        """
        Preprocesses image and mask into normalized tensors (0-1).
        """
        # Convert to float32 and normalize to [0, 1]
        img = np.array(img).astype(np.float32) / 255.0

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            if mask.ndim == 2:
                mask = mask[:, :, None]  # Expand channel dimension
            
            # Normalize mask to [0, 1] if it's in [0, 255]
            if mask.max() > 1.0:
                mask = mask / 255.0
            
            # Apply mask to image
            img = img * mask

        # Convert to tensor: (B, C, H, W)
        tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None) -> float:
        """Calculates Structural Similarity Index Measure."""
        pred_tensor = self._prepare_tensor(img_pred, mask_pred)
        gt_tensor = self._prepare_tensor(img_gt, mask_gt)
        
        # Calculate SSIM
        score = self.ssim_metric(pred_tensor, gt_tensor)
        return score.cpu().item()

    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None) -> float:
        """Calculates LPIPS score."""
        pred_tensor = self._prepare_tensor(img_pred, mask_pred)
        gt_tensor = self._prepare_tensor(img_gt, mask_gt)
        
        # LPIPS expects inputs in range [-1, 1], prepare_tensor gives [0, 1]
        score = self.lpips_metric(pred_tensor * 2 - 1, gt_tensor * 2 - 1)
        return score.cpu().item()

def load_image(path: str) -> Image.Image:
    """Loads an image using PIL and converts to RGB."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load image: {path} ({e})")
        return None

def process_single_task(task_data: dict, device: str) -> dict:
    """
    Worker function to process a single image pair.
    Loads images, validates dimensions, combines masks, and computes metrics.
    """
    result = task_data.copy()
    task_id = f"{task_data['topic']}_{task_data['id']}"

    try:
        # 1. Load JSON Metadata if available
        if task_data.get('json_path'):
            try:
                with open(task_data['json_path'], 'r', encoding='utf-8-sig') as f:
                    result["json_content"] = json.load(f)
            except Exception as e:
                logger.warning(f"[{task_id}] Failed to load JSON: {e}")

        # 2. Load Images
        orig_img = load_image(task_data['image'])
        edit_img = load_image(task_data['edit_image'])
        
        # Load Masks (Grayscale)
        mask1 = cv2.imread(task_data['mask1'], cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(task_data['mask2'], cv2.IMREAD_GRAYSCALE)

        if any(x is None for x in [orig_img, edit_img, mask1, mask2]):
            raise FileNotFoundError("One or more image files could not be loaded.")

        # 3. Strict Dimension Check (No resizing allowed)
        w_orig, h_orig = orig_img.size
        w_edit, h_edit = edit_img.size
        h_m1, w_m1 = mask1.shape[:2]
        h_m2, w_m2 = mask2.shape[:2]

        if not (h_orig == h_edit == h_m1 == h_m2 and w_orig == w_edit == w_m1 == w_m2):
            raise ValueError(
                f"Dimension mismatch! Orig: {orig_img.size}, Edit: {edit_img.size}, "
                f"Mask1: {mask1.shape}, Mask2: {mask2.shape}"
            )

        # 4. Prepare Background Mask
        # Combine masks (Object region)
        combined_mask = cv2.bitwise_or(mask1, mask2)
        # Invert to get Background region (1/255 for background, 0 for object)
        inv_mask = (combined_mask < 128).astype(np.uint8) * 255

        # 5. Compute Metrics
        # Note: Instantiating metrics inside threads ensures thread-safety for CUDA streams
        metrics = MetricsCalculator(device)
        
        ssim_score = metrics.calculate_ssim(edit_img, orig_img, inv_mask, inv_mask)
        lpips_score = metrics.calculate_lpips(edit_img, orig_img, inv_mask, inv_mask)

        result["ssim_similarity_outside_mask"] = ssim_score
        result["lpips_similarity_outside_mask"] = lpips_score

    except Exception as e:
        # Log error but don't crash main process
        logger.error(f"[{task_id}] Error: {e}")
        result["ssim_similarity_outside_mask"] = None
        result["lpips_similarity_outside_mask"] = None

    return result

def scan_directories(source_dir: str, edited_dir: str) -> list:
    """
    Scans source and edited directories to match files based on ID.
    Returns a list of task dictionaries.
    """
    tasks = []
    logger.info("Scanning directories for matching files...")

    if not os.path.isdir(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        return []

    # Iterate over topics
    for topic in os.listdir(source_dir):
        topic_path = os.path.join(source_dir, topic)
        if not os.path.isdir(topic_path):
            continue

        # Iterate over image IDs
        for img_id in os.listdir(topic_path):
            img_dir = os.path.join(topic_path, img_id)
            if not os.path.isdir(img_dir):
                continue

            try:
                # Format ID: 1 -> "1" for source, "001" for edited
                id_num = int(img_id)
                src_id_str = str(id_num)
                edit_id_str = f"{id_num:03d}"
            except ValueError:
                continue

            # Construct paths
            src_img_path = os.path.join(img_dir, f"{src_id_str}.jpg")
            edited_img_path = os.path.join(edited_dir, topic, f"{edit_id_str}_edited.jpg")
            mask1_path = os.path.join(img_dir, f"{src_id_str}_mask.jpg")

            # Validate basic existence
            if not os.path.exists(src_img_path): continue
            if not os.path.exists(edited_img_path):
                logger.debug(f"Edited image missing for ID {img_id}")
                continue
            if not os.path.exists(mask1_path): continue

            # Find second mask and JSON dynamically
            mask2_path = None
            json_path = None
            
            try:
                files = os.listdir(img_dir)
                # Find any mask that isn't mask1
                other_masks = [f for f in files if f.endswith("_mask.jpg") and f != f"{src_id_str}_mask.jpg"]
                json_files = [f for f in files if f.endswith(".json")]

                if len(other_masks) == 1:
                    mask2_path = os.path.join(img_dir, other_masks[0])
                
                if len(json_files) == 1:
                    json_path = os.path.join(img_dir, json_files[0])
            except OSError:
                continue

            # Skip if we don't have the complete set (according to original logic)
            if not mask2_path:
                continue

            tasks.append({
                "topic": topic,
                "id": img_id,
                "image": src_img_path,
                "edit_image": edited_img_path,
                "mask1": mask1_path,
                "mask2": mask2_path,
                "json_path": json_path
            })

    return tasks

def main():
    parser = argparse.ArgumentParser(description="Image Editing Quality Evaluator")
    parser.add_argument("source_dir", help="Path to original data directory")
    parser.add_argument("edited_dir", help="Path to edited/prediction data directory")
    parser.add_argument("--output", default="evaluation_results.json", help="Path to save output JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device (cuda/cpu)")
    args = parser.parse_args()

    # 1. Scan Files
    tasks = scan_directories(args.source_dir, args.edited_dir)
    if not tasks:
        logger.error("No matching tasks found. Please check your directory structure.")
        sys.exit(1)
    
    logger.info(f"Found {len(tasks)} matching tasks.")

    # 2. Process in Parallel
    results = []
    # Adjust max_workers based on GPU VRAM availability if using CUDA
    workers = os.cpu_count() if args.device == 'cpu' else min(4, os.cpu_count())
    
    logger.info(f"Starting processing with {workers} workers on {args.device}...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit tasks
        futures = [executor.submit(process_single_task, task, args.device) for task in tasks]
        
        # Progress bar
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = future.result()
            if res:
                results.append(res)

    # 3. Generate Report
    # Sort for consistent output
    results.sort(key=lambda x: (x.get('topic', ''), int(x.get('id', 0))))

    # Extract valid scores
    valid_scores = [
        r["ssim_similarity_outside_mask"] 
        for r in results 
        if r.get("ssim_similarity_outside_mask") is not None
    ]

    if valid_scores:
        stats = [
            ["Count", len(valid_scores)],
            ["Mean", np.mean(valid_scores)],
            ["Std Dev", np.std(valid_scores)],
            ["Min", np.min(valid_scores)],
            ["Max", np.max(valid_scores)]
        ]
        print("\n" + "="*40)
        print("SSIM Evaluation Summary (Outside Mask)")
        print("="*40)
        print(tabulate(stats, tablefmt="simple"))
        print("="*40 + "\n")
    else:
        logger.warning("No valid metrics calculated.")

    # 4. Save to JSON
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    try:
        with open(args.output, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to: {args.output}")
    except IOError as e:
        logger.error(f"Failed to save output file: {e}")

if __name__ == "__main__":
    main()