#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Editing Quality Assessment Tool (IEQA)

This script evaluates the quality of image editing by comparing an edited image 
against an original image within a specific masked region (usually the background).
It performs SIFT-based alignment before calculating MSE and PSNR metrics.

Author: [Your Name/Handle]
License: MIT
"""

import os
import sys
import cv2
import json
import logging
import argparse
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Core Computation Functions ---

def align_images(original: np.ndarray, edited: np.ndarray) -> Optional[np.ndarray]:
    """
    Aligns the edited image to the original image using SIFT feature matching.

    Args:
        original (np.ndarray): The source image (BGR).
        edited (np.ndarray): The edited image to be aligned (BGR).

    Returns:
        Optional[np.ndarray]: The aligned image, or None if alignment fails.
    """
    # Convert images to grayscale
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_edited = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

    # Detect SIFT features
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray_original, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_edited, None)

    if descriptors1 is None or descriptors2 is None:
        logger.debug("No descriptors found in one of the images.")
        return None

    # Match features using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error as e:
        logger.debug(f"FLANN matching error: {e}")
        return None

    # Validate matches
    if not matches or len(matches[0]) != 2:
        return None

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) < 4:
        logger.debug("Not enough good matches found for homography.")
        return None

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate affine transform
    matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.LMEDS)

    if matrix is None:
        return None

    # Warp the edited image
    height, width = original.shape[:2]
    return cv2.warpAffine(edited, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def calculate_masked_metrics(original: np.ndarray, aligned: np.ndarray, mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates MSE and PSNR within the masked region (background).

    Args:
        original (np.ndarray): Original image.
        aligned (np.ndarray): Aligned edited image.
        mask (np.ndarray): Binary mask where >128 indicates the foreground (ignored area).

    Returns:
        Tuple[Optional[float], Optional[float]]: (MSE, PSNR) or (None, None) if validation fails.
    """
    # Resize aligned image if dimensions differ
    if aligned.shape != original.shape:
        aligned = cv2.resize(aligned, (original.shape[1], original.shape[0]))

    # Ensure mask is 2D and matches dimensions
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create inverse mask (Background = 1, Foreground = 0)
    # Assuming the input mask highlights the object to ignore (white), so we invert it.
    inv_mask = (mask < 128).astype(np.uint8)
    background_pixel_count = np.sum(inv_mask)

    if background_pixel_count == 0:
        logger.warning("No background pixels found in mask.")
        return None, None

    original_float = original.astype(np.float32)
    aligned_float = aligned.astype(np.float32)

    # Calculate squared error
    squared_error = (original_float - aligned_float) ** 2
    
    # Apply mask (expand dims to match 3 channels)
    masked_squared_error = squared_error * inv_mask[..., None]

    # Calculate MSE
    total_data_points = background_pixel_count * 3
    mse = np.sum(masked_squared_error) / total_data_points

    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)

    return float(mse), float(psnr)


def combine_masks(mask_paths: List[str]) -> Optional[np.ndarray]:
    """
    Combines multiple mask image files into a single binary mask using bitwise OR.
    """
    if not mask_paths:
        return None

    combined_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    if combined_mask is None:
        return None

    for path in mask_paths[1:]:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Resize if dimensions do not match
        if mask.shape != combined_mask.shape:
            mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            logger.debug(f"Resized mask: {path}")
            
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return combined_mask


# --- Helper Functions ---

def _find_corresponding_files(edited_path: str, original_root_dir: str) -> Tuple[Optional[str], List[str], str]:
    """
    Heuristic to find original image and masks based on the edited file path.
    
    Assumed Structure:
    - Edited:  .../CategoryName/001_edited.jpg
    - Original: original_root_dir/CategoryName/001/001.jpg
    - Masks:    original_root_dir/CategoryName/001/*_mask.jpg
    """
    edited_filename = os.path.basename(edited_path)
    category = os.path.basename(os.path.dirname(edited_path))
    
    # Extract ID (Assumes format "ID_edited.jpg" or similar)
    base_id = edited_filename.split('_')[0] 
    
    source_subfolder = os.path.join(original_root_dir, category, base_id)
    
    if not os.path.isdir(source_subfolder):
        return None, [], base_id

    # Find Original Image (Assumes ID.jpg or ID.png)
    # Using integer conversion to handle "001" vs "1" mismatches if necessary, currently strictly matching ID
    # Try multiple extensions
    original_image_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        temp_path = os.path.join(source_subfolder, f"{int(base_id)}{ext}")
        if os.path.exists(temp_path):
            original_image_path = temp_path
            break
        # Fallback to keep leading zeros if int conversion fails logic above
        temp_path_str = os.path.join(source_subfolder, f"{base_id}{ext}")
        if os.path.exists(temp_path_str):
            original_image_path = temp_path_str
            break

    # Find Masks
    mask_paths = [
        os.path.join(source_subfolder, f) 
        for f in os.listdir(source_subfolder) 
        if f.lower().endswith('_mask.jpg') or f.lower().endswith('_mask.png')
    ]

    return original_image_path, mask_paths, f"{category}/{base_id}"


# --- Main Execution ---

def process_dataset(edited_root_dir: str, original_root_dir: str, output_file: Optional[str] = None):
    """
    Main processing loop.
    """
    logger.info(f"Scanning edited images in: {edited_root_dir}")
    logger.info(f"Looking for ground truth in: {original_root_dir}")

    # 1. Discover Edited Images
    edited_image_paths = []
    for root, _, files in os.walk(edited_root_dir):
        for file in files:
            if file.lower().endswith(('_edited.jpg', '_edited.png')):
                edited_image_paths.append(os.path.join(root, file))

    if not edited_image_paths:
        logger.error("No files ending with '_edited.jpg/png' found.")
        sys.exit(1)

    logger.info(f"Found {len(edited_image_paths)} images to process.")
    
    results: List[Dict[str, Any]] = []

    # 2. Process Loop
    for edited_path in tqdm(edited_image_paths, desc="Processing"):
        try:
            # Find corresponding files
            original_path, mask_paths, unique_id = _find_corresponding_files(edited_path, original_root_dir)

            if not original_path:
                logger.debug(f"Skipping {unique_id}: Original image folder or file not found.")
                continue
            
            if not mask_paths:
                logger.debug(f"Skipping {unique_id}: No masks found.")
                continue

            # Read Images
            original_img = cv2.imread(original_path)
            edited_img = cv2.imread(edited_path)
            final_mask = combine_masks(mask_paths)

            if original_img is None or edited_img is None or final_mask is None:
                logger.warning(f"Skipping {unique_id}: Error reading image files.")
                continue

            # Compute Metrics
            aligned_img = align_images(original_img, edited_img)
            
            if aligned_img is None:
                logger.debug(f"Skipping {unique_id}: Alignment failed.")
                continue

            mse, psnr = calculate_masked_metrics(original_img, aligned_img, final_mask)

            if mse is not None:
                results.append({
                    "id": unique_id,
                    "filename": os.path.basename(edited_path),
                    "mse": round(mse, 4),
                    "psnr": round(psnr, 4)
                })

        except Exception as e:
            logger.error(f"Unexpected error processing {edited_path}: {e}")

    # 3. Reporting
    if not results:
        logger.warning("No valid results computed.")
        return

    avg_mse = np.mean([r['mse'] for r in results])
    avg_psnr = np.mean([r['psnr'] for r in results])

    print("\n" + "="*30)
    print("      EVALUATION SUMMARY      ")
    print("="*30)
    print(f"Images Processed : {len(results)} / {len(edited_image_paths)}")
    print(f"Average MSE      : {avg_mse:.4f}")
    print(f"Average PSNR     : {avg_psnr:.4f} dB")
    print("="*30)

    # 4. Save Results
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": {"avg_mse": avg_mse, "avg_psnr": avg_psnr, "count": len(results)},
                    "details": results
                }, f, indent=4)
            logger.info(f"Results saved to: {output_file}")
        except IOError as e:
            logger.error(f"Failed to save results to {output_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate background preservation metrics (MSE, PSNR) for edited images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("edited_dir", help="Root directory containing the edited images (e.g., *_edited.jpg).")
    parser.add_argument("original_dir", help="Root directory containing original images and masks.")
    parser.add_argument("--output", "-o", help="Path to save detailed results as JSON.", default="evaluation_results.json")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (DEBUG level).")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not os.path.isdir(args.edited_dir):
        logger.error(f"Edited directory not found: {args.edited_dir}")
        sys.exit(1)
        
    if not os.path.isdir(args.original_dir):
        logger.error(f"Original directory not found: {args.original_dir}")
        sys.exit(1)

    process_dataset(args.edited_dir, args.original_dir, args.output)