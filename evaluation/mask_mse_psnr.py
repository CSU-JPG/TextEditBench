# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import re # 引入正则表达式模块

# --- 核心计算函数 (无需改动) ---

def align_images(original, edited):
    """使用 SIFT 特征匹配来对齐两张图像。"""
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_edited = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray_original, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_edited, None)

    if descriptors1 is None or descriptors2 is None:
        return None

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error:
        return None
        
    # 检查 matches 是否为空或不符合格式
    if not matches or len(matches[0]) != 2:
        return None
        
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) < 4:
        return None

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.LMEDS)

    if matrix is None:
        return None

    height, width = original.shape[:2]
    return cv2.warpAffine(edited, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def calculate_masked_metrics(original, aligned, mask):
    """在掩码指定的背景区域内，计算 MSE 和 PSNR。"""
    if aligned.shape != original.shape:
        aligned = cv2.resize(aligned, (original.shape[1], original.shape[0]))
    
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    inv_mask = (mask < 128).astype(np.uint8)
    background_pixel_count = np.sum(inv_mask)
    if background_pixel_count == 0:
        return None, None

    original_float = original.astype(np.float32)
    aligned_float = aligned.astype(np.float32)
    squared_error = (original_float - aligned_float) ** 2
    masked_squared_error = squared_error * inv_mask[..., None]
    
    total_data_points = background_pixel_count * 3
    mse = np.sum(masked_squared_error) / total_data_points

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)
        
    return mse, psnr

def combine_masks(mask_paths):
    """将多个掩码文件合并成一个。"""
    if not mask_paths:
        return None
    combined_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    if combined_mask is None:
        return None
    for path in mask_paths[1:]:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != combined_mask.shape:
            mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            print(f"警告: 掩码尺寸不一致，已调整大小 -> {path}")
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask

# --- 主执行逻辑 (已修改) ---

def main(edited_root_dir, original_root_dir):
    """
    主函数，自动遍历、匹配并评估所有图片。
    """
    print(f"开始处理...\n编辑后图片根目录: {edited_root_dir}\n原始文件根目录: {original_root_dir}\n")
    
    # 1. 查找所有编辑后的图片
    edited_image_paths = []
    for root, _, files in os.walk(edited_root_dir):
        for file in files:
            if file.lower().endswith(('_edited.jpg', '_edited.png')):
                edited_image_paths.append(os.path.join(root, file))

    if not edited_image_paths:
        print("错误：在编辑目录中找不到任何 '_edited.jpg' 或 '_edited.png' 文件。")
        return

    results = []
    
    for edited_image_path in tqdm(edited_image_paths, desc="整体进度"):
        try:
            # 2. 从路径和文件名中解析信息
            edited_filename = os.path.basename(edited_image_path)      # "001_edited.jpg"
            category = os.path.basename(os.path.dirname(edited_image_path)) # "Art & Creative Expression"
            base_id = edited_filename.split('_')[0]                   # "001"
            
            # 3. 构建对应的原始文件和掩码路径
            source_subfolder = os.path.join(original_root_dir, category, base_id)
            if not os.path.isdir(source_subfolder):
                print(f"警告: 找不到 '{base_id}' 在 '{category}' 类别下的原始文件夹，跳过: {source_subfolder}")
                continue
            
            original_image_name = f"{int(base_id)}.jpg"
            original_image_path = os.path.join(source_subfolder, original_image_name)
            if not os.path.exists(original_image_path):
                print(f"警告: 找不到 '{base_id}' 对应的原始图片，跳过: {original_image_path}")
                continue

            mask_paths = [os.path.join(source_subfolder, f) for f in os.listdir(source_subfolder) if f.lower().endswith('_mask.jpg')]
            if not mask_paths:
                print(f"警告: 找不到 '{base_id}' 对应的任何掩码文件，跳过。")
                continue
                
            # 4. 读取和处理文件
            original_img = cv2.imread(original_image_path)
            edited_img = cv2.imread(edited_image_path)
            if original_img is None or edited_img is None:
                print(f"警告: 无法读取图片文件，跳过 {base_id}")
                continue
                
            final_mask = combine_masks(mask_paths)
            if final_mask is None:
                print(f"警告: 合并掩码失败，跳过 {base_id}")
                continue

            # 5. 执行核心计算
            aligned_img = align_images(original_img, edited_img)
            if aligned_img is None:
                print(f"警告: 图像对齐失败，跳过 {base_id} ({category})")
                continue

            mse, psnr = calculate_masked_metrics(original_img, aligned_img, final_mask)
            if mse is not None:
                results.append({"id": f"{category}/{base_id}", "mse": mse, "psnr": psnr})

        except Exception as e:
            print(f"处理 {edited_image_path} 时发生未知错误: {e}")

    # 6. 汇总并打印结果
    if not results:
        print("\n处理完成，但没有计算出任何有效结果。请检查文件路径和文件命名是否符合预期。")
        return

    avg_mse = np.mean([r['mse'] for r in results])
    avg_psnr = np.mean([r['psnr'] for r in results])

    print("\n" + "="*25 + " 评估摘要 " + "="*25)
    print(f"成功处理图片数量: {len(results)} / {len(edited_image_paths)}")
    print(f"平均背景均方误差 (MSE)  : {avg_mse:.4f}")
    print(f"平均背景峰值信噪比 (PSNR): {avg_psnr:.4f} dB")
    print("="*65)
    
    # (可选) 打印每张图片的详细分数
    # print("\n--- 详细分数 ---")
    # for res in results:
    #     print(f"ID: {res['id']:<40} -> MSE: {res['mse']:.4f}, PSNR: {res['psnr']:.4f} dB")


# --- 命令行接口 (已修改) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量评估图片编辑质量。该脚本会自动发现并匹配所有子目录中的文件，合并掩码、对齐图片，并计算背景区域的 MSE 和 PSNR。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 修改了参数名称和帮助信息，使其更清晰
    parser.add_argument("edited_root_dir", help="包含所有编辑后图片的根文件夹路径。\n例如: ./edited_images/gemini-2.5-flash-image-preview")
    parser.add_argument("original_root_dir", help="包含原始图片和掩码的根文件夹路径。\n例如: ./data")
    args = parser.parse_args()

    if not os.path.isdir(args.edited_root_dir):
        print(f"错误: 编辑后图片根目录不存在 -> {args.edited_root_dir}")
        sys.exit(1)
    if not os.path.isdir(args.original_root_dir):
        print(f"错误: 原始文件根目录不存在 -> {args.original_root_dir}")
        sys.exit(1)

    main(args.edited_root_dir, args.original_root_dir)
