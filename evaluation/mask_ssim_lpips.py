# -*- coding: utf-8 -*-
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# [!!! 此脚本已修改 !!!]
# 结合了 SSIM（脚本1）的计算核心 和 MSE（脚本2）的文件 I/O 结构
# [!!!] 新增: 移除了所有 resize 操作。
# [!!!] 新增: 计算前必须保证所有图像和掩码尺寸完全一致。
#

import os
import sys
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from PIL import Image
from tabulate import tabulate
import cv2
import torch
from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

#import metrics
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# --- 来自脚本 1 的核心辅助函数 (SSIM) ---
class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device=device
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        # if mask_pred is not None:
        #     mask_pred = np.array(mask_pred).astype(np.float32)
        #     img_pred = img_pred * mask_pred
        # if mask_gt is not None:
        #     mask_gt = np.array(mask_gt).astype(np.float32)
        #     img_gt = img_gt * mask_gt
        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            if mask_pred.ndim == 2:
                mask_pred = mask_pred[:, :, None]  # 扩展通道维度
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            if mask_gt.ndim == 2:
                mask_gt = mask_gt[:, :, None]  # 扩展通道维度
            img_gt = img_gt * mask_gt
        if mask_pred.max() > 1.0:
            mask_pred = mask_pred / 255.0
        if mask_gt.max() > 1.0:
            mask_gt = mask_gt / 255.0
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.ssim_metric_calculator(img_pred_tensor,img_gt_tensor)
        score = score.cpu().item()
        
        return score

    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        # if mask_pred is not None:
        #     mask_pred = np.array(mask_pred).astype(np.float32)
        #     img_pred = img_pred * mask_pred
        # if mask_gt is not None:
        #     mask_gt = np.array(mask_gt).astype(np.float32)
        #     img_gt = img_gt * mask_gt
        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            if mask_pred.ndim == 2:
                mask_pred = mask_pred[:, :, None]  # 扩展通道维度
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            if mask_gt.ndim == 2:
                mask_gt = mask_gt[:, :, None]  # 扩展通道维度
            img_gt = img_gt * mask_gt  
        if mask_pred.max() > 1.0:
            mask_pred = mask_pred / 255.0
        if mask_gt.max() > 1.0:
            mask_gt = mask_gt / 255.0 
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.lpips_metric_calculator(img_pred_tensor*2-1,img_gt_tensor*2-1)
        score = score.cpu().item()
        
        return score

def choose_win_size(h, w):
    """(来自脚本1) 为 SSIM 选择一个有效的 win_size。"""
    m = min(h, w)
    for k in (7, 5, 3):
        if m >= k:
            return k
    raise ValueError(f"Image too small for SSIM: ({h}, {w})")

def load_image(image_path):
    """(来自脚本1) 使用 Pillow 加载图像并转为 RGB。"""
    return Image.open(image_path).convert("RGB")


# --- 来自脚本 1 的核心计算 (SSIM) ---
# [!!! 此函数已重构 !!!]
def calculate_ssim(orig_img, edit_img, combined_mask_arr, unique_id):
    """
    (来自脚本1, 已重构)
    计算非掩码区域的 SSIM 分数。
    [!] 假设: 所有输入的 (Pillow 图像, Numpy 掩码) 尺寸已对齐。
    [!] 移除: 不再执行任何 resize。
    """
    
    # 1) 获取基准分辨率 (来自已加载的原图)
    W, H = orig_img.size  # Pillow size is (W, H)
    
    # 2) 安全检查 (理论上 process_for_ssim 已保证)
    if edit_img.size != (W, H):
         raise ValueError(f"Internal Error: Edited image dimension mismatch: {edit_img.size} vs Original {orig_img.size}")
    if combined_mask_arr.shape[:2] != (H, W):
        raise ValueError(f"Internal Error: Mask dimension mismatch: {combined_mask_arr.shape[:2]} vs Original {(H, W)}")

    # 3) 准备掩码
    # 掩码数组来自 CV2 (H, W)，已在 process_for_ssim 中合并
    object_mask = (combined_mask_arr > 128).astype(np.uint8)
    rest_mask = (1 - object_mask).astype(np.float32)  # 背景=1，目标=0

    # 4) 转灰度/归一化
    orig_gray = np.array(orig_img.convert('L'), dtype=np.float32) / 255.0
    edit_gray = np.array(edit_img.convert('L'), dtype=np.float32) / 255.0

    # 5) 用“乘零法”计算 Masked SSIM（与脚本1的回落逻辑一致）
    try:
        try:
            win_size = choose_win_size(H, W)
        except Exception:
            win_size = None  # 让 skimage 自选或失败
            
        image_masked = orig_gray * rest_mask
        edit_masked = edit_gray * rest_mask
        
        if win_size is not None:
            masked_score = float(ssim(image_masked, edit_masked, data_range=1.0, win_size=win_size))
        else:
            masked_score = float(ssim(image_masked, edit_masked, data_range=1.0))

    except Exception as e:
        print(f"Error during SSIM calculation for {unique_id}: {e}")
        masked_score = None # 返回 None 以便上层处理

    # 6) 返回结果
    ssim_data = {
        "unique_id": unique_id,
        "ssim_similarity_outside_mask": masked_score,
        "height": H,
        "width": W,
    }
    return ssim_data


# -----------------------------------------------------------------
# [!!! 此函数已重构 !!!]
# 用于桥接 脚本2的数据结构 和 脚本1的计算核心
# -----------------------------------------------------------------
def process_for_ssim(paths):
    """
    **[已重构]**
    处理一组（原图、编辑图、两个掩码、一个JSON）的 SSIM 计算流程。
    [!!!] 新增：在计算前严格检查所有尺寸，禁止 resize。
    """
    result = paths.copy()
    unique_id = f"{paths['topic']}_{paths['id']}"
    
    try:
        # --- 1. 读取 JSON 文件 (来自脚本2的逻辑) ---
        json_content = None
        try:
            with open(paths['json_path'], 'r', encoding='utf-8-sig') as f:
                json_content = json.load(f)
            result["json_content"] = json_content
        except Exception as json_e:
            raise IOError(f"无法读取或解析 JSON 文件 {paths['json_path']}: {json_e}")

        # --- 2. [新] 读取所有图像和掩码 ---
        # 使用 Pillow 读取图像 (因为 calculate_ssim 需要 Pillow objects)
        original_img = load_image(paths['image'])
        edited_img = load_image(paths['edit_image'])
        print(original_img)
        # 使用 CV2 读取掩码 (因为是灰度)
        mask1 = cv2.imread(paths['mask1'], cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(paths['mask2'], cv2.IMREAD_GRAYSCALE)
        
        if original_img is None: raise IOError(f"无法读取原图: {paths['image']}")
        if edited_img is None: raise IOError(f"无法读取编辑图: {paths['edit_image']}")
        if mask1 is None: raise IOError(f"无法读取掩码1: {paths['mask1']}")
        if mask2 is None: raise IOError(f"无法读取掩码2: {paths['mask2']}")

        # --- 3. [!!! 核心修改 !!!] 严格的尺寸检查 ---
        W_o, H_o = original_img.size # Pillow (W, H)
        W_e, H_e = edited_img.size   # Pillow (W, H)
        H_m1, W_m1 = mask1.shape[:2] # CV2 (H, W)
        H_m2, W_m2 = mask2.shape[:2] # CV2 (H, W)

        if not (H_o == H_e == H_m1 == H_m2 and W_o == W_e == W_m1 == W_m2):
            # 抛出异常，由外部的 try...except 捕获
            raise ValueError(
                f"尺寸不匹配 (H,W): 原图({H_o},{W_o}), "
                f"编辑图({H_e},{W_e}), "
                f"掩码1({H_m1},{W_m1}), "
                f"掩码2({H_m2},{W_m2})"
            )
        device = "cuda:6" if torch.cuda.is_available() else "cpu"
        metric_calc = MetricsCalculator(device)
        # --- 4. [新] 合并掩码 (逻辑来自旧的 combine_masks) ---
        combined_mask = cv2.bitwise_or(mask1, mask2)
        inv_mask = (combined_mask < 128).astype(np.uint8)

        # --- 5. 执行 SSIM 计算 (调用重构后的核心) ---
        # 传入已加载的对象，而不是路径
        # ssim_data = calculate_ssim(
        #     original_img, 
        #     edited_img, 
        #     combined_mask, 
        #     unique_id
        # )
        ssim_score=metric_calc.calculate_ssim(edited_img,original_img,inv_mask,inv_mask)
        lpips_score=metric_calc.calculate_lpips(edited_img,original_img,inv_mask,inv_mask)
        # --- 6. 存储结果 (格式同脚本2) ---
        result["ssim_similarity_outside_mask"] = ssim_score
        result["lpips_similarity_outside_mask"]=lpips_score
        
        # result["height"] = ssim_data.get("height")
        # result["width"] = ssim_data.get("width")

    except Exception as e:
        # [!!!] 此处将捕获
        print(f"\n跳过 {unique_id}: {e}")
        result["ssim_similarity_outside_mask"] = None
        result["json_content"] = result.get("json_content", None) # 确保 json 错误不会清空
        
    return result


# -----------------------------------------------------------------
# 函数 run_evaluation 
# (此函数与上一版相同，无需改动)
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# 函数 run_evaluation 
# [!!! 此函数已修改 !!!]
# -----------------------------------------------------------------
def run_evaluation(source_data_dir, edited_data_dir, output_filename):
    """
    **[已修改]**
    主执行函数，协调整个评估流程。
    - [!] 修改了 edited_path 的生成逻辑，以处理 "1" -> "001" 的命名不匹配。
    """
    # 步骤 1: 准备输入数据
    print("\n步骤 1/3: 正在扫描和匹配文件...")
    tasks = []
    
    # 遍历源文件夹 (例如 'data_resized_2')
    for topic_name in os.listdir(source_data_dir):
        topic_path = os.path.join(source_data_dir, topic_name)
        if not os.path.isdir(topic_path):
            continue
            
        # 遍历 Topic 文件夹 (例如 'Scientific & Technical Diagrams')
        for image_id_folder in os.listdir(topic_path):
            image_folder_path = os.path.join(topic_path, image_id_folder)
            if not os.path.isdir(image_folder_path):
                continue
            
            try:
                # image_id_folder 是 "1", "2", ...
                image_id_num = int(image_id_folder)
                # image_id_str 是 "1", "2", ...
                image_id_str = str(image_id_num) 
                
                # [!!! 新增 !!!] 
                # 创建一个 3 位零填充的 ID 字符串，用于编辑后的文件
                # e.g., 1 -> "001", 10 -> "010", 123 -> "123"
                edited_id_str = f"{image_id_num:03d}" 

            except ValueError:
                # 跳过非数字命名的文件夹
                continue
                
            # --- 1. 检查 原图 路径 ---
            # 使用 "1.jpg" (image_id_str)
            original_path = os.path.join(image_folder_path, f"{image_id_str}.jpg")
            if not os.path.exists(original_path):
                continue
            
            # --- 2. 检查 编辑图 路径 ---
            # [!!! 修改 !!!] 
            # 使用 "001_edited.jpg" (edited_id_str) 而不是 "1_edited.jpg" (image_id_folder)
            edited_path = os.path.join(edited_data_dir, topic_name, f"{edited_id_str}_edited.jpg")
            if not os.path.exists(edited_path):
                # 打印一个警告，因为这很可能是个问题
                print(f"\n跳过 {image_folder_path}: 找不到 [编辑图] '{edited_path}'")
                continue

            # --- 3. 检查 掩码1 路径 ---
            # 使用 "1_mask.jpg" (image_id_str)
            mask1_basename = f"{image_id_str}_mask.jpg"
            mask1_path = os.path.join(image_folder_path, mask1_basename)
            if not os.path.exists(mask1_path):
                continue
            
            # --- 4. 动态查找 掩码2 和 JSON 文件 ---
            # (此部分逻辑不变)
            mask2_path = None
            json_path = None
            other_masks_found = []
            json_files_found = []

            try:
                all_files_in_dir = os.listdir(image_folder_path)
                for f in all_files_in_dir:
                    # [!] 注意：这里的掩码2也应遵循 "1_..._mask.jpg" 规则
                    if f.endswith("_mask.jpg") and f != mask1_basename:
                        other_masks_found.append(f)
                    elif f.endswith(".json"):
                        json_files_found.append(f)
            except Exception as e:
                print(f"\n跳过 {image_folder_path}: 无法读取文件夹内容。错误: {e}")
                continue

            # --- 5. 验证 掩码2 ---
            # (此部分逻辑不变)
            if len(other_masks_found) == 0:
                continue
            elif len(other_masks_found) > 1:
                print(f"\n跳过 {image_folder_path}: 找到多个 [掩码2] 候选: {other_masks_found}。")
                continue
            else:
                mask2_path = os.path.join(image_folder_path, other_masks_found[0])

            # --- 6. 验证 JSON 文件 ---
            # (此部分逻辑不变)
            if len(json_files_found) == 0:
                continue
            elif len(json_files_found) > 1:
                print(f"\n跳过 {image_folder_path}: 找到多个 .json 文件: {json_files_found}。")
                continue
            else:
                json_path = os.path.join(image_folder_path, json_files_found[0])

            # --- 7. 所有5个文件均已找到并验证 ---
            # (此部分逻辑不变)
            tasks.append({
                "topic": topic_name,
                "id": image_id_folder, # 仍然使用 "1" 作为原始 ID
                "image": original_path,
                "edit_image": edited_path,
                "mask1": mask1_path,
                "mask2": mask2_path,
                "json_path": json_path 
            })

    if not tasks:
        print("\n错误：未能匹配到任何有效的文件组。请检查文件夹结构和文件命名。")
        return
    
    print(f"成功匹配 {len(tasks)} 组文件。")

    # 步骤 2: 并行计算
    # (此部分逻辑不变)
    print(f"\n步骤 2/3: 正在为 {len(tasks)} 组图片计算 SSIM (Outside Mask)...")
    final_results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_for_ssim, paths) for paths in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="计算进度"):
            result = future.result()
            if result:
                final_results.append(result)

    # 步骤 3: 生成报告和输出文件
    # (此部分逻辑不变)
    print("\n步骤 3/3: 正在生成评估报告...")
    
    final_results.sort(key=lambda x: (x.get('topic', ''), x.get('id', '')))
    
    scores = [r.get("ssim_similarity_outside_mask") for r in final_results if isinstance(r.get("ssim_similarity_outside_mask"), (int, float))]
    
    if scores:
        summary_stats = {
            "count": len(scores),
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
        }
        print("\nSSIM (Outside Mask) 评估摘要:")
        print(tabulate([[
            summary_stats["count"],
            f"{summary_stats['mean']:.4f}",
            f"{summary_stats['std']:.4f}",
            f"{summary_stats['min']:.4f}",
            f"{summary_stats['max']:.4f}",
        ]], headers=["有效数量", "平均值 (越高越好)", "标准差", "最小值", "最大值"], tablefmt="grid"))
    else:
        print("未能计算出任何有效的 SSIM 分数。")

    with open(output_filename, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n详细结果已保存至: {output_filename}")


# -----------------------------------------------------------------
# 命令行入口 (来自脚本2)
# -----------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("\n用法: python <脚本名>.py <源数据文件夹> <编辑后数据文件夹>")
        print("示例: python your_script_name.py ./data_resized_2 ./flux")
        sys.exit(1)

    source_data_dir = sys.argv[1]
    edited_data_dir = sys.argv[2]

    for d, name in [(source_data_dir, "源数据"), (edited_data_dir, "编辑后数据")]:
        if not os.path.isdir(d):
            print(f"错误: {name}文件夹 '{d}' 不存在或不是一个有效的目录。")
            sys.exit(1)
            
    output_filename = "evaluation_ssim/data2/gemini.json"
    
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
        
    run_evaluation(source_data_dir, edited_data_dir, output_filename)











