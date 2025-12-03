import os
import sys
import logging
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import openai
from datetime import datetime

# --- 1. 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
)
logger = logging.getLogger("ImageEvaluator")

# --- 2. 加载环境变量和配置 ---
load_dotenv()

# --- 3. 配置 OpenAI API 客户端 ---
try:
    client = openai.OpenAI(
        base_url=os.getenv('OPENAI_API_URL'),
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=float(os.getenv('OPENAI_TIMEOUT', 900.0)),
        max_retries=2,
    )
    API_MODEL = 'gpt-4o'
    logger.info(f"API客户端已配置，使用模型: {API_MODEL}")
except Exception as e:
    logger.error(f"OpenAI API 客户端配置失败: {e}")
    sys.exit(1)

# --- 4. 固定路径配置 ---
FIXED_INPUT_DIR = Path("./data2_resized_fix")#原图片路径
FIXED_PRED_IMAGES_BASE = Path("./edited_images_2_fix")#编辑后图片路径
FIXED_OUTPUT_BASE = Path("./gpt_evaluation_4")#保存结果路径
FIXED_FAILED_BASE = Path("./gpt_evaluation_failed_4")#保存失败结果路径
FIXED_WORKERS = 20

# --- 5. 辅助函数 ---
def encode_image_to_base64(image_path: str) -> str:
    """将图像编码为base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"编码图片失败 {image_path}: {e}")
        return None

def scan_model_directories(pred_images_base: Path):
    """扫描并返回所有模型目录"""
    model_dirs = []
    for item in sorted(pred_images_base.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            model_dirs.append(item.name)
    return model_dirs

def scan_category_directories(input_dir: Path):
    """扫描并返回所有类别目录"""
    category_dirs = []
    for item in sorted(input_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            category_dirs.append(item.name)
    return category_dirs

def select_from_list(items, item_type="目录", allow_all=False):
    """通用的交互式选择函数"""
    print("\n" + "=" * 50)
    print(f"检测到以下{item_type}:")
    print("=" * 50)
    if allow_all:
        print("0. [全部处理]")
    for idx, item_name in enumerate(items, 1):
        print(f"{idx}. {item_name}")
    print("=" * 50)
    
    while True:
        try:
            choice = input(f"\n请输入要处理的{item_type}编号 (输入 'q' 退出): ").strip()
            if choice.lower() == 'q':
                print("退出程序。")
                sys.exit(0)
            
            choice_idx = int(choice)
            
            if allow_all and choice_idx == 0:
                print(f"\n✓ 已选择: 全部{item_type}\n")
                return None  # 返回 None 表示全部
            
            choice_idx -= 1
            if 0 <= choice_idx < len(items):
                selected = items[choice_idx]
                print(f"\n✓ 已选择: {selected}\n")
                return selected
            else:
                print(f"❌ 请输入 {'0 到' if allow_all else '1 到'} {len(items)} 之间的数字")
        except ValueError:
            print("❌ 输入无效，请输入数字或 'q'")

def convert_number_to_filename(number_str: str) -> str:
    """将三位数字编号转换为文件名用的编号（去除前导零）"""
    return str(int(number_str))

# --- (已删除) check_task_scored 函数 ---
# (此函数已删除，跳过逻辑现在在 main 和 collect_tasks_for_category 中)

# --- 6. 评估维度的 Prompt --- (内容不变)
PROMPTS = {
    "IF": """You are a professional image editing evaluation specialist.

You will receive for each sample:
- input_image: the original unedited image  
- pred_image: the candidate image produced by the model under test  
- prompt: the user editing instruction  
- editing_method: the specific operation type (text_delete | text_insert | text_change | text_correction | font_change | translation | position_shift | rotation | scaling)


Your job:
Evaluate **instruction_following (IF)** on a 0–5 scale. Focus only on whether the pred_image performs exactly the requested operation type (editing_method) and nothing else.  
Do **not** reward or penalize aspects unrelated to instruction compliance.

Penalty clause:  
If pred_image performs wrong operation type, multiple operations when only one requested, or fails to execute the specified operation, subtract points accordingly.

Scoring Scale (0–5):
5 = Exactly the specified operation, no extra changes  
4 = Correct operation with one tiny extra modification  
3 = Generally correct but some unnecessary additions/omissions  
2 = Operation attempted but with significant extra changes  
1 = Wrong operation type or major execution errors  
0 = No detectable change or completely wrong approach

Examples:
- Example 5: Instruction "Delete the word 'SALE'" with editing_method "text_delete" and pred_image shows only that word removed. Score 5.
- Example 3: Deleted "SALE" correctly but also changed font of other text when not requested. Score 3.
- Example 1: Used "text_change" (changed SALE to something else) instead of "text_delete", or deleted wrong text. Score 1.

Reasoning Steps:
1. Identify the requested operation from prompt and editing_method.  
2. Compare pred_image with input_image.  
3. Check if only the specified operation was performed.  
4. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"IF": X, "rationale": "short explanation of why this score"}""",

    "TA": """You are an expert evaluator of text-in-image editing.

You will receive for each sample:
- input_image: the original unedited image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction


Your job:
Evaluate **text_accuracy (TA)** on a 0–5 scale. Focus only on whether the text content in pred_image matches the instruction.
Do **not** reward or penalize aspects unrelated to text accuracy (e.g., font style, layout, background).

Penalty clause:
If the text content is wrong, misspelled, mistranslated, or unchanged when should change, subtract points accordingly.

Scoring Scale (0–5):
5 = Target text fully correct in content/spelling/case
4 = Correct but with one minor spelling/formatting inconsistency
3 = Core text mostly correct but noticeable error remains
2 = Several text elements wrong or inconsistent
1 = Text largely incorrect or unchanged when should change
0 = Completely wrong text or no change

Examples:
- Example 5: Instruction "Change 'Sale' to 'Sold Out'" and pred_image shows exactly "Sold Out". Score 5.
- Example 4: Shows "Sold out" with lowercase 'o'. Score 4.
- Example 3: Shows "Sold Oat" instead of "Sold Out". Score 3.
- Example 1: Still shows "Sale" unchanged. Score 1.

Reasoning Steps:
1. Identify the requested text change from prompt.
2. Extract/compare text from input_image (before) and pred_image (after).
3. Check text content accuracy strictly against instruction.
4. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"TA": X, "rationale": "short explanation of why this score"}""",

    "VC": """You are an expert evaluator of text-in-image editing.

You will receive for each sample:
- input_image: the original unedited image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction

Your job:
Evaluate **visual_consistency (VC)** on a 0–5 scale. Focus only on whether the new or edited elements visually integrate with the rest of the image in font/weight, alignment/perspective, edge anti-aliasing, color/lighting blending.
Do **not** reward or penalize unrelated content changes.

Penalty clause:
If new text or objects look pasted, misaligned, haloed, or style-mismatched, subtract points.

Scoring Scale (0–5):
5 = Perfect integration: matching font, size, color, alignment; no visible artifacts
4 = Very good integration: only tiny mismatch
3 = Moderate integration: visible mismatch (slight halo, mild misalignment)
2 = Poor integration: obvious halo or style mismatch
1 = Very poor integration: pasted look or misaligned
0 = Completely inconsistent/unreadable

Examples:
- Example 5: New text matches surrounding sign perfectly in font, color, and perspective. Score 5.
- Example 3: New text correct content but with mild white halo around edges. Score 3.
- Example 1: New text obviously pasted with wrong color and misaligned. Score 1.

Reasoning Steps:
1. Identify edited elements in pred_image.
2. Compare their visual integration quality to input_image.
3. Check for artifacts, misalignment, or style mismatches.
4. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"VC": X, "rationale": "short explanation of why this score"}""",

    "LP": """You are an expert evaluator of text-in-image editing.

You will receive for each sample:
- input_image: the original unedited image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction

Your job:
Evaluate **layout_preservation (LP)** on a 0–5 scale. Focus only on whether non-target areas remained unchanged compared to input_image.
Do **not** reward or penalize the quality of instructed changes themselves.

Penalty clause:
If pred_image alters background, other objects or layout unnecessarily, subtract points.

Scoring Scale (0–5):
5 = All non-target areas unchanged
4 = Almost unchanged, only tiny disturbance
3 = Minor unrelated disturbance
2 = Several unrelated changes
1 = Large portions altered unnecessarily
0 = Layout completely different

Examples:
- Example 5: Only target text changed from "Open" to "Closed", rest of sign and background untouched. Score 5.
- Example 3: Target text changed correctly but background slightly shifted or blurred. Score 3.
- Example 1: New objects added and overall layout composition different. Score 1.

Reasoning Steps:
1. Identify target and non-target areas from prompt.
2. Compare pred_image with input_image.
3. Detect any unrelated changes to non-target areas.
4. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"LP": X, "rationale": "short explanation of why this score"}""",

    "SE": """You are an expert evaluator of text-in-image editing.

You will receive for each sample:
- input_image: the original unedited image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction
- knowledge_prompt: semantic expectation or logical consequence

Your job:
Evaluate **semantic_expectation (SE)** on a 0–5 scale.
Use the knowledge_prompt to understand linked expectations and check whether these are reflected in pred_image.
Focus only on whether these linked expectations are satisfied.
Do **not** reward or penalize unrelated features.

Penalty clause:
If pred_image fails to reflect expected linked changes or contradicts knowledge_prompt, subtract points.

Scoring Scale (0–5):
5 = Fully matches all linked/knowledge expectations
4 = Matches but with one small linked detail missed
3 = Partially satisfies: core effect present but significant linked effect missing
2 = Several linked effects missing or inconsistent
1 = Contradicts or ignores knowledge expectation
0 = No evidence of linked effect

Examples:
- Example 5: Instruction "Remove chili from dish" + knowledge_prompt "Removing 'chili' implies spicy icon should be absent". Pred_image shows dish without chili and spicy icon gone. Score 5.
- Example 3: Chili removed correctly but spicy icon still present. Score 3.
- Example 1: Chili still present and spicy icon still present, ignoring both instruction and semantic expectation. Score 1.

Reasoning Steps:
1. Identify instruction and linked expectations from prompt + knowledge_prompt.
2. Compare pred_image with input_image.
3. Check whether linked expectations are satisfied beyond direct instruction.
4. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"SE": X, "rationale": "short explanation of why this score"}"""
}

# --- 7. 调用 GPT-5 进行评估 --- (内容不变)
# --- 7. 调用 GPT-5 进行评估 --- (已修改)
def call_gpt5_for_evaluation(dimension: str, input_image_base64: str, pred_image_base64: str, 
                             prompt: str, editing_method: str = None, 
                             knowledge_prompt: str = None, retry_count: int = 3):
    """
    (已修改)
    调用 GPT-5 进行单个维度的评估
    - 删除了 output_image_base64 参数
    """
    for attempt in range(retry_count):
        try:
            system_prompt = PROMPTS[dimension]
            
            # 构建用户消息内容
            user_content = [
                {"type": "text", "text": f"User editing instruction (prompt): {prompt}"}
            ]
            
            if editing_method:
                user_content.append({"type": "text", "text": f"Editing method: {editing_method}"})
            
            if knowledge_prompt and dimension == "SE":
                user_content.append({"type": "text", "text": f"Knowledge prompt: {knowledge_prompt}"})
            
            # (已修改) 添加图片 - 删除了 output_image
            user_content.extend([
                {"type": "text", "text": "Input image (original):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_image_base64}"}},
                {"type": "text", "text": "Pred image (model output):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pred_image_base64}"}},
                
                # (已删除) Output image (expected reference)
                
                {"type": "text", "text": "\n**CRITICAL REQUIREMENT: Your entire response must be ONLY a valid JSON object in this exact format:**\n" + 
                                       f'{{\"{dimension}\": <score 0-5>, \"rationale\": \"<explanation>\"}}\n' +
                                       "**NO markdown, NO code blocks, NO explanations outside the JSON, NO think tags. Just the raw JSON object.**"}
            ])
            
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            # ... (JSON 清理和解析逻辑不变) ...
            result_text = response.choices[0].message.content.strip()
            
            import re
            result_text = result_text.replace('<think>', '').replace('</think>', '')
            result_text = result_text.replace('```json', '').replace('```', '')
            
            first_brace = result_text.find('{')
            if first_brace != -1:
                brace_count = 0
                json_end = -1
                for i in range(first_brace, len(result_text)):
                    if result_text[i] == '{':
                        brace_count += 1
                    elif result_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                if json_end != -1:
                    result_text = result_text[first_brace:json_end]
            
            try:
                result_json = json.loads(result_text)
                if dimension in result_json and "rationale" in result_json:
                    return result_json
                else:
                    logger.warning(f"JSON 缺少必要字段，尝试重试 ({attempt + 1}/{retry_count})")
                    logger.debug(f"解析的JSON: {result_json}")
            except json.JSONDecodeError as je:
                logger.warning(f"JSON 解析失败 (尝试 {attempt + 1}/{retry_count}): {result_text[:200]}")
                if attempt == retry_count - 1:
                    logger.error(f"无法解析 GPT-5 返回的 JSON: {result_text}")
                    return {dimension: 0, "rationale": "JSON parsing failed"}
                
        except Exception as e:
            logger.error(f"调用 GPT-5 评估 {dimension} 时失败 (尝试 {attempt + 1}/{retry_count}): {e}")
            if attempt == retry_count - 1:
                return {dimension: 0, "rationale": f"API call failed: {str(e)}"}

# --- 8. 评估单个任务 --- (内容不变)
# --- 8. 评估单个任务 --- (已修改)
# --- 8. 评估单个任务 --- (已修改)
def evaluate_task(task: dict, sample_dir: Path, pred_images_dir: Path):
    """
    (已修改)
    根据任务内容，评估各项指标（IF, TA, VC, LP, SE）
    - 修改点: 删除了所有 output_image (标准答案) 的逻辑
    """
    
    # --- 1. 从 task 字典中获取数据 ---
    
    task_id = task.get("id")
    prompt = task.get("prompt")
    editing_method = task.get("editing_method")
    knowledge_prompt = task.get("knowledge_prompt", None)

    # (新增) 直接从 task 中获取文件名
    input_image_filename = task.get("input_image")
    
    # (已删除) output_image_filename
    
    # (新增) 健壮性检查
    if not input_image_filename:
        raise ValueError(f"任务 {task_id} 的 JSON 中缺少 'input_image' 键或值为空")
    
    # (已删除) output_image 健壮性检查

    
    # --- 2. (已修改) 拼接路径 ---
    
    # (已修改) 1. input_image 路径: (使用 task 中的 "input_image" 值)
    input_image_path = sample_dir / input_image_filename
    
    # (已删除) 2. output_image 路径
    
    # 3. pred_image 路径: (此逻辑保持不变)
    sample_number = sample_dir.name
    pred_image_path = pred_images_dir / f"{sample_number}_edited.jpg"
    
    
    # --- 3. (已修改) 检查、编码和评估 ---
    
    # (已修改) 检查文件是否存在 - 删除了 output
    for img_path, img_type in [(input_image_path, "input"), (pred_image_path, "pred")]:
        if not img_path.exists():
            raise FileNotFoundError(f"{img_type} 图片未找到: {img_path}")
    
    # (已修改) 编码图片为 base64 - 删除了 output
    input_image_base64 = encode_image_to_base64(str(input_image_path))
    pred_image_base64 = encode_image_to_base64(str(pred_image_path))
    
    # (已修改) 检查编码 - 删除了 output
    if not all([input_image_base64, pred_image_base64]):
        raise ValueError("图片编码失败")
    
    # (已修改) 评估各个维度 - 删除了 output_image_base64 参数
    evaluation_results = {}
    
    logger.info(f"[{task_id}] 开始评估 IF...")
    evaluation_results["IF"] = call_gpt5_for_evaluation(
        "IF", input_image_base64, pred_image_base64, 
        prompt, editing_method
    )
    
    logger.info(f"[{task_id}] 开始评估 TA...")
    evaluation_results["TA"] = call_gpt5_for_evaluation(
        "TA", input_image_base64, pred_image_base64, prompt
    )
    
    logger.info(f"[{task_id}] 开始评估 VC...")
    evaluation_results["VC"] = call_gpt5_for_evaluation(
        "VC", input_image_base64, pred_image_base64, prompt
    )
    
    logger.info(f"[{task_id}] 开始评估 LP...")
    evaluation_results["LP"] = call_gpt5_for_evaluation(
        "LP", input_image_base64, pred_image_base64, prompt
    )
    
    # 如果有 knowledge_prompt，评估 SE
    if knowledge_prompt:
        logger.info(f"[{task_id}] 开始评估 SE...")
        evaluation_results["SE"] = call_gpt5_for_evaluation(
            "SE", input_image_base64, pred_image_base64, 
            prompt, knowledge_prompt=knowledge_prompt
        )
    
    return evaluation_results
# --- 9. 处理单个任务 --- (内容不变)
def process_single_task(task: dict, sample_dir: Path, pred_images_dir: Path, output_dir: Path, failed_dir: Path):
    """处理单个任务，不再写入文件，而是返回结果字典或失败字典"""
    task_id = task.get("id")
    
    try:
        evaluation_results = evaluate_task(task, sample_dir, pred_images_dir)
        
        result = {
            "task_id": task_id,
            "evaluation_results": evaluation_results
        }
        
        logger.info(f"✓ [{task_id}] 评估完成")
        return {"status": "success", "data": result}
        
    except Exception as e:
        logger.error(f"✗ [{task_id}] 处理失败: {e}")
        return {"status": "error", "task_id": task_id, "error": str(e)}

# --- 10. (新增) 收集单个类别目录中的任务 ---
def collect_tasks_for_category(input_dir: Path, selected_category: str, pred_images_dir: Path, 
                             already_scored_set: set):
    """
    (新增)
    扫描单个类别目录，解析JSON，过滤已评测任务，并返回待运行的任务列表。
    
    返回: 
        (tasks_to_run, skipped_tasks_count)
        tasks_to_run: [(task_dict, sample_dir_path), ...]
    """
    
    logger.info(f"--- 正在扫描类别: {selected_category} ---")
    
    tasks_to_run = []
    skipped_tasks = 0
    
    category_dir = input_dir / selected_category
    
    if not category_dir.is_dir():
        logger.error(f"选定的类别目录不存在: {category_dir}")
        return [], 0
    
    if not pred_images_dir.is_dir():
        logger.warning(f"pred图片目录不存在，跳过类别: {pred_images_dir}")
        return [], 0
    
    # 遍历样本目录 (001, 002, 003 等)
    for sample_dir in sorted(category_dir.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name.startswith('.'):
            continue
        
        # --- (JSON 查找逻辑 - 保持不变) ---
        sample_number = sample_dir.name
        stripped_number = convert_number_to_filename(sample_number)
        suffixes_to_check = [
            f"_{sample_number}.json", f".{sample_number}.json",
            f"-{sample_number}.json", f"{sample_number}.json",
        ]
        if stripped_number != sample_number:
            suffixes_to_check.extend([
                f"_{stripped_number}.json", f".{stripped_number}.json",
                f"-{stripped_number}.json", f"{stripped_number}.json",
            ])
        
        json_path = None
        all_files_in_dir = [f for f in sample_dir.iterdir() if f.is_file()]
        found = False
        for suffix in suffixes_to_check:
            for file_path in all_files_in_dir:
                file_name = file_path.name
                if file_name.endswith(suffix):
                    if file_name == suffix:
                        json_path = file_path
                        logger.debug(f"找到JSON文件 (完全匹配): {file_name}")
                        found = True
                        break
                    separator_index = len(file_name) - len(suffix)
                    if file_name[separator_index] == suffix[0]: 
                        json_path = file_path
                        logger.debug(f"找到JSON文件 (后缀匹配): {file_name}")
                        found = True
                        break
            if found:
                break
        # --- (JSON 查找逻辑结束) ---

        if json_path is None:
            logger.warning(f"在 {sample_dir} 中未找到任何JSON文件，已尝试后缀: {suffixes_to_check}")
            continue

        try:
            # --- (JSON 读取、修复、解析逻辑 - 保持不变) ---
            with open(json_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
            content = content.strip()
            if content.endswith(",}"): content = content[:-2] + "}"
            if content.endswith(",]"): content = content[:-2] + "]"
            
            try: data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"[{selected_category}/{sample_number}] JSON 格式错误: {e}")
                continue
            
            if isinstance(data, dict): tasks = data.get("tasks", [data])
            elif isinstance(data, list): tasks = data
            elif isinstance(data, str):
                try:
                    tasks = json.loads(data)
                    if isinstance(tasks, dict): tasks = [tasks]
                except:
                    logger.error(f"[{selected_category}/{sample_number}] 无法解析字符串 JSON")
                    continue
            else:
                logger.error(f"[{selected_category}/{sample_number}] 不支持的 JSON 格式: {type(data)}")
                continue
            if not isinstance(tasks, list): tasks = [tasks]
            # --- (JSON 解析结束) ---
            
            logger.info(f"[{selected_category}/{sample_number}] 找到 {len(tasks)} 个任务")
            
            new_tasks_in_file = 0
            # (已修改) 过滤任务
            for task in tasks:
                if not isinstance(task, dict):
                    logger.error(f"跳过非字典任务: {type(task)}")
                    continue
                
                task_id = task.get("id", "")
                if not task_id:
                    logger.error(f"任务缺少 id 字段，跳过")
                    continue
                
                # (已修改) 核心跳过逻辑
                task_key = (selected_category, task_id)
                
                if task_key in already_scored_set:
                    # logger.info(f"[{selected_category}/{sample_number}] SKIPPING 已评测任务: {task_id}") # (注释掉以减少日志)
                    skipped_tasks += 1
                    continue
                    
                # (已修改) 添加到待运行列表
                tasks_to_run.append((task, sample_dir)) 
                new_tasks_in_file += 1
            
            if new_tasks_in_file == 0 and len(tasks) > 0:
                logger.info(f"[{selected_category}/{sample_number}] 所有任务已评测，跳过")
            elif new_tasks_in_file > 0:
                logger.info(f"[{selected_category}/{sample_number}] 新增 {new_tasks_in_file} 个待评分任务")
                
        except Exception as e:
            logger.error(f"读取 JSON 文件失败: {json_path}, 错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"--- 类别 [{selected_category}] 扫描完成: 找到 {len(tasks_to_run)} 个新任务, 跳过 {skipped_tasks} 个 ---")
    return tasks_to_run, skipped_tasks


# --- 11. 主函数 --- (已修改)
def main():

    # 全局统计 (用于最后的总结)
    total_all = 0
    success_all = 0
    failed_all = 0
    skipped_all = 0
    
    # --- 路径检查 (不变) ---
    if not FIXED_INPUT_DIR.is_dir():
        logger.error(f"输入目录不存在: {FIXED_INPUT_DIR}")
        sys.exit(1)
    if not FIXED_PRED_IMAGES_BASE.is_dir():
        logger.error(f"pred图片基础目录不存在: {FIXED_PRED_IMAGES_BASE}")
        sys.exit(1)
    FIXED_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    FIXED_FAILED_BASE.mkdir(parents=True, exist_ok=True)
    
    
    # --- 1. 扫描并选择模型 (不变) ---
    model_dirs = scan_model_directories(FIXED_PRED_IMAGES_BASE)
    print(model_dirs)
    if not model_dirs:
        logger.error(f"在 {FIXED_PRED_IMAGES_BASE} 下未找到任何模型目录")
        sys.exit(1)
    selected_model = select_from_list(model_dirs, "模型", allow_all=True)
    
    if selected_model is None:
        models_to_process = model_dirs
        logger.info(f"\n将处理所有 {len(models_to_process)} 个模型")
    else:
        models_to_process = [selected_model]
    
    # --- 2. 扫描并选择类别 (不变) ---
    category_dirs = scan_category_directories(FIXED_INPUT_DIR)
    if not category_dirs:
        logger.error(f"在 {FIXED_INPUT_DIR} 下未找到任何类别目录")
        sys.exit(1)
    selected_category = select_from_list(category_dirs, "类别", allow_all=True)
    
    if selected_category is None:
        categories_to_process = category_dirs
        logger.info(f"\n将处理所有 {len(categories_to_process)} 个类别")
    else:
        categories_to_process = [selected_category]
    
    # ==================================================================
    # --- 4. (已重构) 批量处理 ---
    # ==================================================================
    for model in models_to_process:
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"开始处理模型: {model}")
        logger.info(f"{'#'*60}\n")
        
        # --- (按模型加载/保存逻辑 - 保持不变) ---
        results_filename = FIXED_OUTPUT_BASE / f"{model}.json"
        failed_filename = FIXED_FAILED_BASE / f"{model}.json"
        
        model_results = []
        model_failures = []
        model_scored_set = set() # (category, task_id)
        
        if results_filename.exists():
            logger.info(f"正在加载模型 [{model}] 的已有结果: {results_filename}")
            try:
                with open(results_filename, "r", encoding="utf-8") as f:
                    model_results = json.load(f)
                    if not isinstance(model_results, list): model_results = []
                
                for item in model_results:
                    if isinstance(item, dict):
                        key = (item.get('category'), item.get('task_id'))
                        if all(key): model_scored_set.add(key)
                logger.info(f"成功加载 {len(model_results)} 条 [{model}] 的已有评估。")
            except Exception as e:
                logger.error(f"加载 {results_filename} 失败: {e}。将作为新运行开始。")
                model_results = []; model_scored_set = set()
        else:
            logger.info(f"未找到 [{model}] 的旧结果，将创建新文件: {results_filename}")

        if failed_filename.exists():
            try:
                with open(failed_filename, "r", encoding="utf-8") as f:
                    model_failures = json.load(f)
                    if not isinstance(model_failures, list): model_failures = []
            except Exception as e:
                logger.error(f"加载 {failed_filename} 失败: {e}。")
                model_failures = []
        
        pred_images_base_for_model = FIXED_PRED_IMAGES_BASE / model
        
        # --- (此模型的统计 - 保持不变) ---
        total_model_new = 0
        success_model_new = 0
        failed_model_new = 0
        skipped_model_new = 0
        
        # ==================================================================
        # (已修改) 核心逻辑变更:
        # 1. 先收集所有类别的任务
        # 2. 再将所有任务放入一个线程池
        # ==================================================================
        
        logger.info(f"--- 正在为模型 [{model}] 收集所有类别的任务... ---")
        
        all_tasks_to_submit = [] # 格式: (task, sample_dir, pred_images_dir, category)
        
        for category in categories_to_process:
            pred_images_dir_for_cat = pred_images_base_for_model / category
            
            # (已修改) 调用新函数
            tasks_for_this_cat, skipped_in_cat = collect_tasks_for_category(
                input_dir = FIXED_INPUT_DIR,
                selected_category = category,
                pred_images_dir = pred_images_dir_for_cat,
                already_scored_set = model_scored_set
            )
            
            skipped_model_new += skipped_in_cat
            
            # (已修改) 准备提交列表
            for task, sample_dir in tasks_for_this_cat:
                all_tasks_to_submit.append((task, sample_dir, pred_images_dir_for_cat, category))
        
        total_model_new = len(all_tasks_to_submit)
        
        if total_model_new == 0:
            logger.info(f"模型 [{model}] 没有需要处理的新任务 (已跳过 {skipped_model_new} 个已评测任务)。")
            logger.info(f"{'#'*60}\n")
            continue # (新增) 跳过此模型
        
        logger.info(f"--- 模型 [{model}] 共找到 {total_model_new} 个新任务 (跳过 {skipped_model_new} 个已评测)，开始并发处理... ---")
        
        # (已修改) 使用一个大的线程池处理所有任务
        with ThreadPoolExecutor(max_workers=FIXED_WORKERS, thread_name_prefix=f'{model}-Evaluator') as executor:
            
            # (已修改) 提交任务并存储上下文
            futures = {}
            for task, sample_dir, pred_images_dir, category in all_tasks_to_submit:
                future = executor.submit(process_single_task, task, sample_dir, pred_images_dir, FIXED_OUTPUT_BASE, FIXED_FAILED_BASE)
                futures[future] = (task, category) # 存储上下文 (task, category)
            
            # (已修改) 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {model}"):
                task_obj, category = futures[future] # 获取上下文
                task_id = task_obj.get("id", "Unknown ID")
                
                try:
                    result_object = future.result()
                    
                    if result_object.get("status") == "success":
                        res_data = result_object["data"]
                        res_data["model"] = model
                        res_data["category"] = category
                        model_results.append(res_data) # (修改) 追加到 model_results
                        success_model_new += 1
                    else:
                        fail_data = result_object
                        fail_data["model"] = model
                        fail_data["category"] = category
                        model_failures.append(fail_data) # (修改) 追加到 model_failures
                        failed_model_new += 1
                        
                except Exception as e:
                    # 捕获线程执行中的异常
                    logger.error(f"任务 {task_id} 执行异常: {e}")
                    model_failures.append({
                        "status": "error", 
                        "task_id": task_id, 
                        "model": model,
                        "category": category,
                        "error": f"Future execution error: {str(e)}"
                    })
                    failed_model_new += 1

        # ==================================================================
        # (逻辑修改结束)
        # ==================================================================

        # --- (保存和日志记录逻辑 - 保持不变) ---
        logger.info(f"\n{'#'*60}")
        logger.info(f"模型 [{model}] 的所有任务处理完成，正在保存文件...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if results_filename.exists():
            try:
                backup_name = results_filename.with_suffix(f".{timestamp}.bak.json")
                shutil.copy(results_filename, backup_name)
                logger.info(f"已备份旧的 {results_filename.name} 到: {backup_name.name}")
            except Exception as e:
                logger.warning(f"备份 {results_filename.name} 失败: {e}")

        if failed_filename.exists():
             try:
                backup_name = failed_filename.with_suffix(f".{timestamp}.bak.json")
                shutil.copy(failed_filename, backup_name)
                logger.info(f"已备份旧的 {failed_filename.name} 到: {backup_name.name}")
             except Exception as e:
                logger.warning(f"备份 {failed_filename.name} 失败: {e}")

        try:
            with open(results_filename, "w", encoding="utf-8") as f:
                json.dump(model_results, f, indent=4, ensure_ascii=False)
            logger.info(f"✓ [{model}] 总共 {len(model_results)} 条成功结果已保存到: {results_filename}")
        except Exception as e:
            logger.error(f"✗ 保存 [{model}] 结果文件失败: {e}")

        try:
            with open(failed_filename, "w", encoding="utf-8") as f:
                json.dump(model_failures, f, indent=4, ensure_ascii=False)
            logger.info(f"✓ [{model}] 总共 {len(model_failures)} 条失败记录已保存到: {failed_filename}")
        except Exception as e:
            logger.error(f"✗ 保存 [{model}] 失败文件失败: {e}")

        logger.info(f"\n--- 模型 [{model}] 本次运行统计 ---")
        logger.info(f"  新增任务 (本次运行): {total_model_new}")
        logger.info(f"  └─ 成功: {success_model_new}")
        logger.info(f"  └─ 失败: {failed_model_new}")
        logger.info(f"  跳过任务 (已评测): {skipped_model_new}")
        logger.info(f"  --- 累计总数 (在文件中) ---")
        logger.info(f"  累计成功: {len(model_results)}")
        logger.info(f"  累计失败: {len(model_failures)}")
        logger.info(f"{'#'*60}\n")
        
        # 累加到全局统计
        total_all += total_model_new
        success_all += success_model_new
        failed_all += failed_model_new
        skipped_all += skipped_model_new

    # --- 5. (全局统计日志 - 保持不变) ---
    logger.info("\n" + "=" * 60)
    logger.info(" 🎉 所有选定模型的批量任务完成 ")
    logger.info("=" * 60)
    logger.info(" 全局统计摘要 (本次运行) ")
    logger.info("=" * 60)
    logger.info(f"处理模型数: {len(models_to_process)}")
    logger.info(f"处理类别数: {len(categories_to_process)}")
    logger.info(f"总计新增任务 (所有模型): {total_all}")
    logger.info(f"└─ 成功: {success_all}")
    logger.info(f"└─ 失败: {failed_all}")
    logger.info(f"总计跳过任务 (所有模型): {skipped_all}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()