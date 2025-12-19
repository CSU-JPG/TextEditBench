"""
LLM-based Image Editing Evaluation Script
License: MIT
Description: Evaluates image editing quality based on Instruction Following, Text Accuracy, 
Visual Consistency, Layout Preservation, and Semantic Expectation using multimodal LLMs.
"""

import os
import sys
import logging
import json
import base64
import argparse
import shutil
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm
import openai

# --- 1. Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ImageEvaluator")

# --- 2. Environment Setup ---
load_dotenv()

# --- 3. OpenAI Client Setup ---
def setup_client():
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            # Fallback for checking if user provided it via env but maybe named differently
            logger.warning("OPENAI_API_KEY not found in environment variables.")
        
        client = openai.OpenAI(
            base_url=os.getenv('OPENAI_API_URL'), # Optional custom base URL
            api_key=api_key,
            timeout=float(os.getenv('OPENAI_TIMEOUT', 900.0)),
            max_retries=2,
        )
        model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        logger.info(f"API Client configured. Using model: {model}")
        return client, model
    except Exception as e:
        logger.error(f"Failed to configure OpenAI client: {e}")
        sys.exit(1)

CLIENT, API_MODEL = setup_client()

# --- 4. Evaluation Prompts ---
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
# --- 5. Helper Functions ---

def encode_image_to_base64(image_path: Path) -> str:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None

def convert_number_to_filename(number_str: str) -> str:
    """Converts '001' to '1' for filename matching if necessary."""
    try:
        return str(int(number_str))
    except ValueError:
        return number_str

def select_from_list(items, item_type="item", allow_all=True, auto_select_all=False):
    """Interactive selection helper. If auto_select_all is True, skips interaction."""
    if not items:
        return None
        
    if auto_select_all:
        return None # None implies 'All'

    print("\n" + "=" * 50)
    print(f"Detected {item_type}s:")
    print("=" * 50)
    if allow_all:
        print("0. [Process All]")
    for idx, item_name in enumerate(items, 1):
        print(f"{idx}. {item_name}")
    print("=" * 50)
    
    while True:
        try:
            choice = input(f"\nEnter number to process (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                sys.exit(0)
            
            choice_idx = int(choice)
            
            if allow_all and choice_idx == 0:
                print(f"\n✓ Selected: All {item_type}s\n")
                return None
            
            choice_idx -= 1
            if 0 <= choice_idx < len(items):
                selected = items[choice_idx]
                print(f"\n✓ Selected: {selected}\n")
                return selected
            else:
                print(f"❌ Please enter a number between {'0' if allow_all else '1'} and {len(items)}")
        except ValueError:
            print("❌ Invalid input.")

# --- 6. LLM Interaction ---

def call_llm_evaluation(dimension: str, input_b64: str, pred_b64: str, 
                        prompt: str, editing_method: str = None, 
                        knowledge_prompt: str = None, retry_count: int = 3):
    """Generic function to call the LLM for a specific evaluation dimension."""
    
    if dimension not in PROMPTS:
        logger.error(f"Prompt for dimension {dimension} not defined.")
        return {dimension: 0, "rationale": "Prompt missing"}

    system_prompt = PROMPTS[dimension]

    for attempt in range(retry_count):
        try:
            user_content = [
                {"type": "text", "text": f"User editing instruction (prompt): {prompt}"}
            ]
            if editing_method:
                user_content.append({"type": "text", "text": f"Editing method: {editing_method}"})
            if knowledge_prompt and dimension == "SE":
                user_content.append({"type": "text", "text": f"Knowledge prompt: {knowledge_prompt}"})
            
            # NOTE: Evaluation only uses Input and Prediction images (No Ground Truth/Output)
            user_content.extend([
                {"type": "text", "text": "Input image (original):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_b64}"}},
                {"type": "text", "text": "Pred image (model output):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pred_b64}"}},
                {"type": "text", "text": "\n**CRITICAL REQUIREMENT: Your entire response must be ONLY a valid JSON object in this exact format:**\n" + 
                                       f'{{\"{dimension}\": <score 0-5>, \"rationale\": \"<explanation>\"}}\n' +
                                       "**NO markdown, NO code blocks, NO explanations outside the JSON, NO think tags. Just the raw JSON object.**"}
            ])
            
            response = CLIENT.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean JSON string (remove markdown, think tags)
            result_text = result_text.replace('<think>', '').replace('</think>', '')
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            # Extract JSON object using braces
            start = result_text.find('{')
            end = result_text.rfind('}') # find last '}'
            
            # Handle nested braces by counting balance if needed, or simple slice if formatted well
            if start != -1:
                brace_count = 0
                json_end = -1
                for i in range(start, len(result_text)):
                    if result_text[i] == '{': brace_count += 1
                    elif result_text[i] == '}': 
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                if json_end != -1:
                    result_text = result_text[start:json_end]

            result_json = json.loads(result_text)
            
            if dimension in result_json and "rationale" in result_json:
                return result_json
            else:
                logger.warning(f"JSON missing keys, retrying ({attempt + 1}/{retry_count})")
        
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {dimension}: {e}")
            if attempt == retry_count - 1:
                return {dimension: 0, "rationale": f"API Error: {str(e)}"}

# --- 7. Task Processing ---

def evaluate_task(task: dict, sample_dir: Path, pred_images_dir: Path):
    """Evaluates a single task across all dimensions."""
    task_id = task.get("id")
    prompt = task.get("prompt")
    editing_method = task.get("editing_method")
    knowledge_prompt = task.get("knowledge_prompt")
    
    # Get filename from JSON
    input_fname = task.get("input_image")
    
    # Robustness check
    if not input_fname:
        raise ValueError(f"Task {task_id} missing 'input_image' in JSON")
        
    input_path = sample_dir / input_fname
    
    # Prediction Image Logic: Based on folder name (e.g. 001 -> 001_edited.jpg)
    sample_number = sample_dir.name
    pred_path = pred_images_dir / f"{sample_number}_edited.jpg"

    # Check existence
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Pred image not found: {pred_path}")

    # Encode
    input_b64 = encode_image_to_base64(input_path)
    pred_b64 = encode_image_to_base64(pred_path)

    if not all([input_b64, pred_b64]):
        raise ValueError("Failed to encode images.")

    # Execute Evaluations
    results = {}
    
    # Standard dimensions
    logger.debug(f"[{task_id}] Evaluating IF...")
    results["IF"] = call_llm_evaluation("IF", input_b64, pred_b64, prompt, editing_method)
    
    logger.debug(f"[{task_id}] Evaluating TA...")
    results["TA"] = call_llm_evaluation("TA", input_b64, pred_b64, prompt)
    
    logger.debug(f"[{task_id}] Evaluating VC...")
    results["VC"] = call_llm_evaluation("VC", input_b64, pred_b64, prompt)
    
    logger.debug(f"[{task_id}] Evaluating LP...")
    results["LP"] = call_llm_evaluation("LP", input_b64, pred_b64, prompt)

    # Optional dimension
    if knowledge_prompt:
        logger.debug(f"[{task_id}] Evaluating SE...")
        results["SE"] = call_llm_evaluation("SE", input_b64, pred_b64, prompt, knowledge_prompt=knowledge_prompt)

    return results

def process_single_task_wrapper(task: dict, sample_dir: Path, pred_images_dir: Path):
    """Wrapper for thread pool execution to handle exceptions safely."""
    task_id = task.get("id")
    try:
        eval_results = evaluate_task(task, sample_dir, pred_images_dir)
        return {
            "status": "success",
            "data": {
                "task_id": task_id,
                "evaluation_results": eval_results
            }
        }
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        return {
            "status": "error",
            "task_id": task_id,
            "error": str(e)
        }

def collect_tasks(input_dir: Path, category: str, pred_cat_dir: Path, already_scored: set):
    """Scans directories to find JSON tasks that haven't been scored yet."""
    tasks_to_run = []
    skipped_count = 0
    
    cat_dir = input_dir / category
    if not cat_dir.exists():
        return [], 0

    # Iterate through sample folders (e.g., 001, 002)
    for sample_dir in sorted(cat_dir.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name.startswith('.'):
            continue

        # Flexible JSON finding logic
        sample_number = sample_dir.name
        stripped_number = convert_number_to_filename(sample_number)
        suffixes = [
            f"_{sample_number}.json", f".{sample_number}.json",
            f"-{sample_number}.json", f"{sample_number}.json",
        ]
        if stripped_number != sample_number:
            suffixes.extend([
                f"_{stripped_number}.json", f".{stripped_number}.json",
                f"-{stripped_number}.json", f"{stripped_number}.json",
            ])
            
        json_path = None
        for f in sample_dir.iterdir():
            if not f.is_file(): continue
            for suffix in suffixes:
                if f.name.endswith(suffix):
                    json_path = f
                    break
            if json_path: break
        
        if not json_path:
            continue
        
        try:
            with open(json_path, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
                # Simple fix for trailing commas
                if content.endswith(",}"): content = content[:-2] + "}"
                if content.endswith(",]"): content = content[:-2] + "]"
                data = json.loads(content)
            
            task_list = data.get("tasks", [data]) if isinstance(data, dict) else data
            if not isinstance(task_list, list): task_list = [task_list]
            
            for task in task_list:
                if not isinstance(task, dict): continue
                t_id = task.get("id")
                if not t_id: continue
                
                # Resume logic: Check if already scored
                if (category, t_id) in already_scored:
                    skipped_count += 1
                    continue
                
                tasks_to_run.append((task, sample_dir))
                
        except Exception as e:
            logger.error(f"Error reading JSON in {sample_dir}: {e}")

    return tasks_to_run, skipped_count

# --- 8. Main Execution ---

# --- 8. Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="AI Image Editing Evaluation Tool")
    parser.add_argument("--input_dir", type=Path, required=True, help="Path to ground truth/input data")
    parser.add_argument("--pred_dir", type=Path, required=True, help="Path to model predictions")
    parser.add_argument("--output_dir", type=Path, default=Path("./results"), help="Directory to save evaluation results")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent threads")
    parser.add_argument("--no_interactive", action="store_true", help="Skip interactive selection and process all")
    
    args = parser.parse_args()

    # Paths
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = args.output_dir / "failed"
    failed_dir.mkdir(exist_ok=True)

    # 1. Select Models
    model_dirs = [d.name for d in args.pred_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    selected_model = select_from_list(model_dirs, "Model", allow_all=True, auto_select_all=args.no_interactive)
    models_to_process = [selected_model] if selected_model else model_dirs

    # 2. Select Categories
    cat_dirs = [d.name for d in args.input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    selected_category = select_from_list(cat_dirs, "Category", allow_all=True, auto_select_all=args.no_interactive)
    cats_to_process = [selected_category] if selected_category else cat_dirs

    # 【新增 1】初始化全局统计字典
    total_stats = {'new': 0, 'success': 0, 'failed': 0, 'skipped': 0}

    # 3. Processing Loop
    for model_name in models_to_process:
        logger.info(f"\n{'='*60}\nProcessing Model: {model_name}\n{'='*60}")
        
        # Load existing results
        res_file = args.output_dir / f"{model_name}.json"
        fail_file = failed_dir / f"{model_name}_failed.json"
        
        existing_results = []
        already_scored = set()
        
        if res_file.exists():
            try:
                with open(res_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    for item in existing_results:
                        already_scored.add((item.get('category'), item.get('task_id')))
            except Exception:
                logger.warning("Could not read existing results, starting fresh.")

        existing_failures = []
        if fail_file.exists():
             try:
                with open(fail_file, 'r', encoding='utf-8') as f: existing_failures = json.load(f)
             except: pass

        # Collect Tasks
        all_tasks = []
        total_skipped = 0
        
        pred_base = args.pred_dir / model_name
        
        for cat in cats_to_process:
            pred_cat_dir = pred_base / cat
            if not pred_cat_dir.exists():
                logger.warning(f"Prediction directory missing for {model_name}/{cat}")
                continue
                
            tasks, skipped = collect_tasks(args.input_dir, cat, pred_cat_dir, already_scored)
            total_skipped += skipped
            
            for t, s_dir in tasks:
                all_tasks.append((t, s_dir, pred_cat_dir, cat))

        if not all_tasks:
            logger.info(f"No new tasks for {model_name} (Skipped {total_skipped}).")
            # 即使跳过也要更新统计
            total_stats['skipped'] += total_skipped
            continue

        logger.info(f"Starting evaluation for {len(all_tasks)} tasks with {args.workers} workers.")

        # Thread Pool Execution
        new_results = []
        new_failures = []
        
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="Evaluator") as executor:
            futures = {}
            for task, s_dir, p_dir, cat in all_tasks:
                fut = executor.submit(process_single_task_wrapper, task, s_dir, p_dir)
                futures[fut] = (task, cat)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Eval {model_name}"):
                task_ctx, cat = futures[future]
                try:
                    res = future.result()
                    if res['status'] == 'success':
                        data = res['data']
                        data['category'] = cat
                        data['model'] = model_name
                        new_results.append(data)
                    else:
                        res['category'] = cat
                        res['model'] = model_name
                        new_failures.append(res)
                except Exception as e:
                    logger.error(f"Critical error in future: {e}")

        # Save Results
        final_results = existing_results + new_results
        final_failures = existing_failures + new_failures
        
        # Backup
        if res_file.exists():
            shutil.copy(res_file, res_file.with_suffix(f".bak_{datetime.now().strftime('%H%M%S')}"))

        with open(res_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
            
        with open(fail_file, 'w', encoding='utf-8') as f:
            json.dump(final_failures, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved {len(new_results)} new results for {model_name}.")

        # 【新增 2】更新全局统计数据
        n_new = len(new_results)
        n_fail = len(new_failures)
        
        total_stats['new'] += (n_new + n_fail)
        total_stats['success'] += n_new
        total_stats['failed'] += n_fail
        total_stats['skipped'] += total_skipped

    # 【新增 3】打印最终统计摘要
    logger.info("\n" + "=" * 60)
    logger.info(" 🎉 Batch Processing Complete ")
    logger.info("=" * 60)
    logger.info(f"Models Processed: {len(models_to_process)}")
    logger.info(f"Total New Tasks:  {total_stats['new']}")
    logger.info(f"  └─ Success:     {total_stats['success']}")
    logger.info(f"  └─ Failed:      {total_stats['failed']}")
    logger.info(f"Total Skipped:    {total_stats['skipped']}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
