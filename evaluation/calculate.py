import json

# 您的JSON文件名
file_name = "gpt_evaluation_2\gemini-3-pro-image-preview.json" 
# (在我的测试中，我使用了包含3个对象的示例数据)

# 初始化变量
total_objects = 0

# 始终存在的标准指标
standard_indicators = ['IF', 'TA', 'VC', 'LP']
sums = {indicator: 0 for indicator in standard_indicators}

# 针对可选 'SE' 指标的特殊处理
sum_se = 0
count_se = 0 # 需要计算有多少对象 *包含* SE

data = []

# 读取并加载JSON文件
try:
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 检查数据是否为列表
    if not isinstance(data, list):
        print("Error: JSON content is not a list of objects.")
    else:
        # 获取总对象数
        total_objects = len(data)

        # 遍历列表中的每个对象
        for item in data:
            if 'evaluation_results' in item:
                results = item['evaluation_results']
                
                # 1. 计算标准指标的总和
                for key in standard_indicators:
                    if key in results and key in results[key]:
                        sums[key] += results[key][key]
                
                # 2. 计算可选 'SE' 指标的总和与数量
                if 'SE' in results and 'SE' in results['SE']:
                    sum_se += results['SE']['SE']
                    count_se += 1 # 仅在SE存在时才计数

        # 计算平均值
        averages = {}
        average_se = 0.0

        if total_objects > 0:
            # 标准平均值 = 总和 / 总对象数
            for key in standard_indicators:
                averages[key] = sums[key] / total_objects
            
            # SE 平均值 = SE总和 / 包含SE的对象数
            if count_se > 0:
                average_se = sum_se / count_se
        else:
            print("No objects found in the JSON file.")

        # 打印结果
        print(f"Total number of objects: {total_objects}")
        
        print("\nAverage for standard indicators (divided by total):")
        if total_objects > 0:
            for key, avg in averages.items():
                print(f"  {key}: {avg:.2f}") # 格式化为两位小数

        print("\nAverage for optional indicator 'SE' (divided by matching count):")
        # 同样格式化输出
        print(f"  SE: {average_se:.2f} (calculated from {count_se} matching objects)")

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from '{file_name}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")