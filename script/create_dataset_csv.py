import pandas as pd
import re

# ===================== 配置区（只改这里） =====================
INPUT_CSV = "./create_csv/uuid_name_file_1.1.csv"  # 输入你刚才合并打乱的CSV
OUTPUT_CSV = "./dataset_csv/TCGA_Lung_subtyping_1.1.csv"  #  输出文件名
# ============================================================

# 1. 读取CSV
df = pd.read_csv(INPUT_CSV)

# 2. 从 slide name 提取病人ID：TCGA-xx-xxxx
def extract_patient_code(slide_name):
    match = re.match(r"(TCGA-\w+-\w+)", str(slide_name))
    return match.group(1) if match else None

df["patient_code"] = df["slide name"].apply(extract_patient_code)

# 3. 给每个唯一病人分配 case_id：patient_0, patient_1, ...
unique_patients = df["patient_code"].drop_duplicates().tolist()
patient_to_case = {p: f"patient_{i}" for i, p in enumerate(unique_patients)}
df["case_id"] = df["patient_code"].map(patient_to_case)

# 4. 构建输出列
result = df.rename(columns={"slide name": "slide_id"})[["case_id", "slide_id", "label"]]

# 5. 保存
result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print("✅ 生成成功！")
print(f"📁 输出文件：{OUTPUT_CSV}")
print(f"👤 总病人数：{len(unique_patients)}")
print(f"📊 总行数：{len(result)}")