import pandas as pd
import os

# 要合并的 CSV 文件路径
csv_files = ["./create_csv/tcga_slides_luad_1.1.csv", "./create_csv/tcga_slides_lusc_1.2.csv"]

# --------------------- 关键修复 ---------------------
# 输出文件夹（会自动创建，不存在也没关系）
output_folder = "./create_csv"  
# 最终文件名
output_filename = "uuid_name_file_1.1.csv"

# 自动拼接完整路径 + 自动创建文件夹
full_save_path = os.path.join(output_folder, output_filename)
os.makedirs(output_folder, exist_ok=True)  
# ----------------------------------------------------

random_seed = 42

# 1. 读取并合并所有 CSV
dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)
merged_df = pd.concat(dfs, ignore_index=True)

# 2. 随机打乱行顺序
shuffled_df = merged_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# 3. 保存文件（用修复后的路径）
shuffled_df.to_csv(full_save_path, index=False, encoding="utf-8")

print(f"✅ 合并完成！共 {len(shuffled_df)} 行，已打乱并保存到: {full_save_path}")
print(f"📊 列名: {list(shuffled_df.columns)}")