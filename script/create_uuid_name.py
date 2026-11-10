# 1. 你的TCGA数据根目录
# root_dir = r"G:\tcga-data\lung-lusc"
import os
import csv

# 1. 你的TCGA数据根目录
# root_dir = "./original_slide_folder_2"
root_dir = r"G:\tcga-data\lung-lusc"

# 2. 输出配置
output_folder = "./create_csv/"
output_csv = "tcga_slides_lusc_1.2.csv"
full_save_path = os.path.join(output_folder, output_csv)
os.makedirs(output_folder, exist_ok=True)

# 3. 标签
DEFAULT_LABEL = "LUSC"  

# 开始写入CSV
with open(full_save_path, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["uuid", "slide name", "label"])

    for uuid_folder in os.listdir(root_dir):
        uuid_path = os.path.join(root_dir, uuid_folder)
        if not os.path.isdir(uuid_path):
            continue

        for slide_file in os.listdir(uuid_path):
            slide_path = os.path.join(uuid_path, slide_file)
            if not os.path.isfile(slide_path):
                continue

            # 只处理 TCGA 开头的文件
            if slide_file.startswith("TCGA"):
                # ========== 核心：自动去掉 .svs 或 .svs.partial 后缀 ==========
                if slide_file.endswith(".svs.partial"):
                    slide_name = slide_file[:-12]  # 去掉 .svs.partial
                elif slide_file.endswith(".svs"):
                    slide_name = slide_file[:-4]   # 去掉 .svs
                else:
                    continue  # 其他文件跳过

                # 写入最终结果
                writer.writerow([uuid_folder, slide_name, DEFAULT_LABEL])
                print(f"已记录: {uuid_folder} | {slide_name} | {DEFAULT_LABEL}")

print(f"\n✅ 全部处理完成！CSV已保存到: {full_save_path}")