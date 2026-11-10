import os
import shutil

# 定义路径（根据你的实际路径修改）
root_dir = "./original_slide_folder_1"
luad_dir = os.path.join(root_dir, "lung-luad")
lusc_dir = os.path.join(root_dir, "lung-lusc")

def move_folders(src_dir, dest_dir):
    """
    将src_dir下的所有子文件夹移动到dest_dir，自动处理重名
    """
    if not os.path.isdir(src_dir):
        print(f"错误：源目录 {src_dir} 不存在")
        return
    
    for folder_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, folder_name)
        if os.path.isdir(src_path):
            dest_path = os.path.join(dest_dir, folder_name)
            # 处理重名
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_dir, f"{folder_name}_{counter}")
                counter += 1
            # 移动文件夹
            shutil.move(src_path, dest_path)
            print(f"已移动：{src_path} -> {dest_path}")

# 执行合并
print("开始合并 lung-luad...")
move_folders(luad_dir, root_dir)
print("\n开始合并 lung-lusc...")
move_folders(lusc_dir, root_dir)

# （可选）删除空的原目录
for dir_path in [luad_dir, lusc_dir]:
    if os.path.isdir(dir_path) and len(os.listdir(dir_path)) == 0:
        os.rmdir(dir_path)
        print(f"\n已删除空目录：{dir_path}")

print("\n✅ 合并完成！")