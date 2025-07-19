import os
import shutil

# --- 配置 ---
# 源文件夹：存放着所有子文件夹和图片的地方
source_folder = "E:\\data"

# 目标文件夹：所有图片将被提取到这里
destination_folder = 'extracted_images'

# 定义要查找的图片文件扩展名（不区分大小写）
image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

# --- 主程序 ---

def extract_images():
    """
    遍历源文件夹及其子目录，将所有图片复制到目标文件夹。
    """
    # 1. 检查源文件夹是否存在
    if not os.path.isdir(source_folder):
        print(f"错误：源文件夹 '{source_folder}' 不存在。请检查路径是否正确。")
        return

    # 2. 创建目标文件夹（如果不存在）
    os.makedirs(destination_folder, exist_ok=True)
    print(f"图片将被提取到: '{destination_folder}'")
    
    copied_count = 0

    # 3. 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            # 检查文件扩展名是否为图片格式
            if filename.lower().endswith(image_extensions):
                
                source_path = os.path.join(root, filename)
                dest_path = os.path.join(destination_folder, filename)
                
                # 4. 处理文件名冲突
                # 如果目标文件夹中已存在同名文件，则在文件名后添加序号
                counter = 1
                base_name, ext = os.path.splitext(filename)
                while os.path.exists(dest_path):
                    new_filename = f"{base_name}_{counter}{ext}"
                    dest_path = os.path.join(destination_folder, new_filename)
                    counter += 1
                    
                # 5. 复制文件
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"已复制: {source_path}  ->  {dest_path}")
                    copied_count += 1
                except Exception as e:
                    print(f"复制文件时出错 {source_path}: {e}")

    print("\n--- 操作完成 ---")
    if copied_count == 0:
        print("在源文件夹中没有找到任何图片。")
    else:
        print(f"总共成功提取了 {copied_count} 张图片。")


if __name__ == '__main__':
    extract_images()