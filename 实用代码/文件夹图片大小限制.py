import os
import sys

# 尝试导入 Pillow 库，用于处理图片
# 如果用户没有安装这个库，脚本会报错，所以这里做一个友好的提示
try:
    from PIL import Image
except ImportError:
    print("错误：没有找到 'Pillow' 库。")
    print("请打开命令行(终端)，输入以下命令来安装：")
    print("pip install Pillow")
    input("按回车键退出...")
    sys.exit(1)

def resize_images_in_folder(folder_path, max_dimension=1024):
    """
    遍历文件夹，将图片的长或宽限制在 max_dimension 以内。
    保持长宽比缩放。
    """
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在！")
        return

    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # 统计处理数量
    count_processed = 0
    count_skipped = 0
    count_error = 0

    print(f"正在扫描文件夹: {folder_path} ...")
    
    # os.walk 可以遍历子文件夹。如果只需要当前文件夹，可以用 os.listdir
    # 这里我们假设只需要处理当前文件夹下的图片，不递归处理子文件夹
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, filename)
        
        # 判断是不是文件（排除文件夹）
        if not os.path.isfile(file_path):
            continue
            
        # 判断是不是图片（根据后缀名）
        if not filename.lower().endswith(valid_extensions):
            continue
            
        try:
            # 打开图片
            with Image.open(file_path) as img:
                width, height = img.size
                
                # 如果图片的长或宽超过了限制
                if width > max_dimension or height > max_dimension:
                    print(f"正在处理: {filename} (原尺寸: {width}x{height}) ...")
                    
                    # 计算缩放比例
                    # 比如原图 2000x1000, 限制 1024
                    # 缩放比例 = 1024 / 2000 = 0.512
                    # 新高度 = 1000 * 0.512 = 512
                    
                    # 找出长边，计算缩放比例
                    if width > height:
                        ratio = max_dimension / width
                        new_width = max_dimension
                        new_height = int(height * ratio)
                    else:
                        ratio = max_dimension / height
                        new_height = max_dimension
                        new_width = int(width * ratio)
                        
                    # 执行缩放 (Image.LANCZOS 是高质量缩放算法)
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # 保存图片，覆盖原文件
                    # 注意：如果不需要覆盖，可以修改这里的保存路径
                    resized_img.save(file_path)
                    
                    print(f"  -> 已缩放为: {new_width}x{new_height}")
                    count_processed += 1
                else:
                    # 图片尺寸在范围内，不需要处理
                    # print(f"跳过: {filename} (尺寸 {width}x{height} 符合要求)")
                    count_skipped += 1
                    
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            count_error += 1

    print("-" * 30)
    print("处理完成！")
    print(f"共缩放图片: {count_processed} 张")
    print(f"未需处理图片: {count_skipped} 张")
    if count_error > 0:
        print(f"处理出错文件: {count_error} 个")

if __name__ == "__main__":
    print("=== 图片批量尺寸限制工具 ===")
    print("功能：将文件夹内的图片长宽限制在 1024 像素以内 (保持比例)")
    print("-" * 30)
    
    # 获取用户输入的文件夹路径
    # strip('"') 是为了处理用户直接把文件夹拖入终端时可能带有的双引号
    target_folder = input("请输入图片所在的文件夹路径 (可直接拖入文件夹): ").strip('"').strip("'").strip()
    
    if target_folder:
        resize_images_in_folder(target_folder, max_dimension=1024)
    else:
        print("未输入路径，程序退出。")
    
    input("按回车键退出...")
