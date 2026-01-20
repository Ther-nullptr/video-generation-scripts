import os
import shutil

# ================= 配置区域 =================
# 输入目录：就是截图里包含那一堆文件夹的父目录
# 如果脚本就放在该目录下，可以用 "."
source_dir = r"/home/wyj24/project/video_generation/output/Wan2.1-T2V-1.3B-Diffusers_steps25_fps16_samples5_res832x480/overall_consistency" 

# 输出目录：你想把提取出来的视频存放在哪里
output_dir = r"/home/wyj24/project/video_generation/output/Wan2.1-T2V-1.3B-Diffusers_steps25_fps16_samples5_res832x480/overall_consistency_2"
# ===========================================

def extract_and_rename():
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 遍历源目录下的所有项目
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # 1. 确保它是一个文件夹 (对应截图中的 ...house-0.mp4 文件夹)
        if os.path.isdir(folder_path):
            
            # 2. 查找文件夹里面的文件
            files_in_subfolder = os.listdir(folder_path)
            
            # 过滤出视频文件 (这里假设是 .mp4，如果有其他格式可以增加)
            video_files = [f for f in files_in_subfolder if f.lower().endswith('.mp4')]

            if video_files:
                # 按照你的描述，里面只有一个视频，我们取第一个
                original_file_name = video_files[0]
                original_file_path = os.path.join(folder_path, original_file_name)

                # 3. 确定新文件名
                # 你的文件夹名本身就带了 .mp4 (例如 "...house-0.mp4")
                # 所以直接用文件夹名作为新文件名即可
                new_file_name = folder_name
                
                # 如果文件夹名没有后缀，可以用下面这行自动补全：
                # if not new_file_name.lower().endswith('.mp4'):
                #     new_file_name += ".mp4"

                target_path = os.path.join(output_dir, new_file_name)

                # 4. 执行复制 (建议先用 copy 而不是 move，防止出错丢失文件)
                try:
                    shutil.copy2(original_file_path, target_path)
                    print(f"[成功] 提取: {original_file_name} -> 重命名为: {new_file_name}")
                except Exception as e:
                    print(f"[错误] 处理 {folder_name} 时出错: {e}")
            else:
                print(f"[跳过] 文件夹 {folder_name} 中没有找到 mp4 文件")

if __name__ == "__main__":
    # 再次确认路径是否填写正确
    if source_dir == r"path/to/your/source_folder":
        print("请先在代码中修改 source_dir 和 output_dir 的路径！")
    else:
        extract_and_rename()
        print("\n所有任务完成。")