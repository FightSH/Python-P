from PIL import Image
import os


def split_gif_into_frames(gif_path, output_folder):
    # 打开GIF文件
    gif = Image.open(gif_path)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取帧数
    frame_count = gif.n_frames

    # 逐帧处理
    for frame_number in range(frame_count):
        # 跳转到指定帧
        gif.seek(frame_number)

        # 构建输出路径
        output_path = os.path.join(output_folder, f'frame_{frame_number:04d}.png')

        # 保存当前帧
        gif.save(output_path)
        print(f'Saved {output_path}')



# 示例调用
gif_path = 'C:\\Users\Administrator\Desktop\cnn\\cnn.gif'  # 替换为你的GIF文件路径
output_folder = 'C:\\Users\Administrator\Desktop\cnn'  # 替换为你想要保存帧的文件夹路径
split_gif_into_frames(gif_path, output_folder)

