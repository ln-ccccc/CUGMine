# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
from glob import glob

from mmseg.apis import RSImage, RSInferencer


def get_image_files(input_dir):
    """获取目录中所有支持的图像文件"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(input_dir, ext)))
    return sorted(image_files)


def main():
    parser = ArgumentParser()
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--output-dir',
        help='Directory to save result images',
        default='results')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='maximum number of windows inferred simultaneously')
    parser.add_argument(
        '--window-size',
        help='window xsize,ysize',
        default=(224, 224),
        type=int,
        nargs=2)
    parser.add_argument(
        '--stride',
        help='window xstride,ystride',
        default=(224, 224),
        type=int,
        nargs=2)
    parser.add_argument(
        '--thread', default=1, type=int, help='number of inference threads')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = get_image_files(args.input_dir)
    if not image_files:
        print(f"Error: No image files found in directory {args.input_dir}")
        return

    print(f"Found {len(image_files)} images for inference")

    # 初始化推理器
    inferencer = RSInferencer.from_config_path(
        args.config,
        args.checkpoint,
        batch_size=args.batch_size,
        thread=args.thread,
        device=args.device)

    # 批量处理图像
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"Processing image {i}/{len(image_files)}: {image_path}")
            
            # 获取文件名并构建输出路径
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output_dir, f"{name}_result{ext}")
            
            # 加载图像并进行推理
            image = RSImage(image_path)
            inferencer.run(image, args.window_size, args.stride, output_path)
            
            print(f"Result saved to: {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    print("Batch inference completed!")


if __name__ == '__main__':
    main()