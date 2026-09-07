import os
import cv2
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from mmseg.apis import init_model, inference_model
import torch
import matplotlib.pyplot as plt

# =========================
# 1) 类别配置（与您的配置保持一致）
# =========================
palette = [
    ['grassland', [127,127,127]],
    ['forest', [0,0,200]],
    ['building', [0,200,0]], 
    ['road', [144,238,144]],
    ['bareground', [30,30,30]],
    ['water', [8,189,251]]
]

palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]

NUM_CLASSES = len(palette_dict)

# =========================
# 2) GDAL图像加载（基于您的代码）
# =========================
def load_rs_image_with_gdal(img_path: str, to_float32: bool = True):
    """
    基于您原代码的GDAL加载函数，返回（图像数组, 投影信息, 地理变换参数）
    """
    if gdal is None:
        raise RuntimeError("GDAL未安装，无法加载遥感图像和投影信息")
    
    ds = gdal.Open(img_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"警告：无法打开图像 {img_path}")
        return (None, None, None)
    
    # 获取投影和地理变换
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    
    # 读取图像数组并调整维度（GDAL默认[C, H, W] → 转为[H, W, C]）
    img_array = np.einsum('ijk->jki', ds.ReadAsArray())
    
    if to_float32:
        img_array = img_array.astype(np.float32)
    
    ds = None
    return (img_array, proj, geotrans)

# =========================
# 3) GDAL保存TIFF（基于您的代码）
# =========================
def save_tiff_with_projection(save_path: str, pred_mask: np.ndarray, proj: str, geotrans: tuple):
    """
    基于您原代码的GDAL保存函数，保存带投影的TIFF文件
    """
    if gdal is None:
        print("GDAL未安装，无法保存带投影的TIFF")
        return False
    
    if pred_mask.ndim != 2:
        print(f"错误：预测掩码维度为{pred_mask.ndim}，需为单波段[H, W]")
        return False
    
    height, width = pred_mask.shape
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(save_path, width, height, 1, gdal.GDT_Byte)
    if out_ds is None:
        print(f"错误：无法创建TIFF文件 {save_path}")
        return False
    
    # 注入原始投影和地理变换
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)
    
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(pred_mask)
    
    out_band.FlushCache()
    out_ds.FlushCache()
    
    out_band = None
    out_ds = None
    
    print(f"成功保存带投影的TIFF：{save_path}")
    return True

# =========================
# 4) 滑窗推理核心函数（新增）
# =========================
def cosine_weight(tile_size: int) -> np.ndarray:
    """生成余弦权重窗，减少拼接缝隙"""
    w = np.hanning(tile_size).astype(np.float32)
    ww = np.outer(w, w)
    return np.maximum(ww, 1e-6)

def slide_inference_large_image(
    model, 
    img_array: np.ndarray, 
    tile_size: int = 512, 
    stride: int = 256
) -> np.ndarray:
    """
    对大图进行滑窗推理
    
    Args:
        model: 已加载的mmseg模型
        img_array: 输入图像 [H, W, C] (float32)
        tile_size: 滑窗大小，应与训练时一致
        stride: 滑窗步长，stride < tile_size 有重叠，减少边界效应
        
    Returns:
        pred_mask: 预测掩码 [H, W] (uint8)
    """
    H, W, C = img_array.shape
    print(f"开始滑窗推理，图像尺寸: {H}x{W}x{C}")
    
    # 生成权重窗（用于边界融合）
    weight_map = cosine_weight(tile_size)
    
    # 初始化累积数组
    pred_sum = np.zeros((H, W, NUM_CLASSES), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    
    # 计算滑窗位置
    y_positions = list(range(0, max(H - tile_size + stride, 0), stride))
    x_positions = list(range(0, max(W - tile_size + stride, 0), stride))
    
    # 确保覆盖到图像边界
    if H > tile_size and (len(y_positions) == 0 or y_positions[-1] < H - tile_size):
        y_positions.append(H - tile_size)
    if W > tile_size and (len(x_positions) == 0 or x_positions[-1] < W - tile_size):
        x_positions.append(W - tile_size)
        
    # 处理小图情况
    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]
    
    total_patches = len(y_positions) * len(x_positions)
    print(f"滑窗配置: tile_size={tile_size}, stride={stride}")
    print(f"滑窗数量: {len(y_positions)}x{len(x_positions)} = {total_patches}")
    
    # 滑窗推理
    patch_count = 0
    for y in tqdm(y_positions, desc="滑窗推理"):
        for x in x_positions:
            patch_count += 1
            
            # 计算实际patch边界
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            actual_h = y_end - y
            actual_w = x_end - x
            
            # 提取patch
            patch = img_array[y:y_end, x:x_end].copy()
            
            # 如果patch小于tile_size，进行padding到tile_size
            if actual_h < tile_size or actual_w < tile_size:
                pad_h = tile_size - actual_h
                pad_w = tile_size - actual_w
                # 使用reflect模式padding，保持边缘连续性
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            
            # 推理单个patch
            try:
                result = inference_model(model, patch)
                pred_logits = result.pred_sem_seg.data[0].cpu().numpy()
                
                # 处理模型输出格式
                if pred_logits.ndim == 2:  # 输出是类别标签 [H, W]
                    # 转换为one-hot编码
                    pred_one_hot = np.zeros((NUM_CLASSES, tile_size, tile_size), dtype=np.float32)
                    for c in range(NUM_CLASSES):
                        pred_one_hot[c] = (pred_logits == c).astype(np.float32)
                    pred_logits = pred_one_hot
                elif pred_logits.ndim == 3 and pred_logits.shape[0] == NUM_CLASSES:  # 输出是logits [C, H, W]
                    # 应用softmax获得概率
                    pred_logits = torch.softmax(torch.from_numpy(pred_logits), dim=0).numpy()
                else:
                    raise ValueError(f"意外的模型输出形状: {pred_logits.shape}")
                
                # 裁剪回实际patch尺寸
                pred_logits = pred_logits[:, :actual_h, :actual_w]
                pred_logits = np.transpose(pred_logits, (1, 2, 0))  # [H, W, C]
                
                # 计算当前patch的权重
                current_weight = weight_map[:actual_h, :actual_w]
                
                # 累积到全局结果
                pred_sum[y:y_end, x:x_end] += pred_logits * current_weight[..., np.newaxis]
                weight_sum[y:y_end, x:x_end] += current_weight
                
            except Exception as e:
                print(f"推理错误 patch {patch_count}/{total_patches} at ({y}, {x}): {str(e)}")
                continue
    
    # 加权平均并取最大值索引
    print("融合所有patch结果...")
    weight_sum = np.maximum(weight_sum, 1e-6)  # 避免除零
    final_pred = pred_sum / weight_sum[..., np.newaxis]
    pred_mask = np.argmax(final_pred, axis=2).astype(np.uint8)
    
    print(f"推理完成，输出掩码形状: {pred_mask.shape}")
    return pred_mask

# =========================
# 5) 可视化函数（基于您的配色方案）
# =========================
def create_color_mask(pred_mask: np.ndarray) -> np.ndarray:
    """
    根据您的配色方案生成彩色掩码
    """
    pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for idx in palette_dict.keys():
        pred_mask_bgr[np.where(pred_mask == idx)] = palette_dict[idx]
    return pred_mask_bgr

def create_overlay_image(original_img: np.ndarray, pred_mask_bgr: np.ndarray, opacity: float = 0.3) -> np.ndarray:
    """
    创建叠加图像
    """
    # 如果原图是float32，转换为uint8
    if original_img.dtype == np.float32:
        if original_img.max() <= 1.0:
            original_img_u8 = (original_img * 255).astype(np.uint8)
        else:
            # 简单拉伸到0-255
            img_min, img_max = original_img.min(), original_img.max()
            if img_max > img_min:
                original_img_u8 = ((original_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                original_img_u8 = np.zeros_like(original_img, dtype=np.uint8)
    else:
        original_img_u8 = original_img.astype(np.uint8)
    
    # 如果是单通道，转换为三通道
    if original_img_u8.ndim == 2:
        original_img_u8 = cv2.cvtColor(original_img_u8, cv2.COLOR_GRAY2BGR)
    elif original_img_u8.shape[2] == 1:
        original_img_u8 = np.repeat(original_img_u8, 3, axis=2)
    
    # 叠加
    overlay = cv2.addWeighted(original_img_u8, 1-opacity, pred_mask_bgr, opacity, 0)
    return overlay

# =========================
# 6) 单图处理函数（扩展您的原函数）
# =========================
def process_single_large_img(
    img_path: str, 
    model, 
    output_dir: str, 
    tile_size: int = 512, 
    stride: int = 256,
    save_color: bool = True,
    save_overlay: bool = True
):
    """
    处理单张大图，支持滑窗推理
    基于您原有的process_single_img函数扩展
    """
    print(f"\n开始处理: {img_path}")
    
    # 1. 加载图像（使用您的GDAL加载函数）
    img_array, original_proj, original_geotrans = load_rs_image_with_gdal(img_path, to_float32=True)
    if img_array is None:
        print(f"跳过文件（加载失败）: {img_path}")
        return
    
    print(f"图像信息: 尺寸={img_array.shape}, 数据类型={img_array.dtype}")
    
    # 2. 滑窗推理（新增的核心功能）
    try:
        pred_mask = slide_inference_large_image(
            model=model,
            img_array=img_array,
            tile_size=tile_size,
            stride=stride
        )
    except Exception as e:
        print(f"推理失败 {img_path}: {str(e)}")
        return
    
    # 3. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    
    # 3.1 保存预测掩码（TIFF格式，带投影）- 您原有的功能
    tiff_save_name = f"pred-{img_basename}.tif"
    tiff_save_path = os.path.join(output_dir, tiff_save_name)
    save_tiff_with_projection(
        save_path=tiff_save_path,
        pred_mask=pred_mask,
        proj=original_proj,
        geotrans=original_geotrans
    )
    
    # 3.2 保存彩色可视化（基于您的配色方案）
    if save_color:
        pred_mask_bgr = create_color_mask(pred_mask)
        color_save_name = f"pred-{img_basename}_color.png"
        color_save_path = os.path.join(output_dir, color_save_name)
        cv2.imwrite(color_save_path, pred_mask_bgr)
        print(f"保存彩色掩码: {color_save_path}")
    
    # 3.3 保存叠加图像（基于您原有的叠加逻辑）
    if save_overlay:
        try:
            pred_mask_bgr = create_color_mask(pred_mask)
            overlay_img = create_overlay_image(img_array, pred_mask_bgr, opacity=0.3)
            overlay_save_name = f"pred-{img_basename}_overlay.png"
            overlay_save_path = os.path.join(output_dir, overlay_save_name)
            cv2.imwrite(overlay_save_path, overlay_img)
            print(f"保存叠加图像: {overlay_save_path}")
        except Exception as e:
            print(f"叠加图像保存失败: {str(e)}")
    
    print(f"处理完成: {img_path}")

# =========================
# 7) 批量处理函数
# =========================
def batch_process_large_images(
    model,
    input_dir: str,
    output_dir: str,
    tile_size: int = 512,
    stride: int = 256,
    save_color: bool = True,
    save_overlay: bool = True
):
    """
    批量处理大图
    """
    # 支持的图像格式
    valid_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    
    # 获取所有图像文件
    image_files = []
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(filename)
    
    if len(image_files) == 0:
        print(f"在 {input_dir} 中未找到支持的图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"输出目录: {output_dir}")
    print(f"推理参数: tile_size={tile_size}, stride={stride}")
    
    success_count = 0
    fail_count = 0
    
    # 处理每个文件
    for i, img_filename in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理文件: {img_filename}")
        
        full_img_path = os.path.join(input_dir, img_filename)
        
        try:
            process_single_large_img(
                img_path=full_img_path,
                model=model,
                output_dir=output_dir,
                tile_size=tile_size,
                stride=stride,
                save_color=save_color,
                save_overlay=save_overlay
            )
            success_count += 1
        except Exception as e:
            print(f"处理失败: {img_filename}, 错误: {str(e)}")
            fail_count += 1
    
    print(f"\n批量处理完成:")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")

# =========================
# 8) 主函数和使用示例
# =========================
if __name__ == "__main__":
    # 模型配置（您需要根据实际情况修改这些路径）
    config_file = '/home/featurize/data/Hu-dv3up-822/dinov3Uper.py'
    checkpoint_file = '/home/featurize/data/Hu-dv3up-822/iter_85000.pth'
   
    device = 'cuda:0'  # 或 'cpu'
    
    # 加载模型
    print("正在加载模型...")
    model = init_model(config_file, checkpoint_file, device=device)
    print("模型加载完成!")
    
    # 设置路径
    input_dir = "/home/featurize/data/yunnan_dataset_real/img_dir/test"
    output_dir = "/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/outputs/dinouper"
    
    batch_process_large_images(
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=512,        # 滑窗大小，与训练时一致
        stride=256,           # 滑窗步长，stride < tile_size 有重叠
        save_color=True,      # 保存彩色掩码
        save_overlay=True     # 保存叠加图像
    )
    
    print("所有任务完成!")

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True)
#     parser.add_argument("--checkpoint", required=True)
#     parser.add_argument("--input-dir", required=True)
#     parser.add_argument("--output-dir", required=True)
#     parser.add_argument("--tile-size", type=int, default=512)
#     parser.add_argument("--stride", type=int, default=256)
#     parser.add_argument("--device", default="cuda:0")
#     parser.add_argument("--save-color", action="store_true", default=True)
#     parser.add_argument("--save-overlay", action="store_true", default=True)
#     args = parser.parse_args()

#     print("正在加载模型...")
#     model = init_model(args.config, args.checkpoint, device=args.device)
#     print("模型加载完成!")

#     batch_process_large_images(
#         model=model,
#         input_dir=args.input_dir,
#         output_dir=args.output_dir,
#         tile_size=args.tile_size,
#         stride=args.stride,
#         save_color=args.save_color,
#         save_overlay=args.save_overlay
#     )

#     print("所有任务完成!")