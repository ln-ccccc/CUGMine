import os
import cv2
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from mmseg.apis import init_model, inference_model
import torch

# =========================
# 1) Vaihingen 类别与配色
# =========================
# classes=('impervious_surface','building','low_vegetation','tree','car')
# palette=[[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0]]

palette = [
    ['impervious_surface', [255, 255, 255]],
    ['building',           [0,   0,   255]],
    ['low_vegetation',     [0,   255, 255]],
    ['tree',               [0,   255, 0]],
    ['car',                [255, 255, 0]],
]

palette_dict = {idx: each[1] for idx, each in enumerate(palette)}
NUM_CLASSES = len(palette_dict)

# =========================
# 2) GDAL 图像加载
# =========================
def load_rs_image_with_gdal(img_path: str, to_float32: bool = True):
    if gdal is None:
        raise RuntimeError("GDAL未安装，无法加载遥感图像和投影信息")

    ds = gdal.Open(img_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"警告：无法打开图像 {img_path}")
        return (None, None, None)

    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()

    # GDAL: [C, H, W] -> [H, W, C]
    arr = ds.ReadAsArray()
    if arr.ndim == 2:
        # 单通道 -> [H,W,1]
        img_array = arr[:, :, None]
    else:
        img_array = np.einsum('ijk->jki', arr)

    if to_float32:
        img_array = img_array.astype(np.float32)

    ds = None
    return (img_array, proj, geotrans)

# =========================
# 3) GDAL 保存 TIFF（带投影）
# =========================
def save_tiff_with_projection(save_path: str, pred_mask: np.ndarray, proj: str, geotrans: tuple):
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
# 4) 滑窗推理
# =========================
def cosine_weight(tile_size: int) -> np.ndarray:
    w = np.hanning(tile_size).astype(np.float32)
    ww = np.outer(w, w)
    return np.maximum(ww, 1e-6)

def slide_inference_large_image(model, img_array: np.ndarray, tile_size: int = 512, stride: int = 256) -> np.ndarray:
    H, W, C = img_array.shape
    print(f"开始滑窗推理，图像尺寸: {H}x{W}x{C}")

    weight_map = cosine_weight(tile_size)
    pred_sum = np.zeros((H, W, NUM_CLASSES), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    y_positions = list(range(0, max(H - tile_size + stride, 0), stride))
    x_positions = list(range(0, max(W - tile_size + stride, 0), stride))

    if H > tile_size and (len(y_positions) == 0 or y_positions[-1] < H - tile_size):
        y_positions.append(H - tile_size)
    if W > tile_size and (len(x_positions) == 0 or x_positions[-1] < W - tile_size):
        x_positions.append(W - tile_size)

    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]

    total_patches = len(y_positions) * len(x_positions)
    print(f"滑窗配置: tile_size={tile_size}, stride={stride}")
    print(f"滑窗数量: {len(y_positions)}x{len(x_positions)} = {total_patches}")

    patch_count = 0
    for y in tqdm(y_positions, desc="滑窗推理"):
        for x in x_positions:
            patch_count += 1

            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            actual_h = y_end - y
            actual_w = x_end - x

            patch = img_array[y:y_end, x:x_end].copy()

            if actual_h < tile_size or actual_w < tile_size:
                pad_h = tile_size - actual_h
                pad_w = tile_size - actual_w
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            try:
                result = inference_model(model, patch)
                pred_logits = result.pred_sem_seg.data[0].cpu().numpy()

                # 兼容两种输出
                if pred_logits.ndim == 2:
                    pred_one_hot = np.zeros((NUM_CLASSES, tile_size, tile_size), dtype=np.float32)
                    for c in range(NUM_CLASSES):
                        pred_one_hot[c] = (pred_logits == c).astype(np.float32)
                    pred_logits = pred_one_hot
                elif pred_logits.ndim == 3 and pred_logits.shape[0] == NUM_CLASSES:
                    pred_logits = torch.softmax(torch.from_numpy(pred_logits), dim=0).numpy()
                else:
                    raise ValueError(f"意外的模型输出形状: {pred_logits.shape}")

                pred_logits = pred_logits[:, :actual_h, :actual_w]
                pred_logits = np.transpose(pred_logits, (1, 2, 0))  # [H,W,C]

                current_weight = weight_map[:actual_h, :actual_w]
                pred_sum[y:y_end, x:x_end] += pred_logits * current_weight[..., None]
                weight_sum[y:y_end, x:x_end] += current_weight

            except Exception as e:
                print(f"推理错误 patch {patch_count}/{total_patches} at ({y}, {x}): {str(e)}")
                continue

    print("融合所有patch结果...")
    weight_sum = np.maximum(weight_sum, 1e-6)
    final_pred = pred_sum / weight_sum[..., None]
    pred_mask = np.argmax(final_pred, axis=2).astype(np.uint8)

    print(f"推理完成，输出掩码形状: {pred_mask.shape}")
    return pred_mask

# =========================
# 5) 可视化
# =========================
def create_color_mask(pred_mask: np.ndarray) -> np.ndarray:
    pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for idx, color in palette_dict.items():
        pred_mask_bgr[pred_mask == idx] = color
    return pred_mask_bgr

def create_overlay_image(original_img: np.ndarray, pred_mask_bgr: np.ndarray, opacity: float = 0.3) -> np.ndarray:
    # 原图 float32 -> uint8（简单线性拉伸）
    if original_img.dtype == np.float32:
        img_min, img_max = float(original_img.min()), float(original_img.max())
        if img_max > img_min:
            original_img_u8 = ((original_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            original_img_u8 = np.zeros_like(original_img, dtype=np.uint8)
    else:
        original_img_u8 = original_img.astype(np.uint8)

    if original_img_u8.ndim == 2:
        original_img_u8 = cv2.cvtColor(original_img_u8, cv2.COLOR_GRAY2BGR)
    elif original_img_u8.shape[2] == 1:
        original_img_u8 = np.repeat(original_img_u8, 3, axis=2)

    # 如果不是3通道（例如4通道），取前3通道做 overlay
    if original_img_u8.shape[2] > 3:
        original_img_u8 = original_img_u8[:, :, :3]

    overlay = cv2.addWeighted(original_img_u8, 1 - opacity, pred_mask_bgr, opacity, 0)
    return overlay

# =========================
# 6) 单图处理
# =========================
def process_single_large_img(img_path: str, model, output_dir: str, tile_size: int = 512, stride: int = 256,
                             save_color: bool = True, save_overlay: bool = True):
    print(f"\n开始处理: {img_path}")

    img_array, proj, geotrans = load_rs_image_with_gdal(img_path, to_float32=True)
    if img_array is None:
        print(f"跳过文件（加载失败）: {img_path}")
        return

    print(f"图像信息: 尺寸={img_array.shape}, 数据类型={img_array.dtype}")

    pred_mask = slide_inference_large_image(model, img_array, tile_size=tile_size, stride=stride)

    os.makedirs(output_dir, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(img_path))[0]

    # 1) pred tif
    tiff_path = os.path.join(output_dir, f"pred-{img_basename}.tif")
    save_tiff_with_projection(tiff_path, pred_mask, proj, geotrans)

    # 2) color png
    if save_color:
        color = create_color_mask(pred_mask)
        color_path = os.path.join(output_dir, f"pred-{img_basename}_color.png")
        cv2.imwrite(color_path, color)
        print(f"保存彩色掩码: {color_path}")

    # 3) overlay png
    if save_overlay:
        color = create_color_mask(pred_mask)
        overlay = create_overlay_image(img_array, color, opacity=0.3)
        overlay_path = os.path.join(output_dir, f"pred-{img_basename}_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"保存叠加图像: {overlay_path}")

    print(f"处理完成: {img_path}")

# =========================
# 7) 批量处理
# =========================
def batch_process_large_images(model, input_dir: str, output_dir: str, tile_size: int = 512, stride: int = 256,
                              save_color: bool = True, save_overlay: bool = True):
    valid_ext = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_ext]

    if not image_files:
        print(f"在 {input_dir} 中未找到支持的图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件")
    print(f"输出目录: {output_dir}")
    print(f"推理参数: tile_size={tile_size}, stride={stride}")

    for i, fn in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理文件: {fn}")
        process_single_large_img(
            img_path=os.path.join(input_dir, fn),
            model=model,
            output_dir=output_dir,
            tile_size=tile_size,
            stride=stride,
            save_color=save_color,
            save_overlay=save_overlay
        )

    print("批量处理完成!")

# =========================
# 8) 主函数（写死路径）
# =========================
if __name__ == "__main__":
    # 你自己把这两个换成 Vaihingen 对应的 config / checkpoint（写死）

    config_file="/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/work_dirs/dinov3_swin_Vaihingen_V6/dinov3_swin_Vaihingen_V6.py"
    checkpoint_file="/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/work_dirs/dinov3_swin_Vaihingen_V6/best_mIoU_epoch_41.pth"
    device = "cuda:0"
    print("正在加载模型...")
    model = init_model(config_file, checkpoint_file, device=device)
    print("模型加载完成!")

    # Vaihingen val 路径（写死）
    input_dir = "/home/featurize/data/vaihingen/img_dir/val"
    output_dir="/home/featurize/work/mmsegmentation_25717_mine/mmsegmentation/outputs/vaihingen_large_inference"

    batch_process_large_images(
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=512,
        stride=256,
        save_color=True,
        save_overlay=True
    )

    print("所有任务完成!")