import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

from osgeo import gdal

from mmseg.apis import init_model, inference_model


# 每个类别的 BGR 配色
PALETTE = [
    ['grassland', [127, 127, 127]],
    ['forest',    [0, 0, 200]],
    ['building',  [0, 200, 0]],
    ['road',      [144, 238, 144]],
    ['bareground',[30, 30, 30]],
    ['water',     [8, 189, 251]]
]


def build_palette_dict(palette):
    palette_dict = {}
    for idx, each in enumerate(palette):
        palette_dict[idx] = each[1]
    return palette_dict


def get_geo_info(img_path):
    dataset = gdal.Open(img_path)
    if dataset is None:
        return None, None
    geotransform = dataset.GetGeoTransform()
    spatial_ref = dataset.GetProjection()
    return geotransform, spatial_ref


def save_pred_mask(pred_mask, save_path, geotransform, spatial_ref):
    rows, cols = pred_mask.shape
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(save_path, cols, rows, 1, gdal.GDT_Byte)
    outdata.GetRasterBand(1).WriteArray(pred_mask)

    # 有些输入不是带地理参考的图（或 gdal 读不到），则跳过写入地理信息
    if geotransform is not None:
        outdata.SetGeoTransform(geotransform)
    if spatial_ref is not None:
        outdata.SetProjection(spatial_ref)

    outdata = None


def process_single_img(model, img_path, palette_dict, opacity, out_viz_dir, out_mask_dir, save=False):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[WARN] cv2.imread failed: {img_path}")
        return

    # ✅ 关键修复：必须传路径（因为 config 里 test_pipeline 是 LoadImageFromFile）
    result = inference_model(model, img_path)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # 将预测的整数ID，映射为对应类别的颜色
    pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for idx in palette_dict.keys():
        pred_mask_bgr[pred_mask == idx] = palette_dict[idx]

    # 将语义分割预测图和原图叠加显示
    height, width = img_bgr.shape[:2]
    pred_mask_bgr = cv2.resize(pred_mask_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
    pred_viz = cv2.addWeighted(img_bgr, opacity, pred_mask_bgr, 1 - opacity, 0)

    if save:
        base = os.path.basename(img_path)

        save_path_viz = os.path.join(out_viz_dir, f"pred-{base}")
        cv2.imwrite(save_path_viz, pred_viz)

        geotransform, spatial_ref = get_geo_info(img_path)
        save_path_mask = os.path.join(out_mask_dir, f"mask-{base}")
        save_pred_mask(pred_mask, save_path_mask, geotransform, spatial_ref)


def is_image_file(name: str):
    ext = os.path.splitext(name)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="work_dirs/KuangshanDataset-Mask2Former/KuangshanDataset_Mask2Former_1.py")
    parser.add_argument("--checkpoint", default="work_dirs/KuangshanDataset-Mask2Former/best_mIoU_iter_68000.pth")
    parser.add_argument("--img-dir", default="data/yunnan_dataset_real/img_dir/test")
    parser.add_argument("--out-dir", default="outputs/yunnan_predict_Mask2f")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opacity", type=float, default=0.3)
    args = parser.parse_args()

    # 初始化模型
    model = init_model(args.config, args.checkpoint, device=args.device)

    # 调色板
    palette_dict = build_palette_dict(PALETTE)

    # 输出目录（修稳：不会因为重复运行报错，也不依赖 outputs/ 是否存在）
    out_viz_dir = os.path.join(args.out_dir, "testset-pred")
    out_mask_dir = os.path.join(args.out_dir, "mask-pred")
    os.makedirs(out_viz_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # 扫描图片目录
    img_dir = args.img_dir
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img-dir not found: {img_dir}")

    files = [f for f in os.listdir(img_dir) if is_image_file(f)]
    files.sort()

    for name in tqdm(files, desc="Infer"):
        img_path = os.path.join(img_dir, name)
        process_single_img(
            model=model,
            img_path=img_path,
            palette_dict=palette_dict,
            opacity=args.opacity,
            out_viz_dir=out_viz_dir,
            out_mask_dir=out_mask_dir,
            save=True
        )


if __name__ == "__main__":
    main()