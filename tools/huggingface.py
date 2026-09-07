from huggingface_hub import HfApi

# ========= 基本信息 =========
REPO_ID = "therrr/dinoswin"  # 改成你自己的
REPO_TYPE = "model"

api = HfApi()

# ========= 上传权重 =========
api.upload_file(
    path_or_fileobj="work/mmsegmentation_25717_mine/mmsegmentation/work_dirs/dinov3_swin_upernet_swinmainE/epoch_89.pth",
    path_in_repo="dinoswin/best_mIoU.pth",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)
s
# ========= 上传推理 config =========
api.upload_file(
    path_or_fileobj="work/mmsegmentation_25717_mine/mmsegmentation/work_dirs/dinov3_swin_upernet_swinmainE/config_infer.py",
    path_in_repo="dinoswin/config_infer.py",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)
# ========= 上传推理 config =========
api.upload_file(
    path_or_fileobj="work/mmsegmentation_25717_mine/mmsegmentation/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    path_in_repo="dinoswin/pretrained/dinov3_vitl16_pretrain_sat493m.pth",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)
# ========= 上传推理 config =========
api.upload_file(
    path_or_fileobj="work/mmsegmentation_25717_mine/mmsegmentation/work_dirs/dinov3_swin_upernet_swinmainE/config_infer.py",
    path_in_repo="dinoswin/pretrained/swin_base_patch4_window7.pth",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

print("Upload finished successfully.")
