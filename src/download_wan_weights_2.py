from huggingface_hub import snapshot_download

snapshot_download(
    "Wan-AI/Wan2.1-I2V-14B-480P",
    local_dir="/workspace/weights/wan/Wan2.1-I2V-14B-480P",
    local_dir_use_symlinks=False
)