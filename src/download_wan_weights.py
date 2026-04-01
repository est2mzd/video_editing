from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-14B",
    local_dir="/workspace/weights/Wan2.1-T2V-14B",
    local_dir_use_symlinks=False
)