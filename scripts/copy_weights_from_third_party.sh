mkdir -p /workspace/weights/groundingdino/
cp /workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth /workspace/weights/groundingdino/

mkdir -p /workspace/weights/raft/
cp /workspace/third_party/RAFT/models/raft-small.pth /workspace/weights/raft/
cp /workspace/third_party/RAFT/models/raft-chairs.pth /workspace/weights/raft/
cp /workspace/third_party/RAFT/models/raft-sintel.pth /workspace/weights/raft/
cp /workspace/third_party/RAFT/models/raft-kitti.pth /workspace/weights/raft/
cp /workspace/third_party/RAFT/models/raft-things.pth /workspace/weights/raft/

mkdir -p /workspace/weights/wan2.1/
cp /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors /workspace/weights/wan2.1/
cp /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth /workspace/weights/wan2.1/
cp /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth /workspace/weights/wan2.1/