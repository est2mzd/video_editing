# third_party 大容量モデル整理

## 1. /workspace/third_party 配下の容量が大きいモデルファイル

- 10835.57 MB: `/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth`
- 6815.02 MB: `/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors`
- 661.85 MB: `/workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth`
- 484.09 MB: `/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth`

抽出条件:
- 拡張子: `.pth .pt .bin .safetensors .ckpt .onnx .pb .tflite .pdparams`
- サイズ: 100MB 以上

## 2. 上記のうち /workspace/weights 配下にあるもの / ないもの

### あるもの

- `/workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth`
	- `/workspace/weights/groundingdino/groundingdino_swint_ogc.pth`
- `/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth`
	- `/workspace/weights/wan2.1/Wan2.1_VAE.pth`
- `/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors`
	- `/workspace/weights/wan2.1/diffusion_pytorch_model.safetensors`
- `/workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth`
	- `/workspace/weights/wan2.1/models_t5_umt5-xxl-enc-bf16.pth`

### ないもの

- なし

## 3. third_party 側を削除して weights 側へのシンボリックリンクに置換するコマンド

```bash
cd /workspace

# 1) GroundingDINO
rm -f /workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth
ln -s /workspace/weights/groundingdino/groundingdino_swint_ogc.pth \
  /workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth

# 2) VACE Wan2.1 VAE
rm -f /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth
ln -s /workspace/weights/wan2.1/Wan2.1_VAE.pth \
  /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth

# 3) VACE Wan2.1 diffusion weights
rm -f /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors
ln -s /workspace/weights/wan2.1/diffusion_pytorch_model.safetensors \
  /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors

# 4) VACE Wan2.1 text encoder weights
rm -f /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth
ln -s /workspace/weights/wan2.1/models_t5_umt5-xxl-enc-bf16.pth \
  /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth
```

確認コマンド:

```bash
ls -l \
  /workspace/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth \
  /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors \
  /workspace/third_party/VACE/models/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth
```

## 4. /workspace/third_party/RAFT/models も容量に関係なく同様に置換するコマンド

対象ファイル:

- `/workspace/third_party/RAFT/models/raft-kitti.pth` -> `/workspace/weights/raft/raft-kitti.pth`
- `/workspace/third_party/RAFT/models/raft-sintel.pth` -> `/workspace/weights/raft/raft-sintel.pth`
- `/workspace/third_party/RAFT/models/raft-small.pth` -> `/workspace/weights/raft/raft-small.pth`
- `/workspace/third_party/RAFT/models/raft-things.pth` -> `/workspace/weights/raft/raft-things.pth`

```bash
cd /workspace

rm -f /workspace/third_party/RAFT/models/raft-kitti.pth
ln -s /workspace/weights/raft/raft-kitti.pth \
  /workspace/third_party/RAFT/models/raft-kitti.pth

rm -f /workspace/third_party/RAFT/models/raft-sintel.pth
ln -s /workspace/weights/raft/raft-sintel.pth \
  /workspace/third_party/RAFT/models/raft-sintel.pth

rm -f /workspace/third_party/RAFT/models/raft-small.pth
ln -s /workspace/weights/raft/raft-small.pth \
  /workspace/third_party/RAFT/models/raft-small.pth

rm -f /workspace/third_party/RAFT/models/raft-things.pth
ln -s /workspace/weights/raft/raft-things.pth \
  /workspace/third_party/RAFT/models/raft-things.pth
```

確認コマンド:

```bash
ls -l \
  /workspace/third_party/RAFT/models/raft-kitti.pth \
  /workspace/third_party/RAFT/models/raft-sintel.pth \
  /workspace/third_party/RAFT/models/raft-small.pth \
  /workspace/third_party/RAFT/models/raft-things.pth
```
