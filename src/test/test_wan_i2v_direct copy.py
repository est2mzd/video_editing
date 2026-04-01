# test_wan_i2v_direct.py
# ============================================
# 目的：
#   Wan I2V を direct 呼び出しでテスト
# ============================================

import sys
from types import SimpleNamespace

# Wanパス追加
sys.path.append("/workspace/third_party/Wan2.1")

from generate import generate

# ======== 画像の前処理 ========
from PIL import Image
img_path_ori = "/workspace/third_party/Wan2.1/examples/girl.png"
img_path_resized = "/workspace/src/test/girl_resized.png"
img = Image.open(img_path_ori)
img = img.resize((832, 480))  # ← sizeと完全一致
img.save(img_path_resized)

print(f"Original image size: {Image.open(img_path_ori).size}")
print(f"Resized image size: {Image.open(img_path_resized).size}")


# ========= argsを手動構築 =========
args = SimpleNamespace()

# --- 必須 ---
args.task = "i2v-14B"
args.size = "832*480"
args.frame_num = 1 # 4n+1
args.ckpt_dir = "/workspace/weights/Wan2.1-I2V-14B-480P"
args.prompt = "A small girl clearly waving both hands, large motion, dynamic movement"
args.image = img_path_resized

# --- optional（重要）---
args.offload_model = True
args.sample_steps = 2
args.sample_solver = "unipc"
args.sample_shift = 3.0
args.sample_guide_scale = 5.0

# --- 未使用だが必要な場合あり ---
args.ulysses_size = 1
args.ring_size = 1
args.t5_fsdp = False
args.t5_cpu = True   # --- ここが重要 ---
args.dit_fsdp = False
args.save_file = None
args.src_video = None
args.src_mask = ""
args.src_ref_images = None
args.use_prompt_extend = False
args.prompt_extend_method = "local_qwen"
args.prompt_extend_model = None
args.prompt_extend_target_lang = "zh"
args.base_seed = 12345
args.first_frame = None
args.last_frame = None

# ========= 実行 =========
print("Running Wan I2V (direct)...")
generate(args)