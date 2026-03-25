
# /workspace/third_party/Wan2.1/README.md を参照

# User Settings
CKPT_DIR=/workspace/weights/Wan2.1-T2V-14B
PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
PY_PATH=/workspace/third_party/Wan2.1/generate.py

# Run Wan2.1
python3 $PY_PATH  --task t2v-14B --size 1280*720 --ckpt_dir $CKPT_DIR --prompt "$PROMPT"