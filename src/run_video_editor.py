
from datetime import datetime
import os, sys
sys.path.append("/workspace/src")
import yaml
import json
import logging
from src.postprocess.dispatcher import run_method
from src.utils.video_utility import load_video, write_video
from src.parse.instruction_parser_v3_rulebase_trial013_singlefile_kai2 import (
    build_parser as build_parser_1
)
from src.parse.instruction_parser_v3_rulebase_trial020_singlefile import (
    build_parser as build_parser_2
)

# read config
config_path = "/workspace/configs/base_config.yaml"
config = yaml.safe_load(open(config_path, "r"))

# get parameters from config
DEBUG_MODE = config.get("debug_mode", True)
TARGET_SUBCLASS = config.get("target_subclass", "Dolly in")
video_dir = config["video_dir"]
output_dir_top = config["output_dir"]
groundingdino_config_path = config["groundingdino"]["config_path"]
groundingdino_checkpoint = config["groundingdino"]["checkpoint"]

# ================================================================
# Instruction parser
if config['parser_version'] == "v1":
    print("[Info] Using instruction parser version: v1")
    parser = build_parser_1()
elif config['parser_version'] == "v2":
    print("[Info] Using instruction parser version: v2")
    parser = build_parser_2()
else:
    print(f"Unsupported parser version: {config['parser_version']}. Defaulting to v2.")
    parser = build_parser_2()


# read annotation
annotation_path = config["annotation_path"]
annotation = [json.loads(line) for line in open(annotation_path, "r").readlines()]

videos_paths = [f"{video_dir}/{a['video_path']}" for a in annotation]
instructions = [a["instruction"] for a in annotation]

# ================================================================
# For Debug
if DEBUG_MODE or TARGET_SUBCLASS is not None:
    subclasses = [a["selected_subclass"] for a in annotation]
    target_ids = [i for i, a in enumerate(subclasses) if a == TARGET_SUBCLASS]
    videos_paths = [videos_paths[i] for i in target_ids]
    instructions = [instructions[i] for i in target_ids]
    #
    action_type = TARGET_SUBCLASS.replace(" ", "_").lower()
else:
    action_type = "all"


# ================================================================
# Create output directory
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# time_stamp = "test"
output_dir = f"{output_dir_top}/{action_type}_{time_stamp}"
os.makedirs(output_dir, exist_ok=True)

# ================================================================

for i, (video_path, instruction) in enumerate(zip(videos_paths, instructions)):
    print(f"Processing video {i+1}/{len(videos_paths)}: {video_path}")
    frames, fps, width, height = load_video(video_path)
    video_name = os.path.basename(video_path)
    parsed_instruction = parser.infer(instruction)
    print(f"   Parsed instruction: {parsed_instruction}")
    
    if config['parser_version'] == "v1":
        action = parsed_instruction.get("action", "zoom_in")
        target = parsed_instruction.get("target", ["face"])
        params = parsed_instruction.get("params", {})
        params["video_name"] = video_name  # Inject video name for progress display
        print(f"[Info][Task {i+1}] action: {action}, target: {target}, params: {params}")        
        
        out_frame = run_method(
            action=action,
            targets=target,
            frames=frames,
            params=params,
            instruction=instruction,
            logger=logging.getLogger("zoom_in")
        )

        out_path = f"{output_dir}/{video_name}"
        write_video(out_path, out_frame, fps, width, height)
    
    elif config['parser_version'] == "v2":
        max_task_num = config.get("max_task_num", 1)
        
        for i, task in enumerate(parsed_instruction.get("tasks", [])):
            if i >= max_task_num:
                break
            action = task.get("action", "zoom_in")
            target = task.get("target", ["face"])
            params = task.get("params", {})
            params["video_name"] = video_name  # Inject video name for progress display
            print(f"[Info][Task {i+1}] action: {action}, target: {target}, params: {params}")
            
            out_frames = run_method(
                action=action,
                targets=target,
                frames=frames,
                params=params,
                instruction=instruction,
                logger=logging.getLogger("dispatcher_v2")
            )

            out_path = f"{output_dir}/{video_name}"
            write_video(out_path, out_frames, fps, width, height)

print(f"Output saved to: {output_dir}")
