# video_editing

## Submission Run

### Files
- Config: `/workspace/configs/submit.yaml`
- Main runner: `/workspace/scripts/run_submit_package.sh`
- Wrapper: `/workspace/src/run_submit.sh`

### What the script does
1. Runs full processing for all rows in `/workspace/data/annotations.jsonl`
2. Writes submission mp4 files under `/workspace/logs/submit/all_<timestamp>/`
3. Creates zip at `/workspace/logs/submit/all_<timestamp>/submission_mp4.zip`

### For validation
Change the path of the config file. 

Config: `/workspace/configs/submit.yaml`
```yaml
annotation_path: /workspace/data/annotations.jsonl
video_dir: /workspace/data/videos
```

### Build Continer and Run
```bash
./run_all.sh
```

### If you want to run in the container

```bash
chmod +x /workspace/scripts/run_submit_package.sh /workspace/src/run_submit.sh
/workspace/scripts/run_submit_package.sh /workspace/configs/submit.yaml
```

or

```bash
/workspace/scripts/run_submit.sh /workspace/configs/submit.yaml
```

### Output
- mp4 files: `/workspace/logs/submit/all_<timestamp>/*.mp4`
- zip: `/workspace/logs/submit/all_<timestamp>/submission_mp4.zip`
- run log: `/workspace/logs/submit/all_<timestamp>/run_video_editor_v2.log`
- instruction logs: `/workspace/logs/submit/all_<timestamp>/instruction_logs/*.log`