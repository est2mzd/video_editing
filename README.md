# video_editing

## Submission Run

### Files
- Config: `/workspace/config/submit.yaml`
- Main runner: `/workspace/scripts/run_submit_package.sh`
- Wrapper: `/workspace/src/run_submit.sh`

### What the script does
1. Runs full processing for all rows in `/workspace/data/annotations.jsonl`
2. Writes submission mp4 files under `/workspace/logs/submit/all_<timestamp>/`
3. Creates zip at `/workspace/logs/submit/all_<timestamp>/submission_mp4.zip`

### Run

```bash
chmod +x /workspace/scripts/run_submit_package.sh /workspace/src/run_submit.sh
/workspace/scripts/run_submit_package.sh /workspace/config/submit.yaml
```

or

```bash
/workspace/src/run_submit.sh /workspace/config/submit.yaml
```

### Output
- mp4 files: `/workspace/logs/submit/all_<timestamp>/*.mp4`
- zip: `/workspace/logs/submit/all_<timestamp>/submission_mp4.zip`
- run log: `/workspace/logs/submit/all_<timestamp>/run_video_editor_v2.log`
- instruction logs: `/workspace/logs/submit/all_<timestamp>/instruction_logs/*.log`