# video_editing

## Submission Files

Submission files are available below:

[videos.zip in Google Drive](https://drive.google.com/drive/folders/1IK7exchc_GJ7HtC0l1OOzmifw_XApu7D?usp=sharing)

---

## How to Run (Validation)
### Clone the repository

```bash
git clone https://github.com/est2mzd/video_editing.git
cd video_editing
git checkout feature/M3_Submit
```

### Update paths
Update the paths in the config file if necessary:

Config: `./configs/submit.yaml`

The repository root corresponds to `/workspace/` inside the container.

```yaml
# Paths inside the container
annotation_path: /workspace/data/annotations.jsonl
video_dir: /workspace/data/videos
```

### Build Container and Run
```bash
./run_all.sh
```

### If you want to run commands manually inside the container:

```bash
chmod +x /workspace/scripts/run_submit_package.sh /workspace/src/run_submit.sh
/workspace/scripts/run_submit_package.sh /workspace/configs/submit.yaml
```

or

```bash
/workspace/scripts/run_submit.sh /workspace/configs/submit.yaml
```

### Output
- MP4 files: `/workspace/logs/submit/all_<timestamp>/*.mp4`
- ZIP: `/workspace/logs/submit/all_<timestamp>/submission_mp4.zip`
- run log: `/workspace/logs/submit/all_<timestamp>/run_video_editor_v2.log`
- instruction logs: `/workspace/logs/submit/all_<timestamp>/instruction_logs/*.log`

---

## Explanation

### Files
- Config: `/workspace/configs/submit.yaml`
- Main runner: `/workspace/scripts/run_submit_package.sh`
- Wrapper: `/workspace/src/run_submit.sh`

### What the script does
1. Runs full processing for all rows in `/workspace/data/annotations.jsonl`
2. Writes submission MP4 files under `/workspace/logs/submit/all_<timestamp>/`
3. Creates a submission ZIP at `/workspace/logs/submit/all_<timestamp>/submission_mp4.zip`