# 1. Method Overview
This system is a modular video editing pipeline designed for instruction-guided edits in a challenge setting. The input is a video and a natural language instruction, and the output is an edited video that reflects the requested transformation.

A key design choice is a fully rule-based instruction parser in the final pipeline. The parser extracts two core fields from text: action (what to change) and target (which object/region to change). No LLM is used at inference time. This improves determinism, reduces runtime variability, and supports straightforward reproducibility across machines and runs.

The system is organized as independent components for parsing, localization, editing, and temporal stabilization. This modular structure supports maintainability and extension to additional edit types.

# 2. System Pipeline
The end-to-end pipeline is:

## 1. Input
    - Video sequence
    - Text instruction

## 2. Instruction Parsing
    - Rule-based parsing extracts:
        - action (e.g., recolor, replace background, apply style, camera-like motion)
        - target (object category or scene region)

## 3. Target Detection and Mask Generation
    - GroundingDINO detects instruction-relevant objects/regions.
    - SAM refines detections into segmentation masks for precise spatial control.

## 4. Action-Conditioned Editing
    - Color / attribute editing: OpenCV-based operations (HSV transforms and LUT-based mapping).
    - Background editing: Foreground-background separation via segmentation; then replacement or inpainting.
    - Camera motion editing: Monocular depth estimation (MiDaS or Depth-Anything), then depth-aware warping.
    - Style editing: Diffusion-based stylization using Stable Diffusion, with ControlNet when structural guidance is needed.

## 5. Temporal Consistency
    - RAFT optical flow is applied to stabilize edits across frames and reduce flicker.

## 6. Output
    - Edited video sequence with frame-level coherence improvements.

# 3. Models and Techniques
The system uses only pretrained components:

- GroundingDINO: text-conditioned object detection for target localization.
- Segment Anything (SAM): mask extraction from detected regions.
- MiDaS / Depth-Anything: monocular depth prediction for approximate geometry-aware warping.
- Stable Diffusion / ControlNet (optional): style-oriented generation and constrained editing.
- RAFT: optical flow estimation for temporal stabilization.
- OpenCV: deterministic pixel-level edits (HSV, LUT, compositing utilities).

Technique selection is action-dependent, allowing lightweight deterministic processing when possible and generative processing only when needed.

# 4. Training
No additional model training or fine-tuning is performed for this submission.

- All modules are used in inference mode with pretrained weights.
- The pipeline focuses on orchestration, mask quality, edit control logic, and temporal stabilization rather than task-specific retraining.

This design reduces engineering overhead and improves reproducibility for challenge evaluation.

# 5. Data
Evaluation is conducted on the official challenge validation dataset.

- No extra private training data is used for adaptation.
- No additional supervised optimization is introduced in this system version.

# 6. Strengths
- Deterministic behavior: Rule-based parsing and explicit module routing make the pipeline stable and predictable.
- High reproducibility: Pretrained-only setup and fixed rule logic support consistent reruns.
- Modular architecture: Components can be replaced or upgraded independently (parser, detector, segmenter, editor, temporal module).
- Action-aware processing: Different edit classes are handled with appropriate tools rather than a one-size-fits-all approach.

# 7. Limitations
- Parser generalization: Rule-based instruction parsing may fail on unseen phrasing, ambiguous language, or complex multi-step commands.
- Temporal artifacts in generative edits: Diffusion-based frame edits can still produce residual flicker or drift despite flow-based stabilization.
- Approximate camera motion: Depth-based warping relies on monocular depth estimates and does not recover full 3D scene geometry, which can cause distortions in challenging scenes.
- Dependency sensitivity: Quality depends on detection and segmentation reliability; failures in localization propagate to downstream edits.

# 8. Reproducibility
To support replication and verification:

- Full source code is included in the submission.
- Execution scripts and run instructions are included.
- Pretrained model weights are provided via Hugging Face links.
- The final inference pipeline does not rely on non-deterministic LLM parsing.
- The system uses fixed module interfaces and explicit processing stages, enabling reproducible end-to-end runs.

This Fact Sheet is intended to provide a concise, implementation-focused description of the submitted system for workshop review and reproducibility checks.