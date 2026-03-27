https://www.codabench.org/competitions/14065/#/pages-tab

https://xiangbogaobarry.github.io/GIVE-CVPR-VGBE-2026/

https://huggingface.co/datasets/xiangbog/Generic-Instructional-Video-Editing-Challenge-Dataset/tree/main

Text-conditioned General Video Editing. Please download the dataset from GIVE-Challenge-Dataset

This challenge is jointly organized by Texas A&M University, Visko Platform, and Abaka AI.

Important Dates
2026.02.20: Release of Validation Data (Video + editing prompt); validation submission opens.
2026.03.25: Submission deadline.
2026.04.03: Technical report deadline for eligibility for the innovation award.
2026.04.06: Competition results released to participants.

Challenge Overview
The 1st Workshop on Video Generative Models: Benchmarks and Evaluation (VGBE) will be held in June 2026 in conjunction with CVPR 2026.

Recent advances in video generative models, such as Sora, Veo, and Wan, have demonstrated an unprecedented ability to generate high-fidelity, visually stunning content from simple text prompts. As these models move from pure generation toward practical creative workflows, the focus is shifting to video editing. This transition is crucial because real-world applications—ranging from film production to robotics simulation—require granular control and creative iteration. Editing allows for the refinement of specific elements, like swapping a character's outfit or changing the weather.

However, precise video editing remains a significant challenge due to the strict requirements of video quality, temporal consistency and exclusivity of edit. Unlike static image editing, a video edit must remain perfectly stable across time to avoid "flickering" or "drifting" pixels. Furthermore, ensuring a model modifies only the intended content—such as changing a car's color without altering the background or lighting—requires a deep semantic understanding of 3D geometry and physical interactions. Achieving this level of robust, instruction-driven control while maintaining visual realism is very challenging.

Hosting this challenge accelerates the development of video models that can move beyond simple generation toward precise, instruction-based controllability. It provides a standardized benchmark to evaluate how effectively these systems can maintain temporal consistency and spatial exclusivity in diverse, real-world editing scenarios.

The top ranked participants will be awarded and invited to describe their solution to the associated VGBE workshop at CVPR 2026.
The results of the challenge will be published at VGBE 2026 workshop (CVPR Proceedings).


Task Definition
Task: Text-conditioned General Video Editing

Given an Original Image and an Editing Text Prompt the model must generate a video that:

Instruction Following: Does the edited video accurately reflect the semantic intent of the instruction?
Rendering Quality: Is the edited video temporally consistent and visually realistic?
Exclusivity of Edit: Has the model modified only the intended content without introducing unnecessary changes?
Output Specifications
To ensure fairness and standardized evaluation, all submissions must adhere to the following technical constraints:

Frame: The generated video sequence must have strictly the same number of frames as the original video.
Resolution: Minimum: 480p (e.g., 854×480).
Recommended: 720p (e.g., 1280×720) or higher for optimal evaluation of fine-grained details.
Aspect Ratio: The output video must preserve the aspect ratio of the input video. Cropping or distorting the input aspect ratio will result in great score deduction.
Recommended Baselines / Architectures
We encourage participants to explore or build upon recent efficient architectures, such as:

PISCO : Precise Video Instance Insertion with Sparse Control
VACE: All-in-One Video Creation and Editing
Any closed-source or open-source model / pipeline is welcome.


Evaluation
The evaluation process consists of two primary components to ensure both technical excellence and practical utility:

Automated Evaluation (VBench): We utilize VBench to provide an objective assessment of video quality, focusing on technical metrics and perceptual fidelity.
Human Evaluation: Following the April 20 submission deadline, a panel of experts will score each entry across three key dimensions:
Instruction Following: Does the edited video accurately reflect the semantic intent of the text prompt?
Rendering Quality: Is the video temporally consistent and visually realistic?
Exclusivity of Edit: Did the model modify only the intended content without introducing artifacts or unnecessary changes?
Human Evaluation Score: Calculated as the average of the three dimensions above. Detailed scoring rubrics will be released at a later date.

Final Score Calculation
To balance objective performance with human-centric quality, the final ranking is determined by: Final Score=0.2×VBench Score+0.8×Human Evaluation Score


Awards
We have established a total prize pool of $1,000 USD. The tentative distribution is as follows:

Highest Score Award (Champion): $500 USD + Award Certificate

Innovation Award: $500 USD + Award Certificate
	This award recognizes technically novel, methodologically inspiring, or practically impactful contributions. A technical report is required to be eligible.


Issues & Contact
Technical Discussions: Please utilize the community forum on the official challenge page.
Inquiries: For specific questions, contact the organizing committee at tcve-cvpr-2026@googlegroups.com.

---
# Update : 2023/03/24
Dear Participants,

We would like to share two important updates regarding the Generic Instructional Video Editing (GIVE) Challenge.

First, the submission deadline has been extended to April 5th, 2026.

Second, some participants previously reported that they were unable to access the dataset. To make things easier and more consistent for everyone, we have updated the dataset download path. Please use the following link to download the dataset:

GIVE Dataset: https://huggingface.co/datasets/xiangbog/Generic-Instructional-Video-Editing-Challenge-Dataset

We hope this extension and updated download path will give you a smoother experience preparing your submission.

We encourage you to make the most of this opportunity — there is a $1,000 prize pool for the challenge, including the Highest Score Award (Champion) and the Innovation Award.

Good luck with your submission, and we look forward to seeing your exciting results!

Best regards,
The GIVE Challenge Organizers

----------------------------------------------------------------------------------
Dear Participants,

We would like to share several important updates regarding the Image-to-Video Consistent Generation Challenge at CVPR VGBE 2026.

The final submission deadline has been extended to April 5, 2026 (AOE). For the final phase, participants are required to submit their results together with the fact sheet.

At this final stage, the validation set contains 70 samples. Please make sure your submission is prepared accordingly.

For the fact sheet, please use the following template:
Fact Sheet Template: https://www.overleaf.com/read/jpdrctbsbxcr#9c6e1c

For the final submission, please include your videos, code, checkpoints(or accessible links to the checkpoints, e.g., Hugging Face), and execution instructions/README in one single ZIP file.
We understand that model checkpoints can be very large, so you may instead provide us with your Hugging Face link for the model weights

If you have difficulty submitting the materials through the platform, you may also share a Google Drive link by email at mingyang@tamu.edu.

We also encourage you to make the most of this opportunity. The challenge features a total prize pool of $1,900 USD, with the following tentative awards:

1st Place: $1,000 USD + Award Certificate
2nd Place: $600 USD + Award Certificate
3rd Place: $300 USD + Award Certificate
Good luck with your submission, and we look forward to seeing your exciting results!

Best regards,
The CVPR 2026 VGBE Challenge Organizers