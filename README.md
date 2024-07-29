# Stable-Diffusion-Anomaly-Detection

Visual anomaly detection is essential for industrial quality inspection and medical diagnosis. Previous research in this field has focused on training custom models for each specific task, which requires thousands of images and annotation. In this work, we depart from this approach, drawing inspiration from reconstruction-based methodologies and leveraging the remarkable zero-shot generalization capabilities of foundation models. We propose a novel framework, Stable Diffusion Anomaly Detection (SDAD), which operates by reconstructing target images using pre-trained diffusion models and employs Segment Anything to enhance the adaptability of modern foundation models to anomaly detection. In VisA and MVTec-AD dataset, SDAD achieves state-of-the-art results in zero-shot visual anomaly detection without further tuning. This highlights the effectiveness of our framework in achieving superior anomaly detection performance without the task-specific constraints of traditional approaches.

## Results
![Visa1](/assets/VisA1.png)
![Visa2](/assets/VisA2.png)
![Visa3](/assets/VisA3.png)
![MVTec1](/assets/MVTec1.png)
![MVTec2](/assets/MVTec2.png )
![MVTec3](/assets/MVTec3.png )

## Installation

```bash
python -m pip install -r requirements.txt
```

**Download the pretrained weights**

```bash
mkdir models
cd models
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Inference
```bash
python scripts/main.py
```
