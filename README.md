# Stable-Diffusion-Anomaly-Detection

## ABSTRACT
Visual anomaly detection is essential for industrial quality inspection and medical diagnosis. Previous research in this field has focused on training custom models for each specific task, which requires thousands of images and annotation. In this work, we depart from this approach, drawing inspiration from reconstruction based methodologies and leveraging the remarkable zero-shot generalization capabilities of foundation models. We propose a novel framework, Stable Diffusion Anomaly Detection (SDAD), which operates by reconstructing target images using pre-trained diffusion models and employs Segment Anything to enhance the adaptability of modern foundation models to anomaly detection. In VisA dataset, SDAD achieves pixel level state-of-the-art results in zero-shot visual anomaly detection without further tuning. This highlights the effectiveness of our framework in achieving superior anomaly detection performance without the task-specific constraints of traditional approaches.

## SDAD Framework
Our proposed anomaly detection framework, Stable Diffusion Anomaly Detection (SDAD), consists of three components which are background remover, image reconstructor, and change detector, as shown in Figure 1. The initial step in the SDAD framework involves the use of a background remover to generate a mask that isolates the object from the background, allowing the image reconstructor to focus more attentively on the object itself. This strategic step is essential, as defects are expected to manifest on the objects rather than on the background. Second, we employ a denoising U-Net as an image reconstructor to transform the defective image into a defect-free version. Leveraging that image reconstructor generates samples representing the entire distribution of normal samples while being incapable of generating samples deviating from that distribution. This enables the detection of anomalies by comparing the anomalous input with its predicted flawless reconstruction. Lastly, the change detector component is employed to compare the input image with the reconstructed image to identify and highlight the differences, thereby producing the resultant mask highlighting detected anomalies. Through this multi-stage process, our framework offers a comprehensive solution for anomaly detection, leveraging the strengths of each component to effectively identify and delineate anomalies within the input images.

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
