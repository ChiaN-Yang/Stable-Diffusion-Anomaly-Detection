"""Reconstruct images from anomaly to normal using SAM and Stable Diffusion Inpainting"""
import os
import numpy as np
import torch
from PIL import Image
from segment_anything import build_sam, SamAutomaticMaskGenerator
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import StableDiffusionInpaintPipeline


STRENGTH = 0.1

def main(save_path):
    folder_types = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    for folder_type in folder_types:
        print(f'Processing {folder_type}...')
        run(save_path, folder_type)


def run(save_path, anomaly_type):
    os.makedirs(os.path.join(save_path, anomaly_type, 'Anomaly'))
    os.makedirs(os.path.join(save_path, anomaly_type, 'Normal'))

    folder_path = f"/mnt/d/dataset/VisA_20220922/{anomaly_type}/Data/Images/Normal"
    file_names = os.listdir(folder_path)
    ref_image_name = os.path.join(folder_path, file_names[0])
    print(f'Reference image: {ref_image_name}')

    for category in ['Anomaly', 'Normal']:
        folder_path = os.path.join(f'/mnt/d/dataset/VisA_20220922/{anomaly_type}/Data/Images/', category)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            print(img_path)
            reconstruct(save_path, img_path, ref_image_name, anomaly_type, category)


def reconstruct(save_path: str, local_image_path: str, ref_image_path:str, anomaly_type:str, category:str):
    IMAGE_SIZE = (512, 512)
    # Load demo image
    image_source = Image.open(local_image_path).convert("RGB")
    image_source = np.asarray(image_source)
    # Load reference image
    ref_image = Image.open(ref_image_path).convert("RGB")

    # Run the segmentation model
    masks = sam_predictor.generate(image_source)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    image_mask = ~masks[0]['segmentation']

    # Image Inpainting
    image_source_pil = Image.fromarray(image_source)
    image_mask_pil = Image.fromarray(image_mask)

    save_path = os.path.join(save_path, anomaly_type, category)
    image_source_for_inpaint = image_source_pil.resize(IMAGE_SIZE)
    image_mask_for_inpaint = image_mask_pil.resize(IMAGE_SIZE)
    ref_image = ref_image.resize(IMAGE_SIZE)
    image_source_for_inpaint.save(f"{save_path}/source_{os.path.basename(local_image_path)}")
    image_mask_for_inpaint.save(f"{save_path}/mask_{os.path.basename(local_image_path)}")

    ip_model.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    generator = torch.Generator(device="cpu").manual_seed(33)
    image_inpainting = ip_model(
        prompt='best quality, high quality', 
        image = image_source_for_inpaint,
        mask_image = image_mask_for_inpaint,
        ip_adapter_image=ref_image,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
        num_images_per_prompt=1, 
        num_inference_steps=50,
        generator=generator,
        strength=STRENGTH,
    ).images
    image_inpainting[0].save(f"{save_path}/inpaint_{os.path.basename(local_image_path)}")


def load_models():
    # Load SAM model
    sam = build_sam(checkpoint="models/sam_vit_h_4b8939.pth")
    sam.to('cuda')
    sam_predictor = SamAutomaticMaskGenerator(
            sam, pred_iou_thresh=0.98, stability_score_thresh=0.92, min_mask_region_area=10000)

    # Load stable diffusion inpainting models
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", 
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    ).to("cuda")

    # load SD pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        image_encoder = image_encoder, torch_dtype=torch.float16, safety_checker=None)
    pipeline = pipeline.to("cuda")
    return sam_predictor, pipeline


if __name__ == "__main__":
    sam_predictor, ip_model = load_models()
    main(f"/mnt/d/reconstruct/inpaint_strength_{STRENGTH}")
