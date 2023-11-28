"""Reconstruct images from anomaly to normal using SAM and Stable Diffusion Inpainting"""
import os
import numpy as np
import torch
from PIL import Image
from segment_anything import build_sam, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapter


STRENGTH = 0.5

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
    ref_image = Image.open(ref_image_path)
    ref_image.resize(IMAGE_SIZE)

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
    image_source_for_inpaint.save(f"{save_path}/source_{os.path.basename(local_image_path)}")
    image_mask_for_inpaint.save(f"{save_path}/mask_{os.path.basename(local_image_path)}")

    image_inpainting = ip_model.generate(pil_image=ref_image, num_samples=1, num_inference_steps=50,
        seed=42, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint, strength=STRENGTH)[0]
    image_inpainting.save(f"{save_path}/inpaint_{os.path.basename(local_image_path)}")


def load_models():
    # Load SAM model
    sam = build_sam(checkpoint="models/sam_vit_h_4b8939.pth")
    sam.to('cuda')
    sam_predictor = SamAutomaticMaskGenerator(
            sam, pred_iou_thresh=0.98, stability_score_thresh=0.92, min_mask_region_area=10000)

    # Load stable diffusion inpainting models
    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # load SD pipeline
    pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    pipe = pipe.to("cuda")

    # load ip-adapter
    image_encoder_path = "models/image_encoder/"
    ip_ckpt = "models/ip-adapter_sd15.bin"
    device = "cuda"
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
    return sam_predictor, ip_model


if __name__ == "__main__":
    sam_predictor, ip_model = load_models()
    main(f"/mnt/d/reconstruct/strength_{STRENGTH}")
