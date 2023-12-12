"""Reconstruct images from anomaly to normal using SAM and Stable Diffusion Inpainting"""
import os
import numpy as np
import torch
import yaml
from PIL import Image
from pathlib import Path
from segment_anything import build_sam, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapter


class Reconstructer:
    DATASET = "mtd"
    STRENGTHS = [0.1, 0.2, 0.5, 0.7]
    IMAGE_SIZE = (512, 512)

    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.parser, self.enable_mask = self._read_data_setting()
        self.sam_predictor, self.ip_model = self._load_models()

    def _read_data_setting(self):
        with open(f"data/{self.DATASET}.yaml", "r") as stream:
            parser = yaml.load(stream, Loader=yaml.CLoader)
        return parser, parser["dataset"]["mask_enable"]

    def _load_models(self):
        if self.enable_mask:
            # Load SAM model
            sam = build_sam(checkpoint="models/sam_vit_h_4b8939.pth")
            sam.to('cuda')
            sam_predictor = SamAutomaticMaskGenerator(
                    sam, pred_iou_thresh=0.98, stability_score_thresh=0.92, min_mask_region_area=10000)
        else:
            sam_predictor = None

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
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
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
    
    def run_all(self):
        for strength in self.STRENGTHS:
            self.strength = strength
            self.run()

    def run(self):
        for category in self.parser["categorys"]:
            print(f'Processing {category}...')

            # Load reference image
            root_path = Path(self.parser["dataset"]["path"])
            folder_path = root_path / category / self.parser["dataset"]["splite"]
            file_names = [x for x in (folder_path/"good").iterdir() if x.is_file()]
            ref_image_path = folder_path / file_names[0]
            print(f'Reference image: {ref_image_path}')

            for anomaly_type in os.listdir(folder_path):
                self.save_folder_path = self.save_path / f"strength_{self.strength}" / category / anomaly_type
                os.makedirs(self.save_folder_path)
                for img_path in (folder_path/anomaly_type).iterdir():
                    print(img_path)
                    self.reconstruct(img_path, ref_image_path)

    def reconstruct(self, img_path: str, ref_image_path: str):
        # Load demo image
        image_source = Image.open(img_path).convert("RGB")
        image_source = np.asarray(image_source)
        # Load reference image
        ref_image = Image.open(ref_image_path)
        ref_image.resize(self.IMAGE_SIZE)

        # Image Inpainting
        image_source_pil = Image.fromarray(image_source)

        if self.enable_mask:
            # Run the segmentation model
            masks = self.sam_predictor.generate(image_source)
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            image_mask = ~masks[0]['segmentation']
            image_mask_pil = Image.fromarray(image_mask)
            image_mask_for_inpaint = image_mask_pil.resize(self.IMAGE_SIZE)
            image_mask_for_inpaint.save(self.save_folder_path / f"mask_{img_path.name}")
        else:
            image_mask_for_inpaint = Image.new("L", self.IMAGE_SIZE, (255))
        
        image_source_for_inpaint = image_source_pil.resize(self.IMAGE_SIZE)
        image_source_for_inpaint.save(self.save_folder_path / f"source_{img_path.name}")

        image_inpainting = self.ip_model.generate(pil_image=ref_image, num_samples=1, num_inference_steps=50,
            seed=42, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint, strength=self.strength)[0]
        image_inpainting.save(self.save_folder_path / f"inpaint_{img_path.name}")


if __name__ == "__main__":
    reconstructer = Reconstructer("/mnt/d/reconstruct")
    reconstructer.run_all()
