"""Detect anomaly using SSIM."""
import os
import csv
import cv2
import numpy as np
import yaml
from skimage.metrics import structural_similarity
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import square, area_opening, binary_dilation
from skimage.io import imsave
from skimage.util import img_as_ubyte
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry


class AnomalyDetector:
    STRENGTH = 0.1
    STRENGTHS = [0.1, 0.2, 0.5, 0.7]
    THRESHOLDS = [77, 100, 127, 147, 177, 200]
    IMAGE_SIZE = (512, 512)
    AREA_THRESH = 120

    def __init__(self, reconstrcted_img_path: str, results_path: str, dataset: str):
        self.reconstrcted_img_path = Path(reconstrcted_img_path)
        self.base_results_path = Path(results_path)
        self.parser, self.enable_mask = self._read_data_setting(dataset)
        self.mask_predictor = self._load_models()

    @staticmethod
    def _read_data_setting(dataset):
        with open(f"data/{dataset}.yaml", "r") as stream:
            parser = yaml.load(stream, Loader=yaml.CLoader)
        return parser, parser["dataset"]["mask_enable"]

    def _load_models(self):
        # Load SAM model
        sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
        sam.to('cuda')
        mask_predictor = SamPredictor(sam)
        return mask_predictor

    def run_all_strength(self):
        for strength in self.STRENGTHS:
            self.STRENGTH = strength
            self.run_all_threshold()

    def run_all_threshold(self):
        for thresh in self.THRESHOLDS:
            self.lower_thresh = thresh
            self.run()

    def run(self):
        results_path = self.base_results_path/f"strength_{self.STRENGTH}_thresh_{self.lower_thresh}"
        for category in self.parser["categorys"]:
            print(f'Processing {category}...')
            category_path = results_path / category
            os.makedirs(category_path)

            csv_path = category_path / "output.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['name', 'ground_truth', 'anomaly_score', 'thresh_score',
                                 'ssim_score', 'mask_area'])

            original_data_path = Path(self.parser["dataset"]["path"])
            for anomaly_type in os.listdir(original_data_path / category / self.parser["dataset"]["splite"]):
                os.makedirs(category_path / anomaly_type)
                folder_path = self.reconstrcted_img_path / f"strength_{self.STRENGTH}" / category / anomaly_type
                source_images = [x for x in folder_path.iterdir() if x.name.split('_')[0]=='source']
                for img_file in source_images:
                    img_name = img_file.name.split('_')[1]
                    true_label= 0 if anomaly_type == "good" else 1
                    anomaly_score, thresh_score, ssim_score, mask_area = self.detect(
                            folder_path, img_name, results_path/category/anomaly_type)
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([img_file, true_label, anomaly_score, thresh_score, ssim_score, mask_area])

    def detect(self, folder_path, file_name, results_dir):
        source_img_path = folder_path / f'source_{file_name}'
        inpaint_image_name = folder_path / f'inpaint_{file_name}'
        # Compute SSIM
        # Load images
        print(source_img_path)
        before = cv2.imread(str(source_img_path))
        after = cv2.imread(str(inpaint_image_name))

        # reseze image
        before = cv2.resize(before, self.IMAGE_SIZE)
        after = cv2.resize(after, self.IMAGE_SIZE)
        cv2.imwrite(str(results_dir/f"before_{file_name}"), before)

        # Convert images to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        if self.enable_mask:
            mask_path = folder_path / f'mask_{file_name}'
            mask = cv2.imread(str(mask_path), 0)
            before_gray = cv2.bitwise_and(before_gray, before_gray, mask=mask)
            after_gray = cv2.bitwise_and(after_gray, after_gray, mask=mask)

        # Compute SSIM between the two images
        (score, diff) = structural_similarity(
                before_gray, after_gray, gaussian_weights=True, full=True)

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1]
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")
        diff_inv = cv2.bitwise_not(diff)
        diff_inv_rgb = cv2.cvtColor(diff_inv, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(results_dir/f"diff_{file_name}"), diff_inv)
        bw = area_opening(diff_inv > self.lower_thresh, self.AREA_THRESH, connectivity=5)
        bw = binary_dilation(bw, square(9))
        bw_write = img_as_ubyte(bw)
        imsave(str(results_dir/f"bw_{file_name}"), bw_write)

        # remove artifacts connected to image border
        cleared = clear_border(bw)
        cleared = img_as_ubyte(cleared)
        cv2.imwrite(str(results_dir/f"cleared_{file_name}"), cleared)
        # label image regions
        label_image, num_label = label(cleared, return_num=True, connectivity=2)
        max_area = 0
        leftr = leftc = rightr = rightc = 0
        mask_area = []
        mask_data = np.zeros_like(before_gray)
        if num_label:
            for region in regionprops(label_image):
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = cv2.rectangle(diff_inv_rgb, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
                mask_area.append(region.area)
                if region.area > max_area:
                    max_area = region.area
                    leftr, leftc, rightr, rightc = region.bbox
            cv2.imwrite(str(results_dir/f"rect_{file_name}"), rect)

            input_box = np.array([leftr, leftc, rightr, rightc])
            self.mask_predictor.set_image(before)
            masks, scores, logits = self.mask_predictor.predict(
                box=input_box[None, :],
                multimask_output=False
            )
            mask_data = masks[0]*255
            cv2.imwrite(str(results_dir/f"real_mask_{file_name}"), mask_data)
        return sum(diff_inv.flatten()), sum(bw.flatten()), score, sum(mask_data)


if __name__ == "__main__":
    detector = AnomalyDetector("/mnt/d/reconstruct/Old/MVTec/", "/mnt/d/results/sam_pro/", "mvtec")
    detector.run_all_strength()
