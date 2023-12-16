"""Detect anomaly using SSIM."""
import os
import csv
import cv2
import numpy as np
import yaml
from skimage.metrics import structural_similarity
from pathlib import Path


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

    @staticmethod
    def _read_data_setting(dataset):
        with open(f"data/{dataset}.yaml", "r") as stream:
            parser = yaml.load(stream, Loader=yaml.CLoader)
        return parser, parser["dataset"]["mask_enable"]

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
        diff_inv = 255 - diff
        cv2.imwrite(str(results_dir/f"diff_{file_name}"), diff_inv)

        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, self.lower_thresh, 255, cv2.THRESH_TOZERO_INV)[1]
        cv2.imwrite(str(results_dir/f"thresh_{file_name}"), thresh)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        mask = np.zeros((512, 512), dtype='uint8')

        mask_area = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > self.AREA_THRESH:
                cv2.drawContours(mask, [c], 0, (255,255,255), -1)
                mask_area.append(area)

        cv2.imwrite(str(results_dir/f"mask_{file_name}"), mask)
        return sum(diff_inv.flatten()), sum(thresh.flatten()), score, sum(mask_area)


if __name__ == "__main__":
    detector = AnomalyDetector("/mnt/d/reconstruct", "/mnt/d/results/", "visa")
    detector.run_all_strength()
