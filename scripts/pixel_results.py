import os
import cv2
import numpy as np
import pandas as pd
import yaml
import glob
from sklearn.metrics import roc_auc_score, f1_score
from typing import List


class PixelResultsToCSV:
    def __init__(self, result_path, strength: float, dataset: str):
        self.result_path = result_path
        self.strength = strength
        self.dataset = dataset
        self.results_to_csv()

    def results_to_csv(self):
        with open(f"data/{self.dataset}.yaml", "r") as stream:
            parser = yaml.load(stream, Loader=yaml.CLoader)
            categorys = parser["categorys"]

        auroc_scores = []
        f1_scores = []
        for thresh in [77, 100, 127, 147, 177, 200]:
            thresh_auroc, thresh_f1 = self.get_score_list(
                    f'{self.result_path}/strength_{self.strength}_thresh_{thresh}', categorys)
            auroc_scores.append(thresh_auroc)
            f1_scores.append(thresh_f1)
        pd.DataFrame(auroc_scores).to_csv(f'{self.result_path}/strength_{self.strength}_auroc_pixel.csv', header=categorys)
        pd.DataFrame(f1_scores).to_csv(f'{self.result_path}/strength_{self.strength}_f1_pixel.csv', header=categorys)

    def get_score_list(self, folder_path: str, categorys: List[str]):
        thresh_auroc = []
        thresh_f1 = []
        for category in categorys:
            auroc, f1= self.read_all_images_f1_score(os.path.join(folder_path, category, 'bad'), category)
            thresh_auroc.append(round(auroc*100, 2))
            thresh_f1.append(round(f1*100, 2))
        return thresh_auroc, thresh_f1

    def read_all_images_f1_score(self, folder_path, category):
        f1_scores = []
        aurocs = []
        gt_folder = f"/mnt/d/dataset/VisA_20220922/visa_pytorch/{category}/ground_truth/bad/*.png"
        for gt_path, file_path in zip(glob.glob(gt_folder), glob.glob(f"{folder_path}/mask_*")):
            auroc, f1= self.calculate_pixel_auc(gt_path, file_path)
            f1_scores.append(f1)
            print(f"Category: {category}, AUROC: {auroc}, F1 Score: {f1}")
            aurocs.append(auroc)
        return np.mean(aurocs), np.mean(f1_scores)

    @staticmethod
    def calculate_pixel_auc(y_true, y_pred):
        print('GT', y_true, '\nMask', y_pred)
        y_true = cv2.imread(str(y_true))
        y_pred = cv2.imread(str(y_pred))
        y_pred = cv2.resize(y_pred,  y_true.shape[:2][::-1])
        # Flatten the ground truth and predicted anomaly scores
        y_true_flat = y_true.flatten()
        y_true_flat[y_true_flat>0] = 1
        y_pred_flat = y_pred.flatten()
        y_pred_flat[y_pred_flat>0] = 1

        # Calculate the AUROC score
        auroc = roc_auc_score(y_true_flat, y_pred_flat)
        # Calculate the F1 score
        f1 = f1_score(y_true_flat, y_pred_flat)

        return auroc, f1


if __name__ == '__main__':
    for strength in [0.1, 0.2, 0.5, 0.7]:
        PixelResultsToCSV(result_path='/mnt/d/results/Old/visa/visa_test_without_mask/', strength=strength, dataset='visa')
