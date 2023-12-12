import os
import numpy as np
import torch
import pandas as pd
import yaml
from torchmetrics import PrecisionRecallCurve, AUROC
from typing import List


class ResultsToCSV:
    def __init__(self, result_path, strength: float, dataset: str, metric: str):
        self.result_path = result_path
        self.strength = strength
        self.dataset = dataset
        self.metric = metric
        self.results_to_csv()

    def results_to_csv(self):
        with open(f"data/{self.dataset}.yaml", "r") as stream:
            parser = yaml.load(stream, Loader=yaml.CLoader)
            categorys = parser["categorys"]
        
        scores = []
        one_folder_path = f'{self.result_path}/strength_{self.strength}_thresh_77'
        scores.append(self.get_score_list(one_folder_path, 'anomaly_score', categorys))
        for thresh in [77, 100, 127, 147, 177, 200]:
            thresh_list = self.get_score_list(
                    f'{self.result_path}/strength_{self.strength}_thresh_{thresh}', 'thresh_score', categorys)
            scores.append(thresh_list)
        scores.append(self.get_score_list(one_folder_path, 'ssim_score', categorys))
        pd.DataFrame(scores).to_csv(f'{self.result_path}/strength_{self.strength}_{self.metric}_scores.csv', header=categorys)

    def get_score_list(self, folder_path: str, data_type: str, categorys: List[str]):
        thresh_list = []
        for folder_type in categorys:
            thresh = self.get_score(os.path.join(folder_path, folder_type, 'output.csv'), data_type)
            thresh_list.append(thresh)
        return thresh_list

    def get_score(self, csv_path, data_type):
        anomaly_areas, normal, abnormal, true_labels = self.read_csv(csv_path, data_type)

        anomaly_max = np.max(anomaly_areas)
        anomaly_min = np.min(anomaly_areas)
        anomaly_areas = (anomaly_areas - anomaly_min) / (anomaly_max - anomaly_min)

        if self.metric == 'auroc':
            auroc = AUROC(task="binary")
            score = auroc(torch.tensor(anomaly_areas), torch.tensor(true_labels))
        elif self.metric == 'f1':
            pr_curve = PrecisionRecallCurve(task="binary")
            precision, recall, thresholds = pr_curve(torch.tensor(anomaly_areas).float(), torch.tensor(true_labels))
            f1_score = (2 * precision * recall) / (precision + recall)
            score = torch.max(f1_score)
        elif self.metric == 'aupr':
            pr_curve = PrecisionRecallCurve(task="binary")
            pr_curve.update(torch.tensor(anomaly_areas).float(), torch.tensor(true_labels))
            score = pr_curve.compute()
        else:
            raise ValueError(f'Unknown metric: {self.metric}')
        score = round(score.item(), 4)*100
        return f"{score:.2f}"

    @staticmethod
    def read_csv(csv_path:str, data_type: str):
        df = pd.read_csv(csv_path)
        anomaly_areas = []
        normal = []
        abnormal = []
        true_labels = []
        for index, row in df.iterrows():
            if row['ground_truth'] == 0:
                normal.append(row[data_type])
            else:
                abnormal.append(row[data_type])
            anomaly_areas.append(row[data_type])
            true_labels.append(row['ground_truth'])

        # convert list to array
        anomaly_areas = np.array(anomaly_areas)
        normal = np.array(normal)
        abnormal = np.array(abnormal)
        return anomaly_areas, normal, abnormal, true_labels


if __name__ == '__main__':
    for strength in [0.1, 0.2, 0.5, 0.7]:
        ResultsToCSV(result_path='/mnt/d/results/visa/visa_test_without_mask', strength=strength, dataset='visa', metric='f1')
