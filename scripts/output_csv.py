import os
import numpy as np
import torch
import pandas as pd
import yaml
from torchmetrics import PrecisionRecallCurve, AUROC
from typing import List


def main(strength: float):
    with open("data/visa.yaml", "r") as stream:
        parser = yaml.load(stream, Loader=yaml.CLoader)
        categorys = parser["categorys"]
    
    scores = []
    one_folder_path = f'/mnt/d/results/strength_{strength}_thresh_77'
    scores.append(get_score_list(one_folder_path, 'anomaly_score', categorys))
    for thresh in [77, 100, 127, 147, 177, 200]:
        thresh_list = get_score_list(
                f'/mnt/d/results/strength_{strength}_thresh_{thresh}', 'thresh_score', categorys)
        scores.append(thresh_list)
    scores.append(get_score_list(one_folder_path, 'ssim_score', categorys))
    pd.DataFrame(scores).to_csv(f'/mnt/d/results/strength_{strength}_scores.csv', header=categorys)


def get_score_list(folder_path: str, data_type: str, categorys: List[str]):
    thresh_list = []
    for folder_type in categorys:
        thresh = get_score(os.path.join(folder_path, folder_type, 'output.csv'), data_type)
        thresh_list.append(thresh)
    return thresh_list


def get_score(csv_path, data_type):
    anomaly_areas, normal, abnormal, true_labels = read_csv(csv_path, data_type)

    anomaly_max = np.max(anomaly_areas)
    anomaly_min = np.min(anomaly_areas)
    anomaly_areas = (anomaly_areas - anomaly_min) / (anomaly_max - anomaly_min)

    auroc = AUROC(task="binary")
    auroc_score = auroc(torch.tensor(anomaly_areas), torch.tensor(true_labels))

    pr_curve = PrecisionRecallCurve(task="binary")
    precision, recall, thresholds = pr_curve(torch.tensor(anomaly_areas).float(), torch.tensor(true_labels))

    f1_score = (2 * precision * recall) / (precision + recall)
    threshold = thresholds[torch.argmax(f1_score)]
    threshold = threshold*(anomaly_max - anomaly_min) + anomaly_min
    auroc_score = round(auroc_score.item(), 4)*100
    return f"{auroc_score:.2f}"


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
    main(strength=0.1)
