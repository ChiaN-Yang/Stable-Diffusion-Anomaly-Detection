import os
import numpy as np
import torch
import pandas as pd
from torchmetrics import PrecisionRecallCurve, AUROC


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


def run(folder_path, folder_type, data_type):
    return get_score(os.path.join(folder_path, folder_type, 'output.csv'), data_type)


def get_score_list(folder_path, data_type='thresh_score'):
    thresh_list = []
    folder_types = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    for folder_type in folder_types:
        thresh = run(folder_path, folder_type, data_type)
        thresh_list.append(thresh)
    return thresh_list


if __name__ == "__main__":
    STRENGTH = 0.5
    scores = []
    scores.append(
            get_score_list(f'/mnt/d/results/strength_{STRENGTH}_thresh_77', 'anomaly_score'))
    for thresh in [77, 100, 127, 147, 177, 200]:
        thresh_list = get_score_list(f'/mnt/d/results/strength_{STRENGTH}_thresh_{thresh}')
        scores.append(thresh_list)
    pd.DataFrame(scores).to_csv(f'/mnt/d/results/strength_{STRENGTH}_scores.csv')
