"""Read results from csv file."""
import os
import numpy as np
import matplotlib.pyplot as plt
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
    # print('threshold: ', threshold)
    # print('f1 score: ', torch.max(f1_score))
    print('auroc score: ', auroc_score, '\n')

    # plt.hist(normal, bins=50, label='normal')
    # plt.hist(abnormal, bins=50, label='anomaly')
    # plt.axvline(threshold, color='r', linewidth=3, linestyle='dashed', label='{:0.3f}'.format(threshold))
    # plt.legend(loc='upper right')
    # plt.title("Normal and Anomaly Loss")
    # plt.show()


def run(folder_path, folder_type):
    print(f'Processing {folder_type}...')
    for data_type in ['anomaly_score', 'thresh_score', 'ssim_score', 'mask_area']:
        print(data_type)
        get_score(os.path.join(folder_path, folder_type, 'output.csv'), data_type)
    print('--------------------\n')

def main(folder_path):
    folder_types = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    for folder_type in folder_types:
        run(folder_path, folder_type)


if __name__ == "__main__":
    STRENGTH = 0.1
    THRESHOLD = 200
    main(f'/mnt/d/results/strength_{STRENGTH}_thresh_{THRESHOLD}')
    # run('/mnt/d/results/strength_0.1_thresh_127', 'cashew')
