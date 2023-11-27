"""Detect anomaly using SSIM."""
import os
import csv
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def detection(folder_path, file_name, results_dir, object_type, folder):
    results_dir = os.path.join(results_dir, object_type, folder)
    source_img_path = os.path.join(folder_path, f'source_{file_name}')
    inpaint_image_name = os.path.join(folder_path, f'inpaint_{file_name}')
    mask_image_name = os.path.join(folder_path, f'mask_{file_name}')
    # Compute SSIM
    # Load images
    before = cv2.imread(source_img_path)
    after = cv2.imread(inpaint_image_name)
    image_mask = cv2.imread(mask_image_name, 0)

    # reseze image
    before = cv2.resize(before, (512, 512))
    after = cv2.resize(after, (512, 512))
    image_mask = cv2.resize(image_mask, (512, 512))

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    before_gray = cv2.bitwise_and(before_gray, before_gray, mask=image_mask)
    after_gray = cv2.bitwise_and(after_gray, after_gray, mask=image_mask)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, gaussian_weights=True, full=True)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_inv = 255 - diff
    cv2.imwrite(os.path.join(results_dir, f"diff_{file_name}"), diff_inv)

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_TOZERO_INV)[1]
    cv2.imwrite(os.path.join(results_dir, f"thresh_{file_name}"), thresh)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    mask = np.zeros(image_mask.shape, dtype='uint8')

    mask_area = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 120:
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            mask_area.append(area)

    cv2.imwrite(os.path.join(results_dir, f"mask_{file_name}"), mask)
    return sum(diff_inv.flatten()), sum(thresh.flatten()), score, sum(mask_area)


def run(image_dir, results_dir, object_type):
    os.makedirs(f'{results_dir}/{object_type}/Anomaly', exist_ok=True)
    os.makedirs(f'{results_dir}/{object_type}/Normal', exist_ok=True)

    csv_path = f'{results_dir}/{object_type}/output.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'ground_truth', 'anomaly_score', 'thresh_score', 'ssim_score', 'mask_area'])

    for folder in ['Anomaly', 'Normal']:
        folder_path = os.path.join(image_dir, object_type, folder)
        for img_file in os.listdir(folder_path):
            file_type = img_file.split('_')[0]
            file_name = img_file.split('_')[1]
            if file_type == 'source':
                print(os.path.join(folder_path, img_file))
                true_label= 1 if folder == 'Anomaly' else 0
                anomaly_score, thresh_score, ssim_score, mask_area = detection(
                        folder_path, file_name, results_dir, object_type, folder)
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([img_file, true_label, anomaly_score, thresh_score, ssim_score, mask_area])
            else:
                continue


def main(image_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    object_types = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    for object_type in object_types:
        print(f'Processing {object_type}...')
        run(image_dir, results_dir, object_type)


if __name__ == "__main__":
    STRENGTH = 0.2
    for THRESHOLD in [77, 100, 127, 147, 177, 200]:
        main(f"/mnt/d/reconstruct/strength_{STRENGTH}", f"/mnt/d/results/strength_{STRENGTH}_thresh_{THRESHOLD}")
    # run("./outputs", "/mnt/d/results/strength_0.1_thresh_127", "cashew")
