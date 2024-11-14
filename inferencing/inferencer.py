import argparse
from datetime import datetime
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pathlib
import sys
import torch
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(".."))
from base.UNet import UNet
from base.Utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inferencer for Siamese Network Tumour Tracker')
    parser.add_argument('--model', '-m', type=str, default="")
    parser.add_argument('--dataset', '-d', type=str, default="")
    parser.add_argument('--data_format', '-f', type=str, default="png")
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--gpu_id', '-gpu', type=str, default="5")
    args = parser.parse_args()

    print("Reading inferencing parameters...")
    # configs = Config(os.path.abspath("inferencing/config.json"))

    cuda = f'cuda:{args.gpu_id}'
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    print(f"{device} will be used for inferencing...")

    # Create predictions directory
    print("Creating predictions directory...")
    if args.output != None:
        prediction_version_path = args.output
    else:
        predictions_path = os.path.dirname(args.model)
        prediction_version_path = os.path.abspath(f'{predictions_path}/preds_{os.path.basename(os.path.normpath(args.dataset))}')
    os.makedirs(prediction_version_path, exist_ok=True)
    prediction_version_logs_path = os.path.abspath(prediction_version_path + '/logs')
    os.makedirs(prediction_version_logs_path, exist_ok=True)
    dual_writer = DualWriter(open(prediction_version_logs_path + '/predictions.log', 'w'))
    sys.stdout = dual_writer

    # Loading the model
    print("Loading the model...")
    model = UNet(in_channels=3, num_classes=1)
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Loading the dataset
    print("Loading the dataset...")
    path_map = {}
    path_list = [f for f in pathlib.Path(args.dataset).glob("*." + args.data_format)]
    for idx, path in enumerate(path_list):
        filename_with_ext = os.path.basename(path)
        filename, _ = os.path.splitext(filename_with_ext)
        path_map[filename] = path
    print(f'Test Set Size = {len(path_map)}')

    # Inferencing
    print("Inferencing...")
    sorted_path_map = {key: path_map[key] for key in sorted(path_map.keys())}
    frame_list = list(sorted_path_map.items())
    dice_scores = 0
    com_errors = 0
    bbox_center_errors = 0
    dice_list = []
    dice_list2 = []
    dice_list3 = []
    dice_list4 = []
    is_initial = True
    with torch.no_grad():
        for template_image_path, search_image_path in zip(frame_list, frame_list[1:] + [None]):
            if search_image_path != None:
                template_image = cv2.imread(os.path.normpath(template_image_path[1]), cv2.IMREAD_GRAYSCALE)
                template_image_name = template_image_path[0] + '.' + args.data_format
                template_mask = cv2.imread(os.path.join(args.dataset, 'masks', template_image_name), cv2.IMREAD_GRAYSCALE)
                search_image = cv2.imread(os.path.normpath(search_image_path[1]), cv2.IMREAD_GRAYSCALE)
                search_image_name = search_image_path[0] + '.' + args.data_format
                search_mask = cv2.imread(os.path.join(args.dataset, 'masks', search_image_name), cv2.IMREAD_GRAYSCALE)

                # template_size, search_size = crop_sizes(crop_mask)
                # template_image = crop(template_image, crop_mask, template_size)
                # template_mask = crop(template_mask, crop_mask, template_size)
                # search_image = crop(search_image, crop_mask, search_size)
                # search_mask = crop(search_mask, crop_mask, search_size)

                template_image = cv2.resize(template_image, (256, 256), interpolation = cv2.INTER_LINEAR)
                template_mask = cv2.resize(template_mask, (256, 256), interpolation = cv2.INTER_LINEAR)
                search_image = cv2.resize(search_image, (256, 256), interpolation = cv2.INTER_LINEAR)
                search_mask = cv2.resize(search_mask, (256, 256), interpolation = cv2.INTER_LINEAR)

                template_image = cv2.normalize(template_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16F)
                template_mask[template_mask < 255.0] = 0.0
                template_mask[template_mask == 255.0] = 1.0
                template_mask = 1 - template_mask
                search_image = cv2.normalize(search_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16F)
                search_mask[search_mask < 255.0] = 0.0
                search_mask[search_mask == 255.0] = 1.0
                search_mask = 1 - search_mask

                if is_initial:
                    initial_mask = template_mask
                    previous_mask = template_mask
                    is_initial = False

                image = np.stack([search_image, previous_mask, template_image], axis=0)
                
                image = torch.from_numpy(image).to(device=device, dtype=torch.float32).unsqueeze(0)
                template_image_tensor = torch.from_numpy(template_image).to(device=device, dtype=torch.float32).unsqueeze(0)
                template_mask_tensor = torch.from_numpy(template_mask).to(device=device, dtype=torch.float32).unsqueeze(0)
                search_image_tensor = torch.from_numpy(search_image).to(device=device, dtype=torch.float32).unsqueeze(0)
                search_mask_tensor = torch.from_numpy(search_mask).to(device=device, dtype=torch.float32).unsqueeze(0)

                pred_mask = model(image)
                dice = dice_score(pred_mask.squeeze(1), search_mask_tensor)

                binary_mask_tensor = (pred_mask.squeeze(0).squeeze(0) > 0.1).float()
                binary_mask = binary_mask_tensor.cpu().numpy()
                previous_mask = cv2.resize(binary_mask, (256, 256), interpolation = cv2.INTER_LINEAR)

                image2 = np.stack([search_image, initial_mask, template_image], axis=0)
                image2 = torch.from_numpy(image2).to(device=device, dtype=torch.float32).unsqueeze(0)
                pred_mask2 = model(image2)
                dice2 = dice_score(pred_mask2.squeeze(1), search_mask_tensor)
                binary_mask_tensor2 = (pred_mask2.squeeze(0).squeeze(0) > 0.1).float()
                binary_mask2 = binary_mask_tensor2.cpu().numpy()

                com_true = center_of_mass(search_mask_tensor.squeeze(0))
                com_pred = center_of_mass(binary_mask_tensor)
                com_pred_f = center_of_mass(binary_mask_tensor2)
                com_error = euclidean_distance(com_pred, com_true)
                com_error_f = euclidean_distance(com_pred_f, com_true)
                
                true_bboxes = bbox(search_mask_tensor)
                pred_bboxes = bbox(binary_mask_tensor.unsqueeze(0))
                predf_bboxes = bbox(binary_mask_tensor2.unsqueeze(0))
                x1_t, y1_t, x2_t, y2_t = true_bboxes[0]
                x1, y1, x2, y2 = pred_bboxes[0]
                x1f, y1f, x2f, y2f = predf_bboxes[0]
                true_bbox_center = bbox_center(true_bboxes[0])
                pred_bbox_center = bbox_center(pred_bboxes[0])
                predf_bbox_center = bbox_center(predf_bboxes[0])
                bbox_center_error = euclidean_distance(pred_bbox_center, true_bbox_center)
                bbox_center_error_f = euclidean_distance(predf_bbox_center, true_bbox_center)
                
                _, axes = plt.subplots(1, 3, figsize=(15, 5))
                plt.subplots_adjust(bottom=0.2)
                axes[0].set_title("Actual")
                axes[0].imshow(search_image, cmap='gray', interpolation='none')
                axes[0].imshow(np.ma.masked_where(search_mask == 0, search_mask), cmap=cm.jet, alpha=0.5)
                axes[0].add_patch(patches.Rectangle((x1_t, y1_t), x2_t-x1_t, y2_t-y1_t, linewidth=2, edgecolor='green', fill=False))
                axes[0].scatter(com_true[1], com_true[0], color='green')
                # axes[0].axis("off")

                axes[1].set_title("Predicted - Moving Template")
                axes[1].imshow(search_image, cmap='gray', interpolation='none')
                axes[1].imshow(np.ma.masked_where(binary_mask == 0, binary_mask), cmap=cm.jet, alpha=0.5)
                axes[1].add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', fill=False))
                axes[1].scatter(com_pred[1], com_pred[0], color='red')
                # axes[1].axis("off")

                axes[2].set_title("Predicted - Fixed Template")
                axes[2].imshow(template_image, cmap='gray', interpolation='none')
                axes[2].imshow(np.ma.masked_where(binary_mask2 == 0, binary_mask2), cmap=cm.jet, alpha=0.5)
                axes[2].add_patch(patches.Rectangle((x1f, y1f), x2f-x1f, y2f-y1f, linewidth=2, edgecolor='blue', fill=False))
                axes[2].scatter(com_pred_f[1], com_pred_f[0], color='blue')
                # axes[1].axis("off")

                axes[0].set_xlabel('True CoM: ' + str(com_true) + '\nTrue BBox Center: ' + str(true_bbox_center))
                axes[1].set_xlabel('Pred CoM: ' + str(com_pred) + '\nPred BBox Center: ' + str(pred_bbox_center) + '\nError CoM: ' + str(com_error) + '\nError BBox Center: ' + str(bbox_center_error))
                axes[2].set_xlabel('Pred CoM: ' + str(com_pred_f) + '\nPred BBox Center: ' + str(predf_bbox_center) + '\nError CoM: ' + str(com_error_f) + '\nError BBox Center: ' + str(bbox_center_error_f))

                plt.savefig(f'{prediction_version_path}/{search_image_path[0]}.png')
                plt.close()
                print(f"Frame {search_image_path[0]} - DICE : {dice} CoM: {com_pred} CoM Error: {com_error} BBox Center: {pred_bbox_center} BBox Center Error: {bbox_center_error}")
                dice_scores += dice
                com_errors += com_error
                bbox_center_errors += bbox_center_error
                dice_list.append(dice*100)
                dice_list2.append(dice2*100)

        plt.title("DICE vs Frame")
        plt.plot(dice_list, marker=',', label='DICE with template mask as previous predicted mask')
        plt.plot(dice_list2, marker=',', color='red', label='DICE with fixed template mask')
        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("DICE")
        plt.savefig(f"{prediction_version_path}/#DICE_vs_Frame.png")

        print(f"Average DICE: {dice_scores/len(frame_list)}")
        print(f"Average CoM Error: {(com_errors/len(frame_list))*0.8}")
        print(f"Average Geometrical Error: {(bbox_center_errors/len(frame_list))*0.8}")
