import argparse
from datetime import datetime
import json
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.ndimage as ndimage
import shutil
import sys
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(".."))
from base.Config import Config
from base.DRRDataset import DRRDataset
from base.UNet import UNet
from base.Utils import *

DATASET_META_FILENAME = "dataset.meta"

# Plotting the losses
def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path +'/loss_plot.png')

# Save model checkpoint
def save_checkpoint(epoch, model, checkpoint_path, checkpoint_preds_path, search_images, true_masks, pred_masks):
    os.makedirs(checkpoint_preds_path)
    state_dict = model.state_dict()
    torch.save(state_dict, f"{checkpoint_path}/model_epoch{epoch}.pth")
    pred_bboxes = bbox(pred_masks)
    true_bboxes = bbox(true_masks)
    for i, (search_image, true_mask, true_bbox, pred_mask, pred_bbox) in enumerate(zip(search_images, true_masks, true_bboxes, pred_masks, pred_bboxes)):
        x1_t, y1_t, x2_t, y2_t = true_bbox
        x1, y1, x2, y2 = pred_bbox

        true_CoM = center_of_mass(true_mask)
        pred_CoM = center_of_mass(pred_mask)
        CoM_error = euclidean_distance(pred_CoM, true_CoM)

        search_image = search_image.cpu().numpy()
        true_mask = true_mask.cpu().numpy()
        pred_mask = pred_mask.cpu().numpy()

        true_bbox_center = bbox_center(true_bbox)
        pred_bbox_center = bbox_center(pred_bbox)
        bbox_center_error = euclidean_distance(true_bbox_center, pred_bbox_center)

        _, axes = plt.subplots(1, 2, figsize=(10, 8))
        axes[0].set_title("Actual")
        axes[0].imshow(search_image, cmap='gray', interpolation='none')
        axes[0].imshow(np.ma.masked_where(true_mask == 0, true_mask), cmap=cm.jet, alpha=0.5)
        axes[0].add_patch(patches.Rectangle((x1_t, y1_t), x2_t-x1_t, y2_t-y1_t, linewidth=2, edgecolor='green',fill=False))
        axes[0].scatter(true_CoM[1], true_CoM[0], color='green')
        axes[0].set_xlabel('True CoM: ' + str(true_CoM) + '\nTrue BBox Center: ' + str(true_bbox_center))
        # axes[0].axis("off")

        axes[1].set_title("Predicted")
        axes[1].imshow(search_image, cmap='gray', interpolation='none')
        axes[1].imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap=cm.jet, alpha=0.5)
        axes[1].add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', fill=False))
        axes[1].scatter(pred_CoM[1], pred_CoM[0], color='red')
        axes[1].set_xlabel('Pred CoM: ' + str(pred_CoM) + '\nPred BBox Center: ' + str(pred_bbox_center) + '\nError CoM: ' + str(CoM_error) + '\nError BBox Center: ' + str(bbox_center_error))
        # axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{checkpoint_preds_path}/{i}.png")
        plt.close()
    print(f'Checkpoint epoch {epoch} saved!') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trainer for Siamese Network Tumour Tracker')
    parser.add_argument('--dataset_meta', '-meta', type=str, default=DATASET_META_FILENAME)
    parser.add_argument('--dataset_dir', '-d', type=str, default="")
    parser.add_argument('--gpu_id', '-gpu', type=str, default="5")
    args = parser.parse_args()

    print("Reading training hyperparameters...")
    configs = Config(os.path.abspath("training/config.json"))

    cuda = f'cuda:{args.gpu_id}'
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    print(f"{device} will be used for training...")

    # Create checkpoints directory
    print("Creating checkpoint directory...")
    checkpoints_path = os.path.abspath('checkpoints/')
    os.makedirs(checkpoints_path, exist_ok=True)
    checkpoint_version_path = os.path.abspath(f'{checkpoints_path}/train_{os.path.basename(os.path.normpath(args.dataset_dir))}')
    os.makedirs(checkpoint_version_path, exist_ok=True)
    checkpoint_version_logs_path = os.path.abspath(checkpoint_version_path + '/logs')
    os.makedirs(checkpoint_version_logs_path, exist_ok=True)
    dual_writer = DualWriter(open(checkpoint_version_logs_path + '/train.log', 'w'))
    sys.stdout = dual_writer

    # Loading the datasets
    print("Loading the dataset...")
    print("Dataset: ", args.dataset_dir)
    dataset_metadata_path = os.path.join(args.dataset_dir, args.dataset_meta)
    metadata = pd.read_csv(dataset_metadata_path)
    if configs.train_size != "None":
        metadata = metadata[0:int(configs.train_size)]
    train_len = int(len(metadata) - len(metadata) * configs.val_ratio)
    train_metadata = metadata[0:train_len]
    validation_metadata = metadata[train_len:]
    train_size = len(train_metadata)
    validation_size = len(validation_metadata)

    train_set = DRRDataset(args.dataset_dir, train_metadata, configs)
    validation_set = DRRDataset(args.dataset_dir, validation_metadata, configs)

    train_loader = DataLoader(train_set, batch_size=configs.batch_size, num_workers=configs.train_num_workers, pin_memory=True, sampler=None)
    val_loader = DataLoader(validation_set, batch_size=configs.batch_size, num_workers=configs.val_num_workers, pin_memory=True, sampler=None)
    print(f'Train Set Size = {train_size}')
    print(f'Training batch size = {configs.batch_size}')
    print(f'# of training batches = {len(train_loader)}')
    print(f'Validation Set Size = {validation_size}')
    print(f'Validation batch size = {configs.batch_size}')
    print(f'# of validation batches = {len(val_loader)}')

    # Initializing the model
    print("Initializing the model...")
    model = UNet(in_channels=3, num_classes=1)
    model = model.to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=configs.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, configs.lr_schedular_mode, factor=configs.lr_schedular_factor, patience = configs.lr_schedular_patience, min_lr=configs.lr_min)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=configs.gradient_scaler)

    # Training
    print("Training...")
    train_writer = SummaryWriter(f"{checkpoint_version_logs_path}/train")
    validation_writer = SummaryWriter(f"{checkpoint_version_logs_path}/validation")
    num_decimals = 7
    train_loss = None
    train_losses = []
    val_loss = None
    val_losses = []
    best_loss = float('inf')
    best_dice = 0
    best_loss_dice = float('inf')
    best_loss_dice_epoch = 0
    for epoch in range(1, configs.num_epochs + 1):
        model.train()
        running_train_loss = 0
        running_train_dice = 0
        # correct_pixels = 0
        # total_pixels = 0

        for param_group in optimizer.param_groups:
            print(f"LR at epoch {epoch}: {param_group['lr']}")

        with tqdm(total=train_size, desc=f'Epoch {epoch}/{configs.num_epochs}', unit='img') as pbar:
            for train_batch_idx, batch in enumerate(train_loader):
                images = batch['image']
                # search_images = batch['search_image']
                search_masks = batch['search_mask']
                # template_images = batch['template_image']
                # template_masks = batch['template_mask']

                assert images[0].shape[1] != model.in_channels, \
                    f'Network has been defined with {model.in_channels} input channels, ' \
                    f'but loaded images have {images[0].shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                # search_images = search_images.to(device=device, dtype=torch.float32)
                search_masks = search_masks.to(device=device, dtype=torch.float32)
                # template_images = template_images.to(device=device, dtype=torch.float32)
                # template_masks = template_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=configs.autocast):
                    masks = model(images)
                    loss = focal_loss(masks.squeeze(1), search_masks, reduction="mean")
                    dice = dice_score(masks.squeeze(1), search_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                running_train_loss += loss.item()
                running_train_dice += dice
                # binary_masks = (masks.squeeze(1) > 0.5).float()
                # correct_pixels += (binary_masks == search_masks).sum()
                # total_pixels += torch.numel(binary_masks)
                # running_train_acc = correct_pixels/total_pixels
                pbar.update(images.shape[0])
                pbar.set_postfix(loss = float(loss.item()), dice_score = float(dice))
            
        epoch_train_loss = float(running_train_loss/len(train_loader))
        train_losses.append(epoch_train_loss)
        epoch_train_dice = float(running_train_dice/len(train_loader))
        train_writer.add_scalar("Loss", epoch_train_loss, epoch)
        train_writer.add_scalar("Dice", epoch_train_dice, epoch)

        # Epoch Validation
        running_val_loss = 0.0
        running_val_dice = 0.0
        running_val_com_error = 0.0
        running_val_bbox_center_error = 0.0
        model.eval()
        with torch.no_grad():
            with tqdm(total=validation_size, desc=f'Epoch {epoch} Validation', unit='img', leave=False) as val_pbar:
                for val_batch_idx, batch in enumerate(val_loader):
                    images = batch['image']
                    search_images = batch['search_image']
                    search_masks = batch['search_mask']
                    # template_images = batch['template_image']
                    # template_masks = batch['template_mask']

                    images = images.to(device=device, dtype=torch.float32)
                    search_images = search_images.to(device=device, dtype=torch.float32)
                    search_masks = search_masks.to(device=device, dtype=torch.float32)
                    # template_images = template_images.to(device=device, dtype=torch.float32)
                    # template_masks = template_masks.to(device=device, dtype=torch.float32)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=configs.autocast):
                        masks = model(images)
                        loss = focal_loss(masks.squeeze(1), search_masks, reduction="mean")
                        dice = dice_score(masks.squeeze(1), search_masks)

                    running_val_loss += loss.item()
                    running_val_dice += dice
                    binary_masks = (masks.squeeze(1) > 0.5).float()
                    # correct_pixels += (binary_masks == search_masks).sum()
                    # total_pixels += torch.numel(binary_masks)
                    # running_val_acc = correct_pixels/total_pixels
                    error_CoM, error_bbox_center = center_of_mass_and_bbox_center_error(binary_masks, search_masks)
                    running_val_com_error += error_CoM
                    running_val_bbox_center_error += error_bbox_center
                    val_pbar.update(images.shape[0])
                    val_pbar.set_postfix(val_loss = float(loss.item()), val_dice_score = float(dice))
        
        epoch_val_loss = float(running_val_loss/len(val_loader))
        epoch_val_dice = float(running_val_dice/len(val_loader))
        epoch_val_loss_dice = epoch_val_loss * (1-epoch_val_dice)
        # epoch_val_accuracy = float(running_val_acc*100)
        epoch_val_CoM_error = float(running_val_com_error/len(val_loader))
        epoch_val_bbox_center_error = float(running_val_bbox_center_error/len(val_loader))
        scheduler.step(epoch_val_loss)
        val_losses.append(epoch_val_loss)
        validation_writer.add_scalar("Loss", epoch_val_loss, epoch)
        validation_writer.add_scalar("Dice", epoch_val_dice, epoch)
        # validation_writer.add_scalar("Accuracy", epoch_val_accuracy, epoch)
        validation_writer.add_scalar("CoM Error", epoch_val_CoM_error, epoch)
        validation_writer.add_scalar("Geometric Center Error", epoch_val_bbox_center_error, epoch)
    
        print(f'train_loss: {round(epoch_train_loss, num_decimals)} val_loss: {round(epoch_val_loss, num_decimals)}')
        print(f'train_dice: {round(epoch_train_dice, num_decimals)} val_dice: {round(epoch_val_dice, num_decimals)}')


        if (epoch%10==0):
            preds_path = f"{checkpoint_version_path}/preds_epoch{epoch}"
            save_checkpoint(epoch, model, checkpoint_version_path, preds_path, search_images[0:10], search_masks[0:10], binary_masks[0:10])
        elif (epoch_val_loss_dice < best_loss_dice):
            prev_best_dice_preds_path = os.path.abspath(f"{checkpoint_version_path}/best_dice_epoch{best_loss_dice_epoch}")
            curr_best_dice_preds_path = os.path.abspath(f"{checkpoint_version_path}/best_dice_epoch{epoch}")
            if os.path.exists(prev_best_dice_preds_path):
                shutil.rmtree(prev_best_dice_preds_path)
                os.remove(f"{checkpoint_version_path}/model_epoch{best_loss_dice_epoch}.pth")
            save_checkpoint(epoch, model, checkpoint_version_path, curr_best_dice_preds_path, search_images[0:10], search_masks[0:10], binary_masks[0:10])
            best_dice = epoch_val_dice
            best_loss_dice = epoch_val_loss_dice
            best_loss_dice_epoch = epoch
    
    plot_losses(train_losses, val_losses, checkpoint_version_path)
    train_writer.flush()
    validation_writer.close()

    # Write the configuration object to a JSON file
    with open(f"{checkpoint_version_path}/config.json", 'w') as file:
        json.dump(configs.toJSON(), file, indent=4)

    print(f"Best Dice: {best_dice} at epoch {best_loss_dice_epoch}!")