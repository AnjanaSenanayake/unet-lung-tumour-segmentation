import cv2
import numpy as np
import scipy.ndimage as ndimage
import sys
import torch
from torch import Tensor
import torch.nn.functional as F

def focal_loss(predictions: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2, reduction: str = "none") -> torch.Tensor:
    ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    p_t = predictions * targets + (1 - predictions) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

def dice_score(predictions: Tensor, targets: Tensor, smooth: float = 1e-6):
    assert predictions.size() == targets.size()
    predictions = (predictions > 0.5).float()

    dice = (2 * (predictions * targets).sum()) / (predictions.sum() + targets.sum() + smooth)
    return round(dice.item(), 4)

# Actual bounding box from the segmentation mask
def bbox(segmentation_masks):
    batch_size = segmentation_masks.size(0)
    bounding_boxes = torch.zeros(batch_size, 4)

    for i in range(batch_size):
        non_zero_coords = torch.nonzero(segmentation_masks[i], as_tuple=False)
        
        if non_zero_coords.size(0) > 0:
            y_min, x_min = torch.min(non_zero_coords, dim=0)[0]
            y_max, x_max = torch.max(non_zero_coords, dim=0)[0]
            
            bounding_boxes[i, 0] = x_min
            bounding_boxes[i, 1] = y_min
            bounding_boxes[i, 2] = x_max
            bounding_boxes[i, 3] = y_max
        else:
            bounding_boxes[i] = torch.tensor([0, 0, 0, 0])
    return bounding_boxes

# Calculate the target and search crop sizes
def crop_sizes(segmentation_mask):
    # Get the indices of non-zero pixels
    non_zero_indices = np.argwhere(segmentation_mask > 0)
    
    if non_zero_indices.size == 0:
        raise ValueError("No non-zero pixels found in the segmentation mask.")
    
    # Find the minimum and maximum y, x coordinates
    y_min, x_min = np.min(non_zero_indices, axis=0)
    y_max, x_max = np.max(non_zero_indices, axis=0)
    
    # Calculate height and width of the bounding box
    h = y_max - y_min
    w = x_max - x_min

    # Calculate padding and sizes
    p = (h + w) / 2
    template_size = np.sqrt((h + p) * (w + p))
    search_size = 2 * template_size
    
    return int(template_size), int(search_size)

# Calculate the center coordinates of the mask
def center_of_mask(mask):
    non_zero_coords = np.argwhere(mask > 0)
    
    if non_zero_coords.shape[0] == 0:
        raise ValueError("No object found in the mask.")
    
    # Separate x and y coordinates
    y_coords = non_zero_coords[:, 0]
    x_coords = non_zero_coords[:, 1]
    
    # Calculate the mean of x and y coordinates
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)
    
    return center_x, center_y

# Crop the image based on the given center and crop size
def crop(image, mask, crop_size):
    H, W = image.shape
    x_center, y_center = center_of_mask(mask)
    
    x_min = max(0, int(x_center - crop_size//2))
    x_max = min(W, int(x_center + crop_size//2))
    y_min = max(0, int(y_center - crop_size//2))
    y_max = min(H, int(y_center + crop_size//2))
    
    # Crop the image (1, H, W) -> (1, crop_size, crop_size)
    return image[y_min:y_max, x_min:x_max]

def euclidean_distance(coord1, coord2):
    (x1, y1) = coord1
    (x2, y2) = coord2
    return round(np.sqrt((x2 - x1)**2 + (y2 - y1)**2).item(), 4)

def center_of_mass(mask):
    non_zero_indices = torch.nonzero(mask.float())
    
    if non_zero_indices.shape[0] == 0:
        return (0,0)

    center_of_mass = torch.mean(non_zero_indices.float(), dim=0)
    return round(center_of_mass[0].item(), 3), round(center_of_mass[1].item(), 3)

# Calculate batch CoM and BBox center errors
def center_of_mass_and_bbox_center_error(pred_masks, true_masks, pixel_size=0.8):
    total_error_com = 0
    total_error_bbox_center = 0
    pred_bboxes = bbox(pred_masks)
    true_bboxes = bbox(true_masks)
    for true_mask, true_bbox, pred_mask, pred_bbox in zip(true_masks, true_bboxes, pred_masks, pred_bboxes):
        true_CoM = center_of_mass(true_mask)
        pred_CoM = center_of_mass(pred_mask)
        error_CoM = euclidean_distance(pred_CoM, true_CoM)
        total_error_com += error_CoM

        true_bbox_center = bbox_center(true_bbox)
        pred_bbox_center = bbox_center(pred_bbox)
        error_bbox_center = euclidean_distance(pred_bbox_center, true_bbox_center)
        total_error_bbox_center += error_bbox_center

    avg_com_error = total_error_com/pred_masks.shape[0]
    avg_bbox_center_error = total_error_bbox_center/pred_masks.shape[0]
    return avg_com_error*pixel_size, avg_bbox_center_error*pixel_size

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return round((x1.item()+x2.item())/2, 4), round((y1.item()+y2.item())/2,4)

class DualWriter:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        # Write to the console
        self.stdout.write(data)
        # Write to the file
        self.file.write(data)
        # Ensure output is flushed immediately
        self.file.flush()

    def flush(self):
        # Flush both the file and the console
        self.stdout.flush()
        self.file.flush()
