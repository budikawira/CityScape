import numpy as np

def calculate_dice_coefficient(ground_truth, predicted):
    gt_np = ground_truth.cpu().numpy()
    pred_np = predicted.detach().cpu().numpy()
    intersection = np.logical_and(gt_np, pred_np)
    dice_coefficient = (2. * np.sum(intersection)) / (np.sum(gt_np) + np.sum(pred_np))

    # intersection = np.logical_and(ground_truth, predicted)
    # dice_coefficient = (2 * np.sum(intersection)) / (np.sum(ground_truth) + np.sum(predicted))
    return dice_coefficient

def calculate_dice_coefficients(ground_truths, predictions):
    num_samples = len(ground_truths)
    dice_coefficients = np.zeros(num_samples)
    for i in range(num_samples):
        dice_coefficients[i] = calculate_dice_coefficient(ground_truths[i], predictions[i])
    return dice_coefficients

def calculate_iou_score(ground_truth, predicted):
    gt_np = ground_truth.cpu().numpy()
    pred_np = predicted.cpu().numpy()
    
    intersection = np.logical_and(gt_np, pred_np)
    union = np.logical_or(gt_np, pred_np)
    
    iou_score = np.sum(intersection) / np.sum(union)
    
    # intersection = np.logical_and(ground_truth, predicted)
    # union = np.logical_or(ground_truth, predicted)
    # iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculate_iou_scores(ground_truths, predictions):
    num_samples = len(ground_truths)
    iou_scores = np.zeros(num_samples)
    for i in range(num_samples):
        iou_scores[i] = calculate_iou_score(ground_truths[i], predictions[i])
    return iou_scores