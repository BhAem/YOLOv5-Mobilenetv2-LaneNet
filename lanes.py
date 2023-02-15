import json
import os
import numpy as np 
import cv2
from models.LaneNet import LaneNet
from clustering import lane_cluster
import torch
# from counting import start_counting

def _load_model(model_path):
    model = LaneNet()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    return model

def _frame_process(image_frame, model, image_size, threshold):
    image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)
    img = image.copy()

    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, :, :, :]
    image = image / 255
    image = torch.tensor(image, dtype=torch.float)
    segmentation, embeddings = model(image.cuda())

    binary_mask = segmentation.data.cpu().numpy()
    binary_mask = binary_mask.squeeze()

    exp_mask = np.exp(binary_mask - np.max(binary_mask, axis=0))
    binary_mask = exp_mask / exp_mask.sum(axis=0)
    threshold_mask = binary_mask[1, :, :] > threshold
    threshold_mask = threshold_mask.astype(np.uint8)
    threshold_mask = threshold_mask * 255
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(4, 4))
    threshold_mask = cv2.dilate(threshold_mask, kernel, iterations=1)
    mask = cv2.connectedComponentsWithStats(threshold_mask, connectivity=8, ltype=cv2.CV_32S)
    output_mask = np.zeros(threshold_mask.shape, dtype=np.uint8)
    for label in np.unique(mask[1]):
        if label == 0:
            continue
        labelMask = np.zeros(threshold_mask.shape, dtype="uint8")
        labelMask[mask[1] == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > 400:
            output_mask = cv2.add(output_mask, labelMask)
    output_mask = output_mask.astype(np.float64) / 255
    return embeddings, output_mask, img
