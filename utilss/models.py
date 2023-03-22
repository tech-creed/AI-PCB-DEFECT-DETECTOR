import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from efficientnet.tfkeras import EfficientNetB4
import cv2

import torch
import glob

def obj_detector_yolov7(img_path):
    model = torch.hub.load('yolov7', 'custom', 'yolov7/pcb_defect_detection/Detection_Ver3/weights/best.pt', source='local', force_reload=True)
    model.names = ["missing_hole", "mouse_bite", "open_circuit", "short", "spurious_copper", "spur"]
    img = cv2.imread(img_path) 
    results = model(img, size=608)
    results.save()
    result_pd = results.pandas().xyxy[0]
    boxes = [result_pd['xmin'].astype(np.int32).tolist(),result_pd['ymin'].astype(np.int32).tolist(),result_pd['xmax'].astype(np.int32).tolist(),result_pd['ymax'].astype(np.int32).tolist()]
    
    return result_pd['name'], boxes, img

def pcb_defect(batch):
    # print(f"dataset/{batch}/*.jpg")
    # board = glob.glob(f"../dataset/{batch}/*.jpg")
    board = [f'dataset/{batch}/1 (1).jpg', f'dataset/{batch}/1 (2).jpg', f'dataset/{batch}/1 (3).jpg', f'dataset/{batch}/1 (4).jpg', f'dataset/{batch}/1 (5).jpg',
             f'dataset/{batch}/1 (6).jpg', f'dataset/{batch}/1 (7).jpg', f'dataset/{batch}/1 (8).jpg', f'dataset/{batch}/1 (9).jpg', f'dataset/{batch}/1 (10).jpg']
    
    i = 1
    for images in board:
        plt.figure(figsize=(20,120))

        # YoloV7 Inference
        plt.subplot(10,2,1)
        names,boxes,sample = obj_detector_yolov7(images)  

        plt.axis('off')
        plt.imshow(sample)
        plt.savefig(f"static/uploads/{batch}/{i}.png", bbox_inches='tight', pad_inches=0)
        i+=1
