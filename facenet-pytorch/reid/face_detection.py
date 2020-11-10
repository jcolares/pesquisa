from facenet_pytorch import MTCNN, extract_face
import torch
import numpy as np
import mmcv
import cv2
from PIL import Image, ImageDraw
from IPython import display
import glob
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

frames = []
#files = glob.glob("/home/jeff/datasets/TUM Gait/data_person1+2/image/p001/b01/*")
files = glob.glob("reid/b01/*")
for myFile in files:
    fileName = os.path.splitext(os.path.basename(myFile))[0]

    img = Image.open(myFile)

    boxes, probs, points = mtcnn.detect(img, landmarks=True)
    if boxes is not None:
        # Draw boxes and save faces
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        for i, (box, point) in enumerate(zip(boxes, points)):
            draw.rectangle(box.tolist(), width=5)
            for p in point:
                #draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
                draw.point(p)
            extract_face(
                img, box, save_path='reid/output/detected_face_{}_{}.png'.format(fileName, i))
        img_draw.save('reid/output/annotated_faces_{}.png'.format(fileName))
