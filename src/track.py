# 
# This file is part of the share-cv distribution (https://github.com/CheneyYin/share-cv).
# Copyright (c) 2024 Chengyu Yan.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

#
# Demo Video: https://www.bilibili.com/video/BV1Tgv8eGEK3/
#

from collections import defaultdict

from ultralytics import YOLO
import cv2
import numpy as np
from resources import get_model_path, get_sample_path

model_path = get_model_path('yolov8n.pt')
sample_path = get_sample_path('chengdu_traffic.mp4')

model = YOLO(model_path)
track_history = defaultdict(lambda: [])

cap = cv2.VideoCapture(sample_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break;
    [h, w, _] = frame.shape
    frame = cv2.resize(frame, (w // 2, h // 2))
    [result] = model.track(frame, persist=True)

    im = result.plot(font_size=9, line_width=1)  
    cv2.putText(
        im, 
        "FPS: {:.2f} | Speed: {:.2f}ms".format(1000 / result.speed['inference'], result.speed['inference']), 
        (20, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
 
    boxes = result.boxes.xywh.cpu()
    track_ids = [] if result.boxes.id is None else result.boxes.id.int().cpu().tolist()

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 60: 
            track.pop(0)
        
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(im, [points], isClosed=False, color=(255, 0, 0), thickness=2)
    
    cv2.imshow('Track', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()