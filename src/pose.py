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
# Demo Video: https://www.bilibili.com/video/BV1x3v8erE3h
#

from ultralytics import YOLO
import cv2
from resources import get_model_path, get_sample_path

model_path = get_model_path('yolov8n-pose.pt')
sample_path = get_sample_path('dance.mp4')
model = YOLO(model_path)
cap = cv2.VideoCapture(sample_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    [result] = model(frame)
    im = result.plot()

    cv2.putText(
        im, 
        "FPS: {:.2f} | Speed: {:.2f}ms".format(1000 / result.speed['inference'], result.speed['inference']), 
        (20, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.imshow("output", im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()