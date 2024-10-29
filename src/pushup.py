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
# Demo Video: https://www.bilibili.com/video/BV1sVv8euEAY/
#

import cv2

from ultralytics import YOLO, solutions
from resources import get_model_path, get_sample_path

model_path = get_model_path('yolov8n-pose.pt')
sample_path = get_sample_path('pushup.mp4')
model = YOLO(model_path)
cap = cv2.VideoCapture(sample_path)
assert cap.isOpened()
w, h, fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)

gym_object = solutions.AIGym(
    line_thickness=2,
    view_img=False,
    pose_type="pushup",
    kpts_to_check=[6, 8, 10],
    pose_down_angle=100,
    pose_up_angle=125,
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame)
    frame = gym_object.start_counting(frame, results)
    
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()