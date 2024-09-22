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
# Demo Video: https://www.bilibili.com/video/BV1sVv8euESQ/
#
from ultralytics import YOLO, solutions
import cv2
import os

model_name = 'yolov8n.pt'
sample_name = 'chengdu_traffic.mp4'
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
samples_path = os.path.join(root_path, 'samples', 'videos', sample_name);
model_path = os.path.join(root_path, 'models', model_name)

cap = cv2.VideoCapture(samples_path)
# cap = cv2.VideoCapture(0)
model = YOLO(model_path)
assert cap.isOpened()
w, h, fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)

reg_h, reg_w, reg_top = int(h / 4), w, int( h / 5 * 3)

reg_pts = [(0, reg_top), (w, reg_top), (w, reg_top + reg_h), (0, reg_top + reg_h)]

counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=reg_pts,
    classes_names=model.names, 
    draw_tracks=True,
    line_thickness=1,
    region_thickness=2,
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    tracks = model.track(frame, persist=True, show=False)
    frame = counter.start_counting(frame, tracks)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()  # release the capture
cv2.destroyAllWindows()  # destroy all the opened windows