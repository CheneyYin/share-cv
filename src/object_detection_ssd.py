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
# Demo Video: https://www.xiaohongshu.com/discovery/item/6686284b000000000a0071b2?source=webshare&xhsshare=pc_web&xsec_token=ABqS51KED1BaRkHIHgU-s70nnm9X7Jtmk6WmOlqAYlo9c=&xsec_source=pc_share
#

import cv2
import time
import os
from resources import get_model_path, get_sample_path

model_path = get_model_path("frozen_inference_graph.pb")
config_path = get_model_path("ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
classes_path = get_model_path("coco.names")
video_path = get_sample_path("Street-at-Sunset.mp4")

net = cv2.dnn_DetectionModel(model_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
    classes.insert(0, "__background__")

print(classes)


cap = cv2.VideoCapture(video_path)

last_time = 0
classLabelIds, confidences, bboxs = None, None, None

def current_ms():
    return int(round(time.time() * 1000))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    start_time = current_ms()
    classLabelIds, confidences, bboxs = net.detect(frame, confThreshold=0.5)
    end_time = current_ms()
    cost_time = end_time - start_time
    cost_time = 1 if cost_time == 0 else cost_time

    start_draw_time = current_ms()
    idx = 0
    for [x, y, w, h] in bboxs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=1)
        classLabel = classes[classLabelIds[idx]]
        confidence = confidences[idx]
        title = "{}: {:.2f}".format(classLabel, confidence)
        cv2.putText(frame, title, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        idx = idx + 1

    cv2.putText(
         frame, 
         "FPS: {:.2f} | Latency: {:.2f}ms".format(1 / cost_time * 1000, cost_time), 
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow("Output", frame)
    end_draw_time = current_ms()
    draw_time = end_draw_time - start_draw_time
    print("cost_time: {:.2f}ms".format(cost_time))
    print("draw_time: {:.2f}ms".format(draw_time))

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

