#!/usr/bin/env python3
import cv2
import time
import numpy as np
import ast
import os
import argparse
import json
import onnx
from pathlib import Path

os.environ["ZMQ"] = "1"
from cereal import messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType

del os.environ["ZMQ"]
vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
while not vipc_client.connect(False):
    time.sleep(0.1)

st = time.time()
while True:
    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None:
      print('Skipping')
      continue
    if not yuv_img_raw.data.any():
       continue

    imgff = yuv_img_raw.data.reshape(-1, vipc_client.stride)
    imgff = imgff[:vipc_client.height * 3 // 2, :vipc_client.width]
    img = cv2.cvtColor(imgff, cv2.COLOR_YUV2RGB_NV12)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    et = time.time()
    print(et - st)
    # cv2.imwrite(str(Path(__file__).parent / 'yolo.jpg'), yolo_runner.draw_boxes(img, outputs))













