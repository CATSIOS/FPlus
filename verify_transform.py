from ultralytics import YOLO
import cv2
import numpy as np

m = YOLO('sunxds_0.8.0.pt')

img = cv2.imread('game.jpg')  # BGR
h, w = img.shape[:2]
print(f'image size: {w}x{h}')

def det(name, arr):
    r = m.predict(arr, conf=0.05, verbose=False)
    boxes = [(int(b.cls[0]), round(float(b.conf[0]), 3)) for rr in r for b in rr.boxes]
    print(f'{name}: {boxes}')

# 1. 正常 BGR（ultralytics 会自动转 RGB）
det('BGR正常', img)

# 2. R/B 通道交换（模拟颜色错误）
det('R/B交换', img[:, :, [2, 1, 0]])

# 3. 灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
det('灰度', cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

# 4. 上下翻转
det('上下翻转', cv2.flip(img, 0))

# 5. 左右翻转
det('左右翻转', cv2.flip(img, 1))
