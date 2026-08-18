import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

# ===== 模拟 PoseEstimator.estimate 的完整流程 =====
INPUT_SIZE = 320
CONFIDENCE_THRESHOLD = 0.2
MIN_AREA_THRESHOLD = 0.01

img = cv2.imread('game.jpg')  # BGR
original_h, original_w = img.shape[:2]

# 1. preprocessBitmap（letterbox 到 320，填充 114）
scale = min(INPUT_SIZE / original_w, INPUT_SIZE / original_h)
nw, nh = round(original_w * scale), round(original_h * scale)
letterbox_x = (INPUT_SIZE - nw) // 2
letterbox_y = (INPUT_SIZE - nh) // 2
resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
lb = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
lb[letterbox_y:letterbox_y+nh, letterbox_x:letterbox_x+nw] = resized
rgb = lb[:, :, ::-1]  # BGR->RGB

# 2. bitmapToByteBuffer（NCHW，/255）
chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
tensor = np.expand_dims(chw, axis=0)  # [1,3,320,320]

# 3. 推理
it = Interpreter(model_path='app/src/main/assets/sunxds_0.8.0.tflite')
it.allocate_tensors()
it.set_tensor(it.get_input_details()[0]['index'], tensor)
it.invoke()
out = it.get_tensor(it.get_output_details()[0]['index'])[0]  # [6,2100]

# 4. parseOutput：找最大 conf*area 的 box
box = out[:4]      # [4,2100] cx,cy,w,h（归一化）
cls = out[4:]      # [2,2100]
conf = cls.max(axis=0)
best_score = -1
best = None
for i in range(2100):
    c = conf[i]
    if c < CONFIDENCE_THRESHOLD:
        continue
    w, h = box[2, i], box[3, i]
    area = w * h
    if area < MIN_AREA_THRESHOLD:
        continue
    score = c * area
    if score > best_score:
        best_score = score
        best = i
        best_conf = c
        best_cls = int(np.argmax(cls[:, i]))

print(f'parseOutput 选中: conf={best_conf:.4f}, class={best_cls}, anchor={best}')
print(f'  原始输出 box(归一化, 相对320): cx={box[0,best]:.4f}, cy={box[1,best]:.4f}, w={box[2,best]:.4f}, h={box[3,best]:.4f}')

# 5. estimate 里的 letterbox 逆变换
bCx, bCy, bW, bH = box[0,best], box[1,best], box[2,best], box[3,best]
pixelCx = bCx * INPUT_SIZE
pixelCy = bCy * INPUT_SIZE
pixelW = bW * INPUT_SIZE
pixelH = bH * INPUT_SIZE
origCx = (pixelCx - letterbox_x) / scale
origCy = (pixelCy - letterbox_y) / scale
origW = pixelW / scale
origH = pixelH / scale
final_cx = origCx / original_w
final_cy = origCy / original_h
final_w = origW / original_w
final_h = origH / original_h

print(f'  逆变换后 box(归一化, 相对原始图): cx={final_cx:.4f}, cy={final_cy:.4f}, w={final_w:.4f}, h={final_h:.4f}')

# 6. 对比 .pt 的参考值
print(f'  参考(.pt检测): cx=0.479, cy=0.514, w=0.467, h=0.972')
