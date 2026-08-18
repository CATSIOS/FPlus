import numpy as np
import cv2

from ai_edge_litert.interpreter import Interpreter

it = Interpreter(model_path='app/src/main/assets/sunxds_0.8.0.tflite')
it.allocate_tensors()
inp = it.get_input_details()
out = it.get_output_details()

# 预处理 game.jpg
img = cv2.imread('game.jpg')  # BGR
h, w = img.shape[:2]
scale = min(320 / w, 320 / h)
nw, nh = round(w * scale), round(h * scale)
pad_w = (320 - nw) // 2
pad_h = (320 - nh) // 2
resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
letterbox = np.full((320, 320, 3), 114, dtype=np.uint8)
letterbox[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized
rgb = letterbox[:, :, ::-1]  # BGR -> RGB, shape [H, W, C]

def run_nchw(name, arr_hwc):
    # arr_hwc: [H, W, C] float32, 转 NCHW [1, C, H, W]
    chw = arr_hwc.transpose(2, 0, 1)  # [C, H, W]
    tensor = np.expand_dims(chw, axis=0)  # [1, C, H, W]
    it.set_tensor(inp[0]['index'], tensor)
    it.invoke()
    o = it.get_tensor(out[0]['index'])[0]  # [6, 2100]
    cls = o[4:]
    conf = cls.max()
    idx = int(np.argmax(cls.max(axis=0)))
    best_class = int(np.argmax(cls[:, idx]))
    box = o[:4, idx]
    print(f'{name}: conf={conf:.4f}, class={best_class}, box(cx={box[0]:.2f}, cy={box[1]:.2f}, w={box[2]:.2f}, h={box[3]:.2f})')

# 测试 1：输入 [0,1]（除以 255）
run_nchw('NCHW [0,1] /255', rgb.astype(np.float32) / 255.0)

# 测试 2：输入 [0,255]（不除）
run_nchw('NCHW [0,255] 原值', rgb.astype(np.float32))
