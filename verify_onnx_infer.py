import onnxruntime as ort
import numpy as np
import cv2

sess = ort.InferenceSession('sunxds_0.8.0.onnx')

img = cv2.imread('game.jpg')  # BGR
h, w = img.shape[:2]

# letterbox 到 320x320
scale = min(320 / w, 320 / h)
nw, nh = round(w * scale), round(h * scale)
pad_w = (320 - nw) // 2
pad_h = (320 - nh) // 2
resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
letterbox = np.full((320, 320, 3), 114, dtype=np.uint8)
letterbox[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized

# BGR -> RGB -> /255 -> NCHW
rgb = letterbox[:, :, ::-1]
x = rgb.astype(np.float32) / 255.0
x = x.transpose(2, 0, 1)[None]  # [1,3,320,320]

out = sess.run(None, {'images': x})[0]  # [1, 6, 2100]
print('output shape:', out.shape)

# 解析：channel 0-3 box, 4-5 class
out = out[0]  # [6, 2100]
box = out[:4]  # [4, 2100]
cls = out[4:]  # [2, 2100]

# 找最大类别分数
conf = cls.max(axis=0)  # [2100]
best_idx = np.argmax(conf)
best_class = int(np.argmax(cls[:, best_idx]))
print(f'max conf={conf.max():.4f}, class={best_class} ({"player" if best_class==0 else "head"})')
print(f'box(cx={box[0][best_idx]:.4f}, cy={box[1][best_idx]:.4f}, w={box[2][best_idx]:.4f}, h={box[3][best_idx]:.4f})')

# 打印 top5
idx = np.argsort(conf)[::-1][:5]
for i in idx:
    c = int(np.argmax(cls[:, i]))
    print(f'  conf={conf[i]:.4f}, class={c}, cx={box[0][i]:.4f}, cy={box[1][i]:.4f}, w={box[2][i]:.4f}, h={box[3][i]:.4f}')
