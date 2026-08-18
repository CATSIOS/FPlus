from ultralytics import YOLO
import onnxruntime as ort
import numpy as np

# 导出 onnx
m = YOLO('sunxds_0.8.0.pt')
m.export(format='onnx', imgsz=320, half=False)

# 检查 onnx 输入输出
sess = ort.InferenceSession('sunxds_0.8.0.onnx')
print('=== ONNX inputs ===')
for i in sess.get_inputs():
    print(f'  name={i.name}, shape={i.shape}, type={i.type}')
print('=== ONNX outputs ===')
for o in sess.get_outputs():
    print(f'  name={o.name}, shape={o.shape}, type={o.type}')
