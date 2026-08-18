from ultralytics import YOLO

m = YOLO('sunxds_0.8.0.pt')
res = m.predict('game.jpg', conf=0.05, verbose=False)
total = 0
for r in res:
    for b in r.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        xyxy = b.xyxy[0].tolist()
        print(f'class={cls} ({m.names[cls]}), conf={conf:.4f}, xyxy={[round(v,1) for v in xyxy]}')
        total += 1
print('TOTAL DETECTIONS:', total)
