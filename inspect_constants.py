import struct

data = open('app/src/main/assets/sunxds_0.8.0.tflite', 'rb').read()

def f32(x):
    return struct.pack('<f', x)

targets = {
    '255.0': f32(255.0),
    '1/255 (0.0039216)': f32(1.0/255.0),
    '0.00390625 (1/256)': f32(1.0/256.0),
    '114.0 (letterbox灰)': f32(114.0),
    '0.5': f32(0.5),
    '0.447 (114/255)': f32(114.0/255.0),
}

for name, pat in targets.items():
    print(f'{name}: {data.count(pat)} 次')

# 也搜索 uint8 的 255（int8 量化特征）
print('uint8 255 字节 (0xFF):', data.count(b'\xff'))
