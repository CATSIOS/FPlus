import re

data = open('app/src/main/assets/sunxds_0.8.0.tflite', 'rb').read()
print('file size:', len(data))

# 提取所有可读字符串（操作名、张量名等）
strings = re.findall(rb'[\x20-\x7e]{5,}', data)
seen = []
for s in strings:
    try:
        t = s.decode('ascii')
        if t not in seen:
            seen.append(t)
    except:
        pass

# 打印关键操作名
ops = [t for t in seen if t in ('QUANTIZE','DEQUANTIZE','CONV_2D','ADD','MUL','RESIZE','RESHAPE','CONCATENATION','SIGMOID','LOGISTIC','TANH','DIV','SUB','PAD','TRANSPOSE','FULLY_CONNECTED','CUSTOM')]
print('关键操作:', ops)

# 打印所有字符串（前 120 个）
print('--- 所有字符串 ---')
for t in seen[:120]:
    print(t)
