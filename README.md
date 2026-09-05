# FPlus

基于 Android 屏幕捕获的实时目标检测与跟踪应用。通过 MediaProjection 捕获屏幕画面，使用 TFLite + GPU 加速运行 YOLOv12 目标检测模型，在悬浮窗上实时绘制目标框与跟踪区域，集成多阶段目标跟踪与动态亮度增强。

### 交流群

QQ：1106060347
FPlus交流群，用于反馈BUG以及学习交流。禁止一切违法活动。请在合法情况下改编该项目(FPuls)

## 功能特性

- **实时屏幕检测**：MediaProjection 屏幕捕获 + ImageReader 帧采集，40fps 量级端到端延迟

- **多模型支持**：云端托管按需下载（320/640 分辨率 × 标准/OPT 变体），运行时自动识别 box 编码与输入尺寸

- **GPU 加速推理**：TFLite GPU Delegate（FAST\_SINGLE\_ANSWER + FP16 精度、XNNPACK 关闭）单帧推理约 25\~45ms（视机型与模型）

- **多阶段目标跟踪**：C-BIoU 两阶段匹配 + OC-SORT Coast 速度外推 + OCM 方向一致性（转向自抑制）+ OA-SORT 遮挡感知（OAM/BAM）+ 恢复帧 ORU 全信观测 + OATrack 三阶段丢失恢复 + MeMoSORT 高度相似性约束

- **目标锁定保持（track-and-hold）**：锁定目标后不切换到其他目标，杜绝多目标横跳；短暂漏检补位显示、持续丢失快速淡出

- **动态亮度增强**：P90 高分位亮度检测 + EVdiff 对比度判定，区分"环境暗"与"暗部物体"，自动增益与暗部抬升

- **ROI 自适应跟随**：检测区域随目标非线性跟随（死区 + 平方根缩放 + 单帧限幅），无目标时平滑回中

- **平滑显示**：1€ 自适应低通滤波（速度自适应、截止频率 6Hz 封顶、速度低通 0.3Hz）+ 二阶 ease 缓动，目标消失时重置滤波器缓存避免目标切换拖影

- **可折叠设置面板**：原生 ExpandableListView（Android 设置风格），模型切换、跟踪参数、亮度参数、ROI 等全部可调

## 技术栈

| 组件   | 说明                                        |
| ---- | ----------------------------------------- |
| 语言   | Java                                      |
| 推理框架 | TensorFlow Lite 2.x + GPU Delegate        |
| 目标检测 | YOLOv12n（320/640 × 标准/OPT 变体，云端按需下载）      |
| 屏幕捕获 | MediaProjection + ImageReader（RGBA\_8888） |
| 覆盖层  | TYPE\_APPLICATION\_OVERLAY 悬浮窗            |
| 构建   | Gradle + Android Gradle Plugin            |

## 模型

模型托管在 Cloudflare Pages（`https://fplus-models.pages.dev/`），APK 不内置模型：首次使用或切换模型时按需下载到应用私有目录；云端清单 `models.json` 动态维护可用模型（与本地已下载合并展示，支持逐模型下载/删除）。当前云端清单：

| 模型文件                     | 输入尺寸    | 说明             |
| ------------------------ | ------- | -------------- |
| `DeltaForce_640.tflite`  | 640×640 | 默认模型，大分辨率小目标更准 |
| `DeltaForce_320.tflite`  | 320×320 | 同场景，速度优先       |
| `CODM_320.tflite`        | 320×320 | CODM 场景        |
| `General_320_opt.tflite` | 320×320 | 通用 OPT 格式      |

> box 编码按文件名区分：`_opt` 后缀模型输出 cxcywh 像素值（0\~inputSize），非 OPT 输出归一化 cxcywh；`PoseEstimator` 自动识别并归一化。输入布局 NCHW/NHWC 与输入尺寸由模型 shape 运行时自动检测。

## 核心流程

```
屏幕帧 → ROI 裁剪缩放 → 亮度增强(LUT) → TFLite GPU 推理 → 输出解析/坐标反变换
      → 多阶段跟踪(选主/OCM匹配/丢失恢复) → 位置平滑(1€+ease) → 悬浮窗绘制(绿框+ROI黄圈)
```

## 主要高级选项

| 参数                           | 默认   | 说明                      |
| ---------------------------- | ---- | ----------------------- |
| 环境光优化 `bright_auto`          | 开    | 动态增益 + 暗部抬升（P90/EVdiff） |
| 有效检测置信度 `det_valid`          | 0.20 | HIGH 阶段阈值               |
| 跟踪接管置信度 `track_takeover`     | 0.50 | 目标切换门槛                  |
| ROI 比例 `roi_scale`           | 0.70 | 检测区域占短边比例               |
| 最小目标面积 `det_min_area`        | 0.01 | 过滤小 UI 元素（1% 屏幕面积）      |
| 1€ 截止频率 `overlay_min_cutoff` | 2.0  | 静止平滑度                   |
| 1€ 速度系数 `overlay_beta`       | 4.0  | 快速移动响应度（自适应截止频率 6Hz 封顶） |
| ease 缓动 `overlay_ease`       | 0.4  | 二阶平滑保留比例                |
| 尺寸平滑 `overlay_size_smooth`   | 0.45 | 框尺寸平滑系数                 |

## 构建

```bash
# 需要 Android SDK（compileSdk 36）与 JDK 17+
./gradlew assembleDebug
# APK 输出: app/build/outputs/apk/debug/app-debug.apk
```

安装后需授予「屏幕捕获」与「悬浮窗」权限。

## 目录结构

```
app/src/main/java/com/example/fplus/
├── MainActivity.java            # 主界面（权限申请、模型选择）
├── ScreenCaptureService.java    # 屏幕捕获 + 前台服务
├── PoseEstimator.java           # TFLite 推理 + 输出解析 + 多阶段跟踪
├── OverlayView.java             # 悬浮窗绘制（绿框/ROI/淡出）
├── OneEuroFilter.java           # 1€ 自适应低通滤波器
└── AdvancedOptionsActivity.java # 可折叠高级设置面板
```

## 版本历史

- **开发中（未发布）** 大目标低速跟踪审计修复：OCM 方向一致性（转向自抑制）、恢复帧 ORU 全信观测、速度估计连续帧门控与计时基准刷新、1€ 滤波 beta 4.0 + 截止频率 6Hz 封顶

- **v1.1.0** 设置页重构（原生组件+导航项）、目标锁定开关、跟踪修复（CA 外推限幅 / BAM 混合门限 / 1€ 参数调优）、删除废弃布局

- **v1.0.1** 基线版本重构

- **v5.9** 修复目标切换横跳（1€ 滤波器目标消失时重置缓存）；环境光优化/双路交叉核验/抗准心实验项；跟踪锁定保持重构

- **v5.8.x** OPT 模型 box 解析修复（cxcywh 像素值归一化）、原生 ExpandableListView 设置面板、320/416 OPT 模型

- **v5.4+** 动态亮度增益（P90 高分位 + EVdiff 对比度）、性能加固、目标跟踪算法迭代

***

## 免责声明

本项目仅用于计算机视觉与目标跟踪技术的学习与研究。使用者需自行确保使用场景符合当地法律法规及相关软件用户协议。作者不对任何不当使用行为承担责任。
