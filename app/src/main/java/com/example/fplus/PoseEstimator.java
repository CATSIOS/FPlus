package com.example.fplus;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Process;
import android.text.TextUtils;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

public class PoseEstimator {

    private static final String TAG = "PoseEstimator";
    private static final int INPUT_SIZE = 640;
    // 可调参数（从 SharedPreferences 读取，详见 AdvancedOptionsActivity）
    private float CONFIDENCE_THRESHOLD = 0.15f;
    // 判定为「有效目标」的最低置信度：C-BIoU 高置信度检测阈值
    private float VALID_DETECTION_CONF = 0.2f;
    // C-BIoU 低置信度检测阈值（第二阶段救援），用于"救活"被遮挡/漏检但跟踪中的目标
    private static final float BYTETRACK_LOW_CONF = 0.1f;
    private float MIN_AREA_THRESHOLD = 0.01f;
    // 距离准星（屏幕中心）的高斯权重标准差（归一化，越大锁定范围越宽）
    private float CENTER_SIGMA = 0.2f;
    // 方案3：水平偏心角度分数（FPS 横向瞄准是关键，垂直偏心不惩罚）
    // 借鉴 Best Target Selection (Nicholas Gorski) 的"距离+角度"组合评分
    private static final float ANGLE_SIGMA = 0.25f;
    // 评分组合权重：score = (1-W)*distWeight + W*angleWeight
    private float SCORE_W = 0.3f;
    // 当前锁定目标的连续性加成倍数（防止多目标间来回跳）
    private static final float TRACK_BONUS = 2.0f;
    // 接管阈值：新目标置信度 ≥ 此值时绕过 anti-lock-jump 保护，立即切换锁定
    // 解决「人物突然入场」场景下被压制 MAX_TRACK_LOST 帧导致的 ~0.5s 延迟
    // 远处小目标置信度通常 < 0.5，仍受保护；清晰人物入场 > 0.5 可即时响应
    private float TAKEOVER_CONF = 0.5f;
    // 目标锁定：开启后禁用即时接管，只能等当前目标彻底消失（MAX_TRACK_LOST 帧）才允许换锁
    private boolean TARGET_LOCK_ENABLED = false;
    // ROI（黄框）占短边的比例，缩小让识别区域更聚焦中心
    private float ROI_SCALE = 0.7f;
    // 目标丢失时 ROI 回中的平滑系数（每帧移动剩余距离的比例）
    private static final float RECENTER_SMOOTH = 0.2f;
    private static final float INV_255 = 1f / 255f;
    // 亮度/对比度增强：游戏场景偏暗，提亮有助于提升检测率
    // USER_* = 高级选项用户手动设置的基础值；CUR_* = 当前帧实际应用值（USER_* 乘动态补偿倍率 mult）
    private boolean BRIGHT_AUTO_ENABLED = true;  // 高级选项"启用环境光优化"，true=动态增益+暗部抬升，false=只使用 USER_* 基础值
    private float USER_BRIGHT_GAIN = 1.3f;
    private float USER_BRIGHT_OFFSET = 25f;
    // CUR_* 在主线程 bitmapToByteBuffer 里写、第二线程 bitmapToByteBuffer2 里读（brighten），
    // 加 volatile 保证跨线程可见性，避免第二路读到过期亮度值导致两路输入不一致
    private volatile float CUR_BRIGHT_GAIN = 1.3f;
    private volatile float CUR_BRIGHT_OFFSET = 25f;
    private volatile float CUR_BRIGHT_OFFSET_NORM;
    // 动态亮度配置：场景亮度用 P90 高分位（替代平均亮度，避免"亮环境+黑物体"误判为暗）、
    // EMA平滑系数、重建LUT的变化阈值、补偿倍率上下限
    private static final float TARGET_P90_LUMA = 150f;   // 期望的 P90 亮度：亮场景不提亮，暗场景提亮
    private static final float LUM_EMA_ALPHA = 0.3f;
    private static final float REBUILD_GAIN_RATIO = 0.05f;  // ±5% 变化才重建 LUT
    private static final float REBUILD_OFFSET_ABS = 3.0f;    // 绝对值±3 才重建
    private static final float MIN_MULT = 0.75f;
    private static final float MAX_MULT = 1.8f;
    // 暗部抬升（方案B）：仅当"场景亮(P90 高) + 对比度大(亮背景/黑物体)"时启用，
    // 用分段曲线把阴影区抬升、亮部不动，既看清黑物体又不整体发灰
    private static final float SHADOW_LIFT_BRIGHT_MIN = 120f;  // P90 ≥ 此值视为场景亮
    private static final float SHADOW_LIFT_EVDIFF_MIN = 70f;   // 亮部12%均值 − 暗部12%均值 ≥ 此值视为高对比
    private static final float SHADOW_LIFT_MAX = 0.25f;        // 最大抬升强度（归一化亮度域 0~1）
    private static final float SHADOW_LIFT_CUTOFF = 0.45f;     // 阴影判定截止（输入亮度 ≥ 此值不抬升）
    private static final float SHADOW_LIFT_EMA_ALPHA = 0.2f;   // 抬升强度平滑，防 LUT 跳变
    private static final float SHADOW_REBUILD_ABS = 0.02f;     // 抬升强度变化 ≥ 此值才重建 LUT
    private static final int LUM_HIST_BINS = 32;               // 亮度直方图分桶（每桶 8 级）
    // 动态亮度状态字段
    private float emaP90 = -1f;      // P90 场景亮度的 EMA（-1 表示未初始化）
    private float shadowLift = 0f;   // 当前暗部抬升强度（EMA 平滑后的 0~SHADOW_LIFT_MAX）
    private float lastRebuildShadow = -1f;
    private boolean shadowLiftLogged = false;  // 暗部抬升首次激活日志（验证用，只打一次）
    private float lastRebuildGain = -1f;
    private float lastRebuildOffset = -1f;
    private boolean dynBrightComputed = false;

    private Interpreter interpreter;
    private GpuDelegate gpuDelegate;
    private NnApiDelegate nnApiDelegate;
    private Backend backend;
    private String modelName;  // 记录模型文件名，用于 box 格式/版本判定
    private ByteBuffer inputBuffer;
    private FloatBuffer inputFloatBuffer;
    private float[] inputFloats;
    private int[] pixels;
    // convert 零-copy 优化：copyPixelsToBuffer 经 ByteBuffer.wrap 一次 memcpy 直入 heap byte[]（RGBA），
    // 跳过 getPixels 的 int 打包与 DirectBuffer 中转；brightLut 亮度查表（256 项），省每像素浮点乘加 clamp
    private byte[] pixelBytes;
    private float[] brightLut;
    private Bitmap scaledRoi;
    private Canvas scaleCanvas;
    private Paint scalePaint;
    private Rect scaleRect;
    private Rect scaleSrcRect;
    private float[][][] output;

    // 多输出支持（ultralytics onnxsim+onnx2tf 新版把 box/score 分成两个 output tensor）：
    //   multiOutputMode=true : outputBoxes = [1,4,N or N,4]（x1y1x2y2 或 cxcywh）
    //                          outputScores = [1,nc,N or N,nc]（类置信度）
    //                          numChannels = 4 + nc（逻辑上合成单输出的通道数）
    //                          调用 interpreter.run 时用 Map<Integer, Object> outputs
    //   multiOutputMode=false: 旧单 output 模式，output = [1,4+nc,N]
    private boolean multiOutputMode = false;
    private float[][][] outputBoxes;
    private float[][][] outputScores;
    private int numScoreClasses = 0;  // multiOutput 模式下 score 的通道数
    private boolean firstInferLogged = false;  // 第1帧推理后打 WARN 日志，查真实数值
    private boolean realStatsLogged = false;   // 真实画面首帧 conf/box 范围统计（仅OPT模型诊断用，1次）
    private boolean dualVerifyWarned = false;  // 双路交叉核验首次命中FP惩罚的诊断日志（只打1次）
    private boolean antiCrosshairWarned = false;// 抗准心首次命中惩罚的诊断日志（只打1次）
    private int estimateCallCount = 0;          // 调用计数：probe=1, 真实画面首帧=2

    // ===== PoC: 双实例并发推理（榨 GPU 并行算力）=====
    // 目的：验证 TFLite 双 GpuDelegate 实例能否真并发，榨干 GPU 空闲算力且端到端不卡。
    // 第二路跑「中心放大区」（ROI 中心 zoomSize 正方形缩放到 640），远处小目标放大后更易检出。
    // 第二套资源与主路完全隔离，避免并发时 pixels/floats/buffer/canvas 字段冲突。
    // 目标误识别过滤：抗准心/瞄具 UI 误绑定（开镜场景中心准心误识别为小人）。默认关。
    // 只惩罚置信度（×0.35），不硬移除，lock 不受直接影响；必须同时满足3条才罚：
    //   (1) 候选中心与画面几何中心距离 < 1.2% 屏幕对角线（准心永远居中）
    //   (2) 高宽比 h/w 不在真人形范围 [1.6, 3.6]（正圆/十字瞄具通常 h/w 0.8~1.5）
    //   (3) 尺寸足够小：宽 < 2.2% 屏幕宽 且 高 < 2.8% 屏幕高
    private boolean ANTI_CROSSHAIR_ENABLED = false;
    private static final float AC_DIST_RATIO = 0.012f;
    private static final float AC_HW_MIN = 1.6f;
    private static final float AC_HW_MAX = 3.6f;
    private static final float AC_W_RATIO_MAX = 0.022f;
    private static final float AC_H_RATIO_MAX = 0.028f;
    private static final float AC_PENALTY = 0.35f;

    // PoC1 阶段：不融合第二路结果到主路跟踪，仅日志对比检出差异 + 并发性能数据。
    private boolean DUAL_INFER_ENABLED = false;          // 开关：默认关，保证开关关时行为与单实例逐字节一致
    private boolean DUAL_VERIFY_ENABLED = false;         // 实验开关：双路交叉核验（双路开才生效），主路放大区候选需第二路同位置也检出才采信
    private float DUAL_ZOOM = 0.5f;                       // 第二路区域相对主 ROI 边长比例（越小放大倍数越大）
    private Interpreter interpreter2;
    private GpuDelegate gpuDelegate2;
    private ByteBuffer inputBuffer2;
    private FloatBuffer inputFloatBuffer2;
    private float[] inputFloats2;
    private int[] pixels2;
    private float[][][] output2;
    // 第二路多输出模式（boxes/scores 分离）：与主路 outputBoxes/outputScores 对称，避免单 output 模式下 NPE
    private float[][][] outputBoxes2;
    private float[][][] outputScores2;
    private Bitmap scaledRoi2;
    private Canvas scaleCanvas2;
    private Paint scalePaint2;
    private Rect scaleRect2;
    private Rect scaleSrcRect2;
    private ExecutorService secondExecutor;
    // 第二路 infer 耗时（跨线程写、主线程读），AtomicLong 保证可见性
    private final AtomicLong secondInferNanos = new AtomicLong(0L);
    // 双实例性能统计（每 30 帧打一次，对比单路基准）
    private long dualMainInferAccum;
    private long dual2ndInferAccum;
    private long dualTotalAccum;
    private int dualFrames;
    // 第二路当前帧 zoom 区域（每帧由 preprocessBitmap2 计算，供 parseOutput2 反变换）
    private int zoomSize;
    private int zoomX;
    private int zoomY;

    // 输出维度顺序：true 表示 [1, channels, anchors]，false 表示 [1, anchors, channels]
    private boolean channelFirst = true;
    private int numChannels = 0;
    private int numAnchors = 0;

    // Box 编码格式：
    //   false (默认旧格式): [cx, cy, w, h]（归一化 0~1）
    //   true (OPT新模型)  : [x1, y1, x2, y2] corner（ultralytics + onnxsim 折叠Decode后输出）
    // 判定策略：文件名含 "_opt" 初判 true + 锚点启发式统计（x1<x2且y1<y2占多数）验证，
    // 不一致时以统计为准并打日志，防止误判。
    private boolean boxIsXyxy = false;

    // 输入维度顺序：true 表示 NCHW [1,3,H,W]，false 表示 NHWC [1,H,W,3]
    private boolean inputNchw = true;
    // 实际输入尺寸（从模型 inputShape 动态读取，支持 640/416 等）
    private int inputSize = INPUT_SIZE;

    private int roiX;
    private int roiY;
    private int roiSize;

    private int originalWidth;
    private int originalHeight;

    private float[] trackedBox;
    private int trackLostFrames = 0;
    // TRACK_BUFFER：跟踪丢失保留窗口，30fps 下 30 帧 ≈ 1 秒缓冲
    // 论文默认 60 帧（≈2秒），但游戏场景目标短暂出框/被遮挡后通常 1 秒内回归
    // 过短（原 20 帧=0.67 秒）会导致目标短暂闪身后回来重新锁定慢
    private int MAX_TRACK_LOST = 30;
    // C-BIoU Tracker：roboflow trackers v2.3.0 基准（2026-03）——MOT17 HOTA 63.0/IDF1 79.1、
    // SportsMOT HOTA 73.1，均优于 OC-SORT（MOT17 61.9）；但低于 BoT-SORT（MOT17 63.7/MOTA 79.2）。
    // BoT-SORT 优势来自 CMC（相机运动补偿，稀疏光流）；本项目滑动跟随时画面整体平移，
    // 属同类场景，故补了 CMC 轻量替代（cameraMotionActive：冻结速度 + 放宽匹配）。
    // 评估口径应以 DanceTrack（快速运动/相机运动，贴近 FPS）为准，而非 MOT17（行人监控静态相机）。
    // 替代 ByteTrack 纯 EIoU 匹配：buffer 扩展（中心不变，宽高×(1+2*buf)）后再算 IoU，
    // 解决 FPS 横向快速移动时纯 IoU 骤降断锁。两阶段 buffer/阈值不同：
    //   HIGH 阶段：小 buffer(0.3)+低阈值(0.2)，高速仍能匹配
    //   LOW 救援：大 buffer(0.5)+阈值(0.3)，放宽匹配容许遮挡漂移
    // HIGH 阶段 buffered IoU 阈值
    private float TRACK_IOU_THRESHOLD = 0.2f;
    private float CBIoU_BUF_HIGH = 0.3f;   // HIGH 阶段 buffer 扩展比例
    // HIGH 检测框 NMS 去重 IoU 阈值：同一个目标 YOLO 常出 2~3 个重叠高分框，
    // 不处理会导致 OAM occlusion 把 1 人误判成 3 人遮挡 → BAM 不信任观测 → 跟踪漂移。
    // 0.8 严格重叠，同目标重复框必压，多人物不互压
    private static final float HIGH_NMS_IOU_THRESH = 0.8f;

    // 方案 A：MeMoSORT Mo-IoU（ICLR 2026 Submission，DanceTrack HOTA 67.9）的高度相似性分量。
    // 同一人物高度不会突变：横向快移时高度比例一致（0~1的heightSim）加分，
    // 高度差异大（比如旁边另一个人身高明显不同）减分，降低横向重叠时的错匹配。
    // 叠加到 simTrack，高度完全一致（heightSim=1）加 HEIGHT_SIM_WEIGHT，
    // 完全不一致（heightSim=0，高度>2倍差）加 0。
    private static final float HEIGHT_SIM_WEIGHT = 0.2f;

    // 方案 B：OATrack（Sensors 2026，UAV 小目标场景）三阶段渐进分配第三阶段「刚丢失回收」。
    // LOW 救援失败后：trackLostFrames∈[1, RECOVERY_MAX_FRAMES] 的目标去 LOW 集合里
    // 用更宽缓冲(RECOVERY_BUF=0.7)和宽松阈值(RECOVERY_IOU=0.2)再扫一次，
    // 专救"短暂出框1~3帧又回来"Coast经常失效的场景。IDSW -69%（VisDrone 论文数据）。
    private static final int RECOVERY_MAX_FRAMES = 3;
    private static final float RECOVERY_BUF = 0.7f;
    private static final float RECOVERY_IOU = 0.2f;
    private float CBIoU_BUF_LOW = 0.5f;    // LOW 救援 buffer 扩展比例
    private float CBIoU_IOU_LOW = 0.3f;    // LOW 救援 buffered IoU 阈值
    // 速度自适应 buffer：速度归一化（速度/框宽）达到此值时 HIGH buffer 放大到 LOW 上限
    // 静止时 buffer=BUF_HIGH(0.3) 防误配，快移时放大到 BUF_LOW(0.5) 防失配
    private float CBIoU_SPEED_REF = 0.25f; // 速度归一化参考（框宽/帧）
    // YOLOv8-SMOT (arxiv 2507.12087) 距离惩罚：横向快移 IoU 骤降但目标实际位移不大，
    // sim = bufferedIoU + DIST_WEIGHT×(1 - dist/diag)，距离近补分，让 IoU 低仍可匹配防横向断锁
    private float DIST_WEIGHT = 0.25f;
    // 运动动力学 KF (arxiv 2505.07254)：加速度超过此值切 CA（匀加速）模型外推，
    // 横向加速时 CA 预测更准，减少 predTracked 偏差；加速度小用 CV（匀速）
    private float ACC_THRESHOLD = 0.002f;
    // OC-SORT OCM：跟踪目标的速度估计（像素/帧），用于匹配前按 lost 帧数做位置外推
    private float trackVelX = 0f;
    private float trackVelY = 0f;
    // CMC（相机运动补偿）轻量替代：滑动跟随注入触摸时画面整体平移，目标在图像中的
    // 表观位移被全局相机运动污染。此时冻结速度估计（不把相机运动算进 trackVel），
    // 并放宽匹配（buffer 放大到 LOW 上限 + 跳过方向惩罚），避免 IoU 骤降断锁。
    private volatile boolean cameraMotionActive = false;
    // CMC 开关（"cmc_enabled"）：默认开启。关闭时即使滑动跟随生效也不触发 CMC 补偿
    private boolean CMC_ENABLED = true;
    // 最后一次匹配的瞬时速度（未 EMA 平滑），Coast 外推专用：响应变向更快，
    // 避免 EMA 速度残留旧方向导致「角色变向丢失时绿框反向滑动」
    private float lastRawVx = 0f;
    private float lastRawVy = 0f;
    // 上上帧瞬时速度，用于算加速度 (acc = lastRawVx - prevRawVx)，CA 模型外推
    private float prevRawVx = 0f;
    private float prevRawVy = 0f;
    private long lastTrackTimeNanos = 0L;
    private static final float VEL_EMA_ALPHA = 0.5f; // 速度 EMA 平滑系数（大目标基准）
    // 方案1：Area-Adaptive Motion Damping（AKKF, arxiv 2607.12544）
    // 小目标速度噪声相对更大（几像素抖动=大比例位移），应增大阻尼降低 EMA alpha
    // area_norm >= AREA_NORM_REF 时 alpha = VEL_EMA_ALPHA，面积越小 alpha 越接近 ALPHA_MIN
    private static final float VEL_ALPHA_MIN = 0.3f;
    private static final float AREA_NORM_REF = 0.05f;

    // 预测未来位置（借鉴 Sunone Aimbot / Aimmy Kalman Lead Time）
    // PREDICT_SECONDS：绿框显示位置外推的提前量（秒）
    //   - Aimmy 默认 0.10s，Mvsd-scripts 默认 3 帧（60fps≈50ms）
    //   - FPlus 移动端 30fps，0.083s ≈ 2.5 帧抵消端到端延迟
    //   - 仍慢可调到 0.10s，跑过头（急转时绿框冲过）可降到 0.067s
    private float PREDICT_SECONDS = 0.083f;
    // 绿框预判开关（"lead_predict"）：默认开启。
    // 开启=绿框按速度提前外推抵消延迟（会略微超前/回退）；关闭=紧贴检测位置，观感更稳
    // （借鉴开源锁敌：显示框不做预判，仅平滑跟踪，预判留给瞄准点）
    private boolean LEAD_PREDICT_ENABLED = true;
    // MIN_VEL：速度低于此值视为静止，不外推（避免静止时绿框漂移）
    //   - trackVelX 单位为 60fps 基准每帧位移，0.001 相当于 60fps 下 0.1% 图像宽/帧
    private static final float MIN_VEL = 0.001f;
    // OC-SORT Coast（速度外推）参数：丢失期间用速度外推位置继续显示绿框
    //   COAST_MIN_VEL：速度低于此值视为静止丢失，不外推直接淡出（目标真没了）
    //   COAST_MAX_FRAMES：外推最多持续帧数，超过则淡出（防持续飘移）
    //   仅在「明显移动 + 短时丢失」时外推衔接，解决大幅移动瞬时丢检测的闪烁
    private float COAST_MIN_VEL = 0.01f;
    private int COAST_MAX_FRAMES = 2;
    // Coast 滞回记忆：大目标低速真实速度恰在 COAST_MIN_VEL 边界，
    // 硬阈值随速度噪声在相邻漏检事件间翻转 → 补位框「外推/冻结」抽搐。
    // 进入需 ≥1.2× 阈值，已激活只需 ≥0.8× 保持，边界两侧留缓冲。
    private boolean coastEngaged = false;
    // 锁定保持显示窗口：已锁定目标短暂漏检时，补位显示旧目标避免闪烁；
    // 超过此帧数仍无匹配则判定「目标真消失」，停止补位返回 null 让绿框快速淡出。
    // 与 MAX_TRACK_LOST(跟踪状态保留 30 帧) 解耦：显示淡出快，跟踪状态保留久以便目标回归时恢复。
    private static final int LOCK_HOLD_FRAMES = 3;  // ≈75ms @40fps，短暂补位防闪烁，消失更快
    // OA-SORT（CVPR 2026, arxiv 2603.06034）：OAM 深度排序阈值（像素），
    // bbox 底部 y 差值小于此值不判定遮挡，避免抖动误判
    private static final float OAM_DEPTH_THRESHOLD = 5f;
    // BAM 混合门槛：bam 接近 1（观测与预测吻合、无遮挡）时不混合。
    // 原实现 bam<1 即混合，实际 bam 恒 <1 → 每帧都把速度外推噪声混回 trackedBox，
    // 大目标（占满 ROI，框中心抖动大）+ 多肢体框遮挡分 >0 时，轨迹被每帧注入噪声缓慢漂移。
    // 0.9→0.95：继续收紧混合区间，状态层更贴观测（少掺预测），
    // 双重平滑的滞后主要由显示层 1€ 承担，符合 OC-SORT 观测中心理念。
    private static final float BAM_BLEND_THRESH = 0.95f;

    // ===== OC-SORT OCM（方向一致性，arXiv 2203.14360）=====
    // 论文第一限制：高帧率下帧间位移噪声与真实位移同量级，仅靠距离的匹配有歧义。
    // OCM 在匹配竞争分里惩罚「候选相对预测位置的位移方向」与「历史速度方向」不一致的框。
    // 与其余 track_* 参数一致，运行时从 SharedPreferences 读取（可调）
    private float OCM_WEIGHT = 0.15f;     // 满速时最大惩罚（从 matchSim 中扣除）
    private float OCM_SPEED_REF = 0.01f;  // 速度≥此值达满权重；低速惩罚按 0.5~1.0 缓降（不归零）

    // ===== 自适应外推阻尼（残差反馈自学习）=====
    // 问题：固定 PREDICT_SECONDS 外推，目标匀速时贴脸、急停/急转时绿框「冲过头」（前推过冲）。
    // 方案：让外推提前量「自主学习」——每帧用「本帧观测位置」与「上帧预测位置」的残差闭环调节：
    //   观测落在预测前方（残差沿外推方向为正）→ 外推不足 → 增大提前量；
    //   观测落在预测后方（残差为负）→ 外推过度 → 减小提前量。
    // 用 EMA 累积残差 + 死区，避免单帧噪声误调，等效一个带死区的积分控制器。
    private float adaptiveLeadSec = 0.083f;   // 当前有效外推提前量（秒），初值 = PREDICT_SECONDS
    private float leadErrEma = 0f;            // 外推残差 EMA（像素，沿外推方向为正 = 外推不足）
    private float lastLeadDx = 0f;            // 上一帧显示外推量 x（像素）
    private float lastLeadDy = 0f;            // 上一帧显示外推量 y（像素）
    private boolean leadValid = false;        // 上帧是否有有效外推（有速度才外推）
    private int leadLogCounter = 0;           // 自适应外推日志限频计数
    private int detLogCounter = 0;            // 检测诊断日志限频计数

    // ===== 静态图闪烁诊断（临时）：30帧窗口汇总 =====
    // 目的：区分「模型间歇漏检（topConf 掉到 LOW 以下）」vs「有检测但被选中逻辑吞掉」。
    // 每帧无论是否选中都累计窗口统计，满 30 帧打一条汇总日志（含 topConf min/max/avg）。
    private int diagFrame = 0;          // 窗口内帧计数
    private int diagSelected = 0;       // bestPose 非空帧数
    private int diagHighFrames = 0;     // 模型输出含 HIGH 框(≥VALID_DETECTION_CONF)的帧数
    private int diagLowFrames = 0;      // 模型输出含 LOW 框(BYTETRACK_LOW_CONF~HIGH)的帧数
    private int diagRejected = 0;       // 锁定保持（heldByLock）补位帧数
    private int diagCoast = 0;          // 锁定保持中速度外推（coast）补位帧数
    private float diagTopMax = 0f;      // 窗口内「模型最高 conf」的最大值
    private float diagTopMin = 1f;      // 窗口内「模型最高 conf」的最小值
    private double diagTopSum = 0;      // 窗口内「模型最高 conf」累加（算均值）

    private static final float LEAD_ERR_EMA_ALPHA = 0.15f;  // 残差 EMA 系数
    private static final float LEAD_DEADZONE = 2.0f;        // 残差死区（像素），小于此不调
    private static final float LEAD_ADAPT_STEP = 0.004f;    // 每次调整步长（秒）
    private static final float LEAD_MAX = 0.083f;           // 外推上限 = 默认 PREDICT_SECONDS
    private static final float LEAD_MIN = 0.0f;             // 外推下限（可退到 0=完全不外推）

    // 分阶段计时统计
    private long preprocessAccum;
    private long convertAccum;
    private long inferAccum;
    private long parseAccum;
    private int timingFrames;
    private static final int TIMING_WINDOW = 30;

    public enum Backend {
        CPU, GPU, NNAPI
    }

    public static class PersonPose {
        public float[] box;
        /** 本目标的置信度（box 内可能只含 cxcywh 4 元素，conf 独立存放，供降权逻辑读取真实值） */
        public float conf = 0f;
        /** 交叉核验用：此检测路径下的所有 valid 检测框，格式 [px,py,pw,ph,conf]（原图像素坐标） */
        public java.util.List<float[]> allValidBoxes;
    }

    public PoseEstimator(Context context, String modelName, Backend preferred) throws IOException {
        this.modelName = modelName;
        // Box 编码格式：新导出的 _opt 模型（ultralytics export + onnxsim + onnx2tf）
        // 的 Detect 模块已做 box decode，输出前四通道是 x1y1x2y2 corner；
        // 旧模型未经 decode，输出是 cxcywh。按文件名直接区分，100% 可靠。
        boxIsXyxy = modelName != null && modelName.contains("_opt");
        Log.w(TAG, "[INIT] model=" + modelName + " boxIsXyxy=" + boxIsXyxy);

        loadPrefs(context);
        MappedByteBuffer modelBuffer = loadModelFile(context, modelName);

        // 按优先级逐次回退：NPU → GPU → CPU，从用户首选开始
        IOException lastError = null;
        for (Backend b : fallbackOrder(preferred)) {
            try {
                if (tryCreateInterpreter(modelBuffer, b)) {
                    this.backend = b;
                    Log.d(TAG, b + " interpreter initialized");
                    break;
                }
            } catch (Throwable t) {
                lastError = new IOException(b + " 初始化失败", t);
                Log.w(TAG, b + " 初始化失败，尝试下一级", t);
            }
        }

        if (interpreter == null) {
            throw (lastError != null) ? lastError : new IOException("所有推理后端初始化失败");
        }
        Log.w(TAG, "[INIT] ckpt1 interpreter NON-NULL, backend=" + this.backend);

        try {
            int[] inputShape = interpreter.getInputTensor(0).shape();
            Log.w(TAG, "[INIT] ckpt2 inputShape=" + Arrays.toString(inputShape));
            if (inputShape.length == 4 && inputShape[3] == 3) {
                inputNchw = false;  // NHWC [1,H,W,3]
            } else {
                inputNchw = true;   // NCHW [1,3,H,W]
            }
            // 从模型 shape 读取实际输入尺寸（支持 640 / 416 等）
            if (inputShape.length == 4) {
                inputSize = inputNchw ? inputShape[2] : inputShape[1];
            }
            org.tensorflow.lite.Tensor inputTensor = interpreter.getInputTensor(0);
            org.tensorflow.lite.Tensor.QuantizationParams inQ = inputTensor.quantizationParams();
            Log.w(TAG, "[INIT] ckpt3 input dtype=" + inputTensor.dataType()
                    + ", shape: " + Arrays.toString(inputShape)
                    + ", quant scale=" + inQ.getScale() + " zeroPoint=" + inQ.getZeroPoint());

            int outputCount = interpreter.getOutputTensorCount();
            Log.w(TAG, "[INIT] outputTensorCount=" + outputCount);
            for (int oi = 0; oi < outputCount; oi++) {
                int[] outShape = interpreter.getOutputTensor(oi).shape();
                org.tensorflow.lite.Tensor outT = interpreter.getOutputTensor(oi);
                org.tensorflow.lite.Tensor.QuantizationParams outQ2 = outT.quantizationParams();
                Log.w(TAG, "[INIT] output[" + oi + "] shape=" + Arrays.toString(outShape)
                        + " dtype=" + outT.dataType()
                        + " quant scale=" + outQ2.getScale() + " zp=" + outQ2.getZeroPoint());
            }
            int[] outputShape = interpreter.getOutputTensor(0).shape();
            org.tensorflow.lite.Tensor outputTensor = interpreter.getOutputTensor(0);
            org.tensorflow.lite.Tensor.QuantizationParams outQ3 = outputTensor.quantizationParams();
            Log.w(TAG, "[INIT] Output[0] shape: " + Arrays.toString(outputShape)
                    + ", dtype=" + outputTensor.dataType()
                    + ", quant scale=" + outQ3.getScale() + " zeroPoint=" + outQ3.getZeroPoint());

            // ---------- 多输出判定：新版 ultralytics 导出通常 outputCount=2（boxes 分离，scores 分离） ----------
            //   output[0] 通常是 [1,4,N] 或 [1,N,4]（boxes），output[1] 通常是 [1,nc,N] 或 [1,N,nc]（scores）
            //   我们把这个模式抽出来合成单输出语义，保证 parseOutput 旧逻辑最小改动。
            if (outputCount >= 2 && outputShape.length == 3) {
                int[] shapeOut0 = interpreter.getOutputTensor(0).shape();
                int[] shapeOut1 = interpreter.getOutputTensor(1).shape();
                boolean ok0 = shapeOut0.length == 3;
                boolean ok1 = shapeOut1.length == 3;
                int d1_0 = (ok0 && shapeOut0[1] < shapeOut0[2]) ? shapeOut0[1] : (ok0 ? shapeOut0[2] : -1);
                int n_0 = (ok0 && shapeOut0[1] < shapeOut0[2]) ? shapeOut0[2] : (ok0 ? shapeOut0[1] : -1);
                int d1_1 = (ok1 && shapeOut1[1] < shapeOut1[2]) ? shapeOut1[1] : (ok1 ? shapeOut1[2] : -1);
                int n_1 = (ok1 && shapeOut1[1] < shapeOut1[2]) ? shapeOut1[2] : (ok1 ? shapeOut1[1] : -1);
                // 典型：output[0]通道数=4（boxes），output[1]通道数=nc>4（scores），anchor数N一致
                if (ok0 && ok1 && d1_0 == 4 && d1_1 > 4 && n_0 == n_1) {
                    multiOutputMode = true;
                    channelFirst = (shapeOut0[1] < shapeOut0[2]);  // boxes 的维度顺序就是全局顺序
                    numAnchors = n_0;
                    numScoreClasses = d1_1;
                    numChannels = 4 + numScoreClasses;  // 逻辑合成：前4是box，后nc是score
                    Log.w(TAG, "[INIT] DETECTED multi-output (boxes+scores separated) mode! "
                            + "boxes_ch=4 scores_ch=" + numScoreClasses + " anchors=" + numAnchors
                            + " channelFirst=" + channelFirst);
                }
            }
            // 单输出模式下要检查 shape
            if (!multiOutputMode && outputShape.length != 3) {
                interpreter.close();
                throw new IOException("Unexpected output[0] shape (single output mode): " + Arrays.toString(outputShape));
            }

            // 判断维度顺序（如果已经是 multiOutput，上面已经算好了，就跳过）
            if (!multiOutputMode) {
                int dim1 = outputShape[1];
                int dim2 = outputShape[2];
                if (dim1 < dim2) {
                    channelFirst = true;
                    numChannels = dim1;
                    numAnchors = dim2;
                } else {
                    channelFirst = false;
                    numChannels = dim2;
                    numAnchors = dim1;
                }
            }
            Log.w(TAG, "[INIT] FINAL channelFirst=" + channelFirst + ", channels=" + numChannels + ", anchors=" + numAnchors
                    + " multiOutput=" + multiOutputMode);

            if (multiOutputMode) {
                int[] s0 = interpreter.getOutputTensor(0).shape();
                int[] s1 = interpreter.getOutputTensor(1).shape();
                outputBoxes = new float[s0[0]][s0[1]][s0[2]];
                outputScores = new float[s1[0]][s1[1]][s1[2]];
            } else {
                output = new float[outputShape[0]][outputShape[1]][outputShape[2]];
            }
            Log.w(TAG, "[INIT] ckpt4 output array allocated OK");
            inputBuffer = ByteBuffer.allocateDirect(1 * 3 * inputSize * inputSize * 4);
            inputBuffer.order(ByteOrder.nativeOrder());
            inputBuffer.rewind();
            inputFloatBuffer = inputBuffer.asFloatBuffer();
            Log.w(TAG, "[INIT] ckpt5 inputBuffer allocated OK");

            // ===== 双实例并发 PoC：仅开关开启时建第二套资源，关闭时零开销 =====
            if (DUAL_INFER_ENABLED) {
                initSecondInstance(modelBuffer);
            }
            Log.w(TAG, "[INIT] ckpt6 SECOND_INSTANCE init done (or skipped)");

            // ---------- 调试：构造完成后强制跑一张全黑图，确保 FIRST_INFER 日志必出 ----------
            //   （用于反推 _opt 模型的 box 编码/归一化/conf 通道，不需要任何权限/投屏）
            try {
                int probeW = 640, probeH = 640;
                int[] px = new int[probeW * probeH]; // 默认为 0=ARGB全黑
                Bitmap black = Bitmap.createBitmap(probeW, probeH, Bitmap.Config.ARGB_8888);
                black.setPixels(px, 0, probeW, 0, 0, probeW, probeH);
                Log.w(TAG, "[INIT] ckpt7 probe estimate start (black 640x640)...");
                estimate(black);
                black.recycle();
                Log.w(TAG, "[INIT] ckpt8 probe estimate OK");
            } catch (Throwable t) {
                Log.e(TAG, "[INIT_PROBE_EXCEPTION] probe failed", t);
            }
        } catch (Throwable t) {
            Log.e(TAG, "[INIT_FATAL] exception during tensor shape/array init", t);
            try { if (interpreter != null) interpreter.close(); } catch (Throwable ignored) {}
            closeHardwareDelegate();
            interpreter = null;
            throw new IOException("Shape/output init failed", t);
        }
    }

    /**
     * 从 SharedPreferences 读取高级参数，无值则保留默认
     * key 与 AdvancedOptionsActivity 中定义一致
     */
    private void loadPrefs(Context context) {
        SharedPreferences prefs = context.getSharedPreferences("fplus_settings", Context.MODE_PRIVATE);
        // 每个参数单独 try-catch：避免一个坏值连累后续所有合法参数丢失用户配置
        CONFIDENCE_THRESHOLD = parseFloat(prefs, "det_conf", 0.15f, 0.05f, 0.5f);
        VALID_DETECTION_CONF = parseFloat(prefs, "det_valid", 0.2f, 0.1f, 0.5f);
        MIN_AREA_THRESHOLD = parseFloat(prefs, "det_min_area", 0.01f, 0.001f, 0.1f);
        TRACK_IOU_THRESHOLD = parseFloat(prefs, "track_iou", 0.2f, 0f, 1f);
        CBIoU_BUF_HIGH = parseFloat(prefs, "track_buf_high", 0.3f, 0f, 1f);
        CBIoU_BUF_LOW = parseFloat(prefs, "track_buf_low", 0.5f, 0f, 1f);
        CBIoU_IOU_LOW = parseFloat(prefs, "track_iou_low", 0.3f, 0f, 1f);
        CBIoU_SPEED_REF = parseFloat(prefs, "track_speed_ref", 0.25f, 0.1f, 0.6f);
        DIST_WEIGHT = parseFloat(prefs, "track_dist_weight", 0.25f, 0f, 0.5f);
        OCM_WEIGHT = parseFloat(prefs, "ocm_weight", 0.15f, 0f, 0.5f);
        OCM_SPEED_REF = parseFloat(prefs, "ocm_speed_ref", 0.01f, 0.001f, 0.1f);
        ACC_THRESHOLD = parseFloat(prefs, "track_acc_threshold", 0.002f, 0f, 0.02f);
        MAX_TRACK_LOST = parseInt(prefs, "track_max_lost", 30, 5, 60);
        TAKEOVER_CONF = parseFloat(prefs, "track_takeover", 0.5f, 0.3f, 0.9f);
        PREDICT_SECONDS = parseFloat(prefs, "pred_seconds", 0.083f, 0f, 0.2f);
        // 绿框预判：默认开启（"1"/空值均视为开，"0"才关），与 UI "默认打开"一致
        LEAD_PREDICT_ENABLED = !"0".equals(prefs.getString("lead_predict", "1"));
        // CMC：默认开启（"1"/空值 均视为开，"0"才关），与 UI "默认打开"一致
        CMC_ENABLED = !"0".equals(prefs.getString("cmc_enabled", "1"));
        COAST_MIN_VEL = parseFloat(prefs, "coast_min_vel", 0.01f, 0.001f, 0.02f);
        COAST_MAX_FRAMES = parseInt(prefs, "coast_max_frames", 2, 1, 30);
        CENTER_SIGMA = parseFloat(prefs, "score_center_sigma", 0.2f, 0.1f, 0.5f);
        SCORE_W = parseFloat(prefs, "score_w", 0.3f, 0f, 1f);
        ROI_SCALE = parseFloat(prefs, "roi_scale", 0.7f, 0.3f, 1.0f);
        // bright_auto：默认开启（"1"/空值 均视为开，"0"才关），与 UI "默认打开"一致
        String autoStr = prefs.getString("bright_auto", "1");
        BRIGHT_AUTO_ENABLED = !"0".equals(autoStr);
        // 目标锁定：默认关闭（"1"开启，"0"关闭）
        String lockStr = prefs.getString("target_lock", "0");
        TARGET_LOCK_ENABLED = "1".equals(lockStr);
        USER_BRIGHT_GAIN = parseFloat(prefs, "bright_gain", 1.3f);
        USER_BRIGHT_OFFSET = parseFloat(prefs, "bright_offset", 25f);
        CUR_BRIGHT_GAIN = USER_BRIGHT_GAIN;
        CUR_BRIGHT_OFFSET = USER_BRIGHT_OFFSET;
        CUR_BRIGHT_OFFSET_NORM = CUR_BRIGHT_OFFSET * INV_255;
        emaP90 = -1f;                  // 重置 EMA，下一帧按新场景重新收敛
        shadowLift = 0f;
        lastRebuildShadow = -1f;
        shadowLiftLogged = false;
        lastRebuildGain = -1f;         // 下一次必触发 rebuildBrightLut（用新 CUR_*）
        lastRebuildOffset = -1f;
        rebuildBrightLut();  // 亮度参数变化后重建 LUT，convert 遍历用查表替代浮点乘加
        // 双实例并发 PoC 开关（"1"=开），zoom 比例；交叉核验仅在双路开时生效
        DUAL_INFER_ENABLED = "1".equals(prefs.getString("dual_infer", "0"));
        DUAL_VERIFY_ENABLED = DUAL_INFER_ENABLED && "1".equals(prefs.getString("dual_verify", "0"));
        DUAL_ZOOM = parseFloat(prefs, "dual_zoom", 0.5f, 0.3f, 0.8f);
        // 抗准心误识别：独立开关，默认关
        ANTI_CROSSHAIR_ENABLED = "1".equals(prefs.getString("anti_crosshair", "0"));
        antiCrosshairWarned = false;
    }

    private float parseFloat(SharedPreferences prefs, String key, float def) {
        try {
            return Float.parseFloat(prefs.getString(key, Float.toString(def)));
        } catch (NumberFormatException e) {
            Log.w(TAG, "参数 " + key + " 解析失败，使用默认值 " + def);
            return def;
        }
    }

    /** 解析并收敛到 [min,max]：防御 prefs 被外部写入非法值（如 ROI_SCALE 越界导致 drawBitmap 崩溃） */
    private float parseFloat(SharedPreferences prefs, String key, float def, float min, float max) {
        float v = parseFloat(prefs, key, def);
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }

    private int parseInt(SharedPreferences prefs, String key, int def) {
        try {
            return Integer.parseInt(prefs.getString(key, Integer.toString(def)));
        } catch (NumberFormatException e) {
            Log.w(TAG, "参数 " + key + " 解析失败，使用默认值 " + def);
            return def;
        }
    }

    /** 解析并收敛到 [min,max]：防御 prefs 被写入非法值（如 track_max_lost=0 导致状态机退化） */
    private int parseInt(SharedPreferences prefs, String key, int def, int min, int max) {
        int v = parseInt(prefs, key, def);
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }

    private static Backend[] fallbackOrder(Backend preferred) {
        switch (preferred) {
            case NNAPI:
                return new Backend[]{Backend.NNAPI, Backend.GPU, Backend.CPU};
            case GPU:
                return new Backend[]{Backend.GPU, Backend.CPU};
            case CPU:
            default:
                return new Backend[]{Backend.CPU};
        }
    }

    private boolean tryCreateInterpreter(MappedByteBuffer modelBuffer, Backend backend) {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        // 禁用 XNNPACK delegate fallback：GPU delegate 不受支持的算子回落到 XNNPACK CPU 反而更慢
        // （TFLite 2.14+ 默认 XNNPACK，与 GPU delegate 叠加时可能拖慢；显式 false 强制纯 CPU fallback）
        options.setUseXNNPACK(false);

        if (backend == Backend.GPU) {
            try {
                CompatibilityList compatibilityList = new CompatibilityList();
                if (!compatibilityList.isDelegateSupportedOnThisDevice()) {
                    Log.w(TAG, "GPU delegate 不受支持");
                    return false;
                }
                // 显式配置 GpuDelegate.Options：避免不同 TFLite 版本默认值差异
                GpuDelegate.Options gOpts = new GpuDelegate.Options();
                // FAST_SINGLE_ANSWER：低延迟优先（SUSTAINED_SPEED 反而增加小模型开销，项目记忆已确认）
                gOpts.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER);
                // 允许 FP16 精度损失（GPU 内部 FP32→FP16）：高通 GPU FP16 单元是 FP32 的 2-4 倍，
                // 速度提升 ~50%+，YOLO 检测精度损失 <1% mAP，完全可接受。
                // v4.10 误设为 false 导致 infer 从 30ms 涨到 45ms，反向优化。
                gOpts.setPrecisionLossAllowed(true);
                gpuDelegate = new GpuDelegate(gOpts);
                options.addDelegate(gpuDelegate);
                Log.d(TAG, "GPU delegate: FAST_SINGLE_ANSWER + PrecisionLoss=on(FP16) + XNNPACK=off");
            } catch (Throwable t) {
                Log.w(TAG, "创建 GPU delegate 失败", t);
                closeHardwareDelegate();
                return false;
            }
        } else if (backend == Backend.NNAPI) {
            try {
                nnApiDelegate = new NnApiDelegate();
                options.addDelegate(nnApiDelegate);
            } catch (Throwable t) {
                Log.w(TAG, "创建 NNAPI delegate 失败", t);
                closeHardwareDelegate();
                return false;
            }
        }
        // CPU 不需要 delegate

        modelBuffer.rewind();
        try {
            interpreter = new Interpreter(modelBuffer, options);
            return true;
        } catch (Throwable t) {
            Log.w(TAG, backend + " 模型加载失败", t);
            closeHardwareDelegate();
            interpreter = null;
            return false;
        }
    }

    /** 双实例第二路 Interpreter（独立 GpuDelegate，与主路命令队列隔离，可并发） */
    private boolean tryCreateInterpreter2(MappedByteBuffer modelBuffer) {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        options.setUseXNNPACK(false);
        try {
            CompatibilityList cl = new CompatibilityList();
            if (!cl.isDelegateSupportedOnThisDevice()) {
                Log.w(TAG, "[2nd] GPU delegate 不受支持，双实例禁用");
                return false;
            }
            GpuDelegate.Options gOpts = new GpuDelegate.Options();
            gOpts.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER);
            gOpts.setPrecisionLossAllowed(true);
            gpuDelegate2 = new GpuDelegate(gOpts);
            options.addDelegate(gpuDelegate2);
            Log.d(TAG, "[2nd] GPU delegate: FAST_SINGLE_ANSWER + PrecisionLoss=on(FP16) + XNNPACK=off");
        } catch (Throwable t) {
            Log.w(TAG, "[2nd] 创建 GPU delegate 失败", t);
            closeSecondHardwareDelegate();
            return false;
        }
        modelBuffer.rewind();
        try {
            interpreter2 = new Interpreter(modelBuffer, options);
            Log.d(TAG, "[2nd] GPU interpreter 初始化成功");
            return true;
        } catch (Throwable t) {
            Log.w(TAG, "[2nd] 模型加载失败", t);
            closeSecondHardwareDelegate();
            interpreter2 = null;
            return false;
        }
    }

    /** 初始化第二套输入/输出/绘制资源与并发执行器 */
    private void initSecondInstance(MappedByteBuffer modelBuffer) {
        if (!tryCreateInterpreter2(modelBuffer)) {
            Log.w(TAG, "双实例第二路初始化失败，回退单实例");
            return;
        }
        inputBuffer2 = ByteBuffer.allocateDirect(1 * 3 * inputSize * inputSize * 4);
        inputBuffer2.order(ByteOrder.nativeOrder());
        inputBuffer2.rewind();
        inputFloatBuffer2 = inputBuffer2.asFloatBuffer();
        // 与主路对称：多输出模式下分别分配 boxes/scores 缓冲，单输出模式沿用 output2
        if (multiOutputMode) {
            int[] s0 = interpreter.getOutputTensor(0).shape();
            int[] s1 = interpreter.getOutputTensor(1).shape();
            outputBoxes2 = new float[s0[0]][s0[1]][s0[2]];
            outputScores2 = new float[s1[0]][s1[1]][s1[2]];
        } else {
            output2 = new float[output.length][output[0].length][output[0][0].length];
        }
        scaledRoi2 = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
        scaleCanvas2 = new Canvas(scaledRoi2);
        scalePaint2 = new Paint(Paint.FILTER_BITMAP_FLAG);
        scaleRect2 = new Rect(0, 0, inputSize, inputSize);
        scaleSrcRect2 = new Rect();
        secondExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(() -> {
                // Android 线程分类依赖 Process.setThreadPriority（Linux nice），
                // 仅 Thread.MAX_PRIORITY 只改 VM 层调度顺序，对系统温控/降频无效。
                // 必须在线程启动后的 run() 内部调用才能生效，且可能被安全策略拒绝 → 吞异常
                try { Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_DISPLAY); }
                catch (Throwable ignored) { /* 没权限时退回默认优先级 */ }
                r.run();
            }, "SecondInferThread");
            // 保留 Java 层优先级做兜底
            t.setPriority(Thread.MAX_PRIORITY);
            return t;
        });
        Log.d(TAG, "双实例并发已启用，DUAL_ZOOM=" + DUAL_ZOOM);
    }

    private void closeSecondHardwareDelegate() {
        if (gpuDelegate2 != null) {
            gpuDelegate2.close();
            gpuDelegate2 = null;
        }
    }

    public PersonPose estimate(Bitmap bitmap) {
        estimateCallCount++;
        originalWidth = bitmap.getWidth();
        originalHeight = bitmap.getHeight();
        // 动态亮度：每帧重置标记，主路 bitmapToByteBuffer 只计算一次（第二路复用主路结果）
        dynBrightComputed = false;

        long t0 = System.nanoTime();
        Bitmap resizedBitmap = preprocessBitmap(bitmap);
        long t1 = System.nanoTime();
        bitmapToByteBuffer(resizedBitmap);
        long t2 = System.nanoTime();

        // 双实例并发：提交第二路（中心放大区），主线程同时跑主路 infer
        Future<PersonPose> f2 = null;
        long dualStart = 0L;
        if (DUAL_INFER_ENABLED && secondExecutor != null && interpreter2 != null) {
            dualStart = System.nanoTime();
            f2 = secondExecutor.submit(() -> runSecondInference(bitmap));
        }

        if (multiOutputMode) {
            java.util.Map<Integer, Object> outputs = new java.util.HashMap<>();
            outputs.put(0, outputBoxes);
            outputs.put(1, outputScores);
            interpreter.runForMultipleInputsOutputs(new Object[]{inputBuffer}, outputs);
        } else {
            interpreter.run(inputBuffer, output);
        }
        // ---------- 第一帧：WARN 级打前10个anchor的原始box/conf数值，便于定位编码错 ----------
        if (!firstInferLogged) {
            firstInferLogged = true;
            StringBuilder sb = new StringBuilder(1200);
            sb.append("[FIRST_INFER] top10 anchors raw output (multiOutput=").append(multiOutputMode)
                    .append(" boxIsXyxy=").append(boxIsXyxy).append(" chFirst=").append(channelFirst)
                    .append(" numChannels=").append(numChannels).append(" inputSize=").append(inputSize)
                    .append("):\n");
            int showCount = Math.min(10, numAnchors);
            for (int i = 0; i < showCount; i++) {
                float v0 = getOutputValue(0, i);
                float v1 = getOutputValue(1, i);
                float v2 = getOutputValue(2, i);
                float v3 = getOutputValue(3, i);
                // 把后面所有通道的值都打出来（nc<=5的话很紧凑），看清楚 conf 到底在第几通道
                java.util.List<String> extras = new java.util.ArrayList<>();
                float maxConf = 0f;
                int maxC = -1;
                for (int c = 4; c < numChannels; c++) {
                    float cf = getOutputValue(c, i);
                    extras.add("c" + c + "=" + fmt4(cf));
                    if (cf > maxConf) { maxConf = cf; maxC = c; }
                }
                sb.append("  anchor#").append(i).append(": [0]=").append(fmt4(v0)).append(" [1]=").append(fmt4(v1))
                        .append(" [2]=").append(fmt4(v2)).append(" [3]=").append(fmt4(v3))
                        .append("  ").append(TextUtils.join(" ", extras))
                        .append("  maxConf=").append(fmt4(maxConf)).append("(ch=").append(maxC).append(")\n");
            }
            // ---------- 启发式自检：如果 c4..end 全 < 0 或 maxConf < 0.01 并且 [0]~[3] 明显是像素值
            //            → 证明 channelFirst 判反了！真实 shape[1,6,2100] 其实是 [1,N,F] 被我按 [1,F,N] 读了
            //            → 临时翻转读一次，对比 conf 数值范围
            if (numChannels >= 5) {
                float chRangeMax = -Float.MAX_VALUE;
                float chRangeMin = Float.MAX_VALUE;
                for (int c = 4; c < numChannels; c++) {
                    for (int i = 0; i < Math.min(100, numAnchors); i++) {
                        float v = getOutputValue(c, i);
                        if (v > chRangeMax) chRangeMax = v;
                        if (v < chRangeMin) chRangeMin = v;
                    }
                }
                sb.append("  [CHECK ch4..end] min=").append(fmt4(chRangeMin)).append(" max=").append(fmt4(chRangeMax))
                        .append("  (合理范围应为 0~1，sigmoid置信度)\n");
                // 额外打每个 conf 通道前 10 个 anchor 的实际数值，不做翻转读（数组维度不匹配）
                try {
                    java.util.List<String> perCh = new java.util.ArrayList<>();
                    int dispN = Math.min(10, numAnchors);
                    for (int c = 4; c < numChannels; c++) {
                        StringBuilder cb = new StringBuilder();
                        cb.append("ch").append(c).append("=[");
                        for (int i = 0; i < dispN; i++) {
                            if (i > 0) cb.append(",");
                            cb.append(fmt4(getOutputValue(c, i)));
                        }
                        cb.append("]");
                        perCh.add(cb.toString());
                    }
                    sb.append("  [CONF channels] ").append(TextUtils.join(" ", perCh)).append("\n");
                } catch (Throwable t) {
                    sb.append("  [CONF channels] SKIPPED ex=").append(t.getClass().getSimpleName()).append("\n");
                }
            }
            Log.w(TAG, sb.toString());
        }
        long t3 = System.nanoTime();
        PersonPose pose = parseOutput();
        long t4 = System.nanoTime();

        preprocessAccum += t1 - t0;
        convertAccum += t2 - t1;
        inferAccum += t3 - t2;
        parseAccum += t4 - t3;
        timingFrames++;
        if (timingFrames >= TIMING_WINDOW) {
            Log.d(TAG, String.format(
                    "阶段耗时 pre=%.1fms convert=%.1fms infer=%.1fms parse=%.1fms 合计=%.1fms roi=(%d,%d,%d)",
                    preprocessAccum / (double) timingFrames / 1_000_000,
                    convertAccum / (double) timingFrames / 1_000_000,
                    inferAccum / (double) timingFrames / 1_000_000,
                    parseAccum / (double) timingFrames / 1_000_000,
                    (preprocessAccum + convertAccum + inferAccum + parseAccum) / (double) timingFrames / 1_000_000,
                    roiX, roiY, roiSize));
            preprocessAccum = 0;
            convertAccum = 0;
            inferAccum = 0;
            parseAccum = 0;
            timingFrames = 0;
        }

        // 第二路结果收集 + 并发性能对比日志 + 可选交叉核验降权（仅 DUAL_VERIFY_ENABLED=开时生效）
        PersonPose pose2 = null;
        if (f2 != null) {
            try {
                pose2 = f2.get();
                long dualEnd = System.nanoTime();
                long mainInferMs = (t3 - t2) / 1_000_000;
                long secondInferMs = secondInferNanos.get() / 1_000_000;
                long dualTotalMs = (dualEnd - dualStart) / 1_000_000;
                dualMainInferAccum += mainInferMs;
                dual2ndInferAccum += secondInferMs;
                dualTotalAccum += dualTotalMs;
                dualFrames++;
                if (DUAL_VERIFY_ENABLED && pose != null && pose.box != null) {
                    // ===== 实验：双路交叉核验降权（只惩罚置信度，不硬移除，lock 不受直接影响）=====
                    // 判定范围：候选 box 中心需在「第二路实际覆盖的正方形 zoomRect」内（否则第二路本就没该处信息，跳过）
                    float bCx = pose.box[0];
                    float bCy = pose.box[1];
                    if (bCx >= zoomX && bCx <= zoomX + zoomSize
                            && bCy >= zoomY && bCy <= zoomY + zoomSize
                            && pose2 != null && pose2.allValidBoxes != null && !pose2.allValidBoxes.isEmpty()) {
                        boolean matched = false;
                        float bw = pose.box[2];
                        float bh = pose.box[3];
                        float maxDist = (originalWidth + originalHeight) * 0.5f * 0.035f;  // 归一化≈3.5%对角线
                        for (float[] b2 : pose2.allValidBoxes) {
                            float b2Cx = b2[0], b2Cy = b2[1], b2W = b2[2], b2H = b2[3];
                            // 距离判定
                            float ddx = bCx - b2Cx, ddy = bCy - b2Cy;
                            if (ddx * ddx + ddy * ddy <= maxDist * maxDist) { matched = true; break; }
                            // IoU 判定（轻量：先算 AABB，C-BIoU 标准）
                            float b1X1 = bCx - bw * 0.5f, b1Y1 = bCy - bh * 0.5f;
                            float b1X2 = bCx + bw * 0.5f, b1Y2 = bCy + bh * 0.5f;
                            float b2X1 = b2Cx - b2W * 0.5f, b2Y1 = b2Cy - b2H * 0.5f;
                            float b2X2 = b2Cx + b2W * 0.5f, b2Y2 = b2Cy + b2H * 0.5f;
                            float ix = Math.max(0f, Math.min(b1X2, b2X2) - Math.max(b1X1, b2X1));
                            float iy = Math.max(0f, Math.min(b1Y2, b2Y2) - Math.max(b1Y1, b2Y1));
                            float inter = ix * iy;
                            if (inter <= 0f) continue;
                            float union = bw * bh + b2W * b2H - inter;
                            float iou = union > 1e-6f ? inter / union : 0f;
                            if (iou >= 0.22f) { matched = true; break; }
                        }
                        if (!matched) {
                            // 只做降权（45% 保留），不到 0.25 会低于 HIGH 阈值，无法即时接管 lock
                            // 惩罚后在整个 track 流程里被同等对待，但接管 lock 需要 TAKEOVER_CONF=0.5 更难
                            float oldConf = pose.conf;
                            float newConf = oldConf * 0.45f;
                            pose.conf = newConf;
                            if (!dualVerifyWarned) {
                                dualVerifyWarned = true;
                                Log.i(TAG, "[dual_verify] 命中FP惩罚: 主候选("
                                        + (int) bCx + "," + (int) bCy + "," + (int) bw + "," + (int) bh
                                        + ") conf=" + fmt4(oldConf) + "→" + fmt4(newConf)
                                        + " zoomRect=[" + zoomX + "," + zoomY + "," + zoomSize + "]"
                                        + " 第二路候选数=" + pose2.allValidBoxes.size());
                            }
                        }
                    }
                }
                if (dualFrames >= 30) {
                    Log.d(TAG, String.format(
                            "[dual] 主路infer=%.1fms 第二路infer=%.1fms 并发总=%.1fms 第二路检出=%s",
                            dualMainInferAccum / (double) dualFrames,
                            dual2ndInferAccum / (double) dualFrames,
                            dualTotalAccum / (double) dualFrames,
                            (pose2 != null ? "有" : "无")));
                    dualMainInferAccum = 0;
                    dual2ndInferAccum = 0;
                    dualTotalAccum = 0;
                    dualFrames = 0;
                }
            } catch (Exception e) {
                Log.w(TAG, "[2nd] 收集失败", e);
            }
        }

        // ===== 抗准心误识别（实验选项·默认关）：命中3条则 conf × 0.35 软惩罚 =====
        if (ANTI_CROSSHAIR_ENABLED && pose != null && pose.box != null && originalWidth > 0 && originalHeight > 0) {
            float px = pose.box[0];
            float py = pose.box[1];
            float pw = pose.box[2];
            float ph = pose.box[3];
            if (pw > 1e-3f) {
                // 3条同时命中才惩罚：
                // (1) 距屏幕几何中心 < 1.2% 对角线
                float cxC = originalWidth * 0.5f;
                float cyC = originalHeight * 0.5f;
                float ddx = px - cxC, ddy = py - cyC;
                float diag = (originalWidth + originalHeight) * 0.5f;
                boolean nearCenter = (ddx * ddx + ddy * ddy) <= (AC_DIST_RATIO * diag) * (AC_DIST_RATIO * diag);
                if (nearCenter) {
                    // (2) 高宽比不在真人范围（人 h/w ≈ 1.7~3.5；准心/瞄具通常 0.8~1.5 或 > 3.6 细长瞄杆）
                    float hw = ph / pw;
                    boolean badRatio = hw < AC_HW_MIN || hw > AC_HW_MAX;
                    if (badRatio) {
                        // (3) 超小尺寸：宽 < 2.2% 屏幕宽 且 高 < 2.8% 屏幕高
                        boolean tiny = (pw < originalWidth * AC_W_RATIO_MAX) && (ph < originalHeight * AC_H_RATIO_MAX);
                        if (tiny) {
                            float oldConf = pose.conf;
                            float newConf = oldConf * AC_PENALTY;
                            pose.conf = newConf;
                            if (!antiCrosshairWarned) {
                                antiCrosshairWarned = true;
                                String why = " dist%=" + fmt4((float) Math.sqrt(ddx * ddx + ddy * ddy) / diag)
                                        + " h/w=" + fmt4(hw)
                                        + " w%=" + fmt4(pw / originalWidth)
                                        + " h%=" + fmt4(ph / originalHeight);
                                Log.i(TAG, "[anti_crosshair] 命中准心惩罚: 候选("
                                        + (int) px + "," + (int) py + "," + (int) pw + "," + (int) ph
                                        + ") conf=" + fmt4(oldConf) + "→" + fmt4(newConf)
                                        + " 原因=" + why);
                            }
                        }
                    }
                }
            }
        }

        if (pose != null && pose.box != null) {
            // parseOutput 返回原图像素坐标 [cx, cy, w, h]
            float px = pose.box[0];
            float py = pose.box[1];
            float pw = pose.box[2];
            float ph = pose.box[3];

            // ROI 跟随：下一帧 ROI 中心移到目标中心，避免主角移出 ROI 而丢失
            // 传入目标宽度：死区按目标大小反比缩放（方案1），大目标死区收紧防裁剪
            updateRoiCenter(px, py, pw);

            // 转成相对原图的归一化坐标
            pose.box[0] = px / originalWidth;
            pose.box[1] = py / originalHeight;
            pose.box[2] = pw / originalWidth;
            pose.box[3] = ph / originalHeight;
        } else {
            // 未检测到目标：ROI 缓慢回中，重新搜索
            recenterRoi();
        }

        return pose;
    }

    private void recenterRoi() {
        float centerX = originalWidth / 2f;
        float centerY = originalHeight / 2f;
        float curCx = roiX + roiSize / 2f;
        float curCy = roiY + roiSize / 2f;
        float newCx = curCx + (centerX - curCx) * RECENTER_SMOOTH;
        float newCy = curCy + (centerY - curCy) * RECENTER_SMOOTH;
        // 足够接近中心时直接吸附，避免整数/浮点截断导致黄圈停在偏心几个像素
        if (Math.abs(centerX - newCx) < 1f) newCx = centerX;
        if (Math.abs(centerY - newCy) < 1f) newCy = centerY;
        roiX = (int) Math.max(0, Math.min(originalWidth - roiSize, newCx - roiSize / 2f));
        roiY = (int) Math.max(0, Math.min(originalHeight - roiSize, newCy - roiSize / 2f));
    }

    private void updateRoiCenter(float targetCx, float targetCy, float targetW) {
        float curCx = roiX + roiSize / 2f;
        float curCy = roiY + roiSize / 2f;
        float dx = targetCx - curCx;
        float dy = targetCy - curCy;
        float dist = (float) Math.sqrt(dx * dx + dy * dy);

        // 死区：目标在 ROI 中心 8% 范围内不移动，减少黄框抖动。
        // 但大目标（框宽接近 ROI 宽）在死区内自由移动即伸出 ROI 边缘被裁剪，
        // 且缩放滞后让裁切周期性重演 → 跟踪速度被低估、绿框滞后。
        // 修正：死区随目标宽度反比缩放——宽 ≤60% ROI 维持 8% 防抖不变；
        // 再大则收敛到 0%（宽 ≥ROI 时死区消失，ROI 紧跟目标防裁剪）。小目标防抖不受影响。
        float deadZone = roiSize * 0.08f;
        if (targetW > 1e-3f) {
            deadZone *= Math.min(1f, roiSize * 0.6f / targetW);
        }
        if (dist <= deadZone) return;

        // 非线性跟随：超出死区部分按平方根缩放
        // 目标离中心越远跟随幅度越大，但始终小于实际位移；单帧最大移动 35% ROI
        float excess = dist - deadZone;
        float ratio = excess / dist;
        float scale = (float) Math.sqrt(ratio);
        float maxMove = roiSize * 0.35f;
        float move = Math.min(excess * scale, maxMove);
        float moveX = dx / dist * move;
        float moveY = dy / dist * move;

        float newCx = curCx + moveX;
        float newCy = curCy + moveY;
        roiX = (int) Math.max(0, Math.min(originalWidth - roiSize, newCx - roiSize / 2f));
        roiY = (int) Math.max(0, Math.min(originalHeight - roiSize, newCy - roiSize / 2f));
    }

    private Bitmap preprocessBitmap(Bitmap source) {
        int sw = source.getWidth();
        int sh = source.getHeight();
        int newRoiSize = (int) (Math.min(sw, sh) * ROI_SCALE);

        // 仅在画面尺寸变化（如横竖屏切换）时重置 ROI 尺寸与中心；其余帧由 updateRoiCenter 跟随目标
        if (roiSize != newRoiSize) {
            roiSize = newRoiSize;
            roiX = (sw - roiSize) / 2;
            roiY = (sh - roiSize) / 2;
        }

        if (scaledRoi == null) {
            scaledRoi = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
            scaleCanvas = new Canvas(scaledRoi);
            scalePaint = new Paint(Paint.FILTER_BITMAP_FLAG);
            scaleRect = new Rect(0, 0, inputSize, inputSize);
            scaleSrcRect = new Rect();
        }
        // 直接从源矩形缩放绘制，避免每帧 createBitmap/recycle 的开销
        scaleSrcRect.set(roiX, roiY, roiX + roiSize, roiY + roiSize);
        scaleCanvas.drawBitmap(source, scaleSrcRect, scaleRect, scalePaint);
        return scaledRoi;
    }

    private void bitmapToByteBuffer(Bitmap bitmap) {
        if (brightLut == null) rebuildBrightLut();  // 兜底：构造异常未初始化 LUT 时补建
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int len = width * height;
        int byteLen = len * 4;
        if (pixelBytes == null || pixelBytes.length != byteLen) {
            pixelBytes = new byte[byteLen];
        }
        // copyPixelsToBuffer 经 wrap 一次 memcpy 直入 heap byte[]（native RGBA 字节序与 heap 一致），
        // 省掉 DirectByteBuffer 中转 + 二次 bulk get 的整帧拷贝；跳过 getPixels 的逐像素 JNI int 打包
        ByteBuffer wrap = ByteBuffer.wrap(pixelBytes, 0, byteLen).order(ByteOrder.nativeOrder());
        bitmap.copyPixelsToBuffer(wrap);

        // ===== 动态亮度增益：根据 BRIGHT_AUTO_ENABLED 开关选路径 =====
        //   开：抽样直方图 → P90 场景亮度(EMA) → mult → 暗部抬升 → 超阈值重建 LUT
        //   关：固定 mult=1.0、shadowLift=0，直接使用 USER_* 基础增益/偏移，不做任何环境光计算
        if (!dynBrightComputed) {
            if (!BRIGHT_AUTO_ENABLED) {
                // 关闭环境光优化：只使用用户固定设置，保证 mult=1 且不抬升暗部
                CUR_BRIGHT_GAIN = USER_BRIGHT_GAIN;
                CUR_BRIGHT_OFFSET = USER_BRIGHT_OFFSET;
                CUR_BRIGHT_OFFSET_NORM = CUR_BRIGHT_OFFSET * INV_255;
                emaP90 = -1f;
                shadowLift = 0f;
                shadowLiftLogged = false;
                // 变化过小不重建（首次 lastRebuild*=-1 会进入一次）
                boolean gainChanged = (lastRebuildGain < 0f)
                        || (Math.abs(lastRebuildGain) < 1e-4f
                            ? Math.abs(CUR_BRIGHT_GAIN - lastRebuildGain) > REBUILD_GAIN_RATIO * 0.5f
                            : Math.abs(CUR_BRIGHT_GAIN / lastRebuildGain - 1f) > REBUILD_GAIN_RATIO);
                boolean offsetChanged = Math.abs(CUR_BRIGHT_OFFSET - lastRebuildOffset) > REBUILD_OFFSET_ABS;
                boolean shadowChanged = Math.abs(shadowLift - lastRebuildShadow) > SHADOW_REBUILD_ABS;
                if (gainChanged || (lastRebuildOffset < 0f) || offsetChanged || shadowChanged) {
                    rebuildBrightLut();
                    lastRebuildGain = CUR_BRIGHT_GAIN;
                    lastRebuildOffset = CUR_BRIGHT_OFFSET;
                    lastRebuildShadow = shadowLift;
                }
            } else {
                // ===== 方案B：P90 高分位替代平均亮度（亮环境+黑物体时 P90 仍高 → 不提亮，真暗环境 P90 低 → 提亮）；
                // 再用 EVdiff（亮部12%均值 − 暗部12%均值）检测高对比场景，触发暗部抬升曲线看清黑物体。
                int[] hist = new int[LUM_HIST_BINS];  // 32 桶亮度直方图（每桶 8 级）
                int sampleCount = 0;
                int rowStep = 8;  // 每 8 行抽 1 行，每 8 列抽 1 列 = 抽样率 1/64，
                int colStep = 8;  //   对应 EMA 收敛速度仍远快于画面整体亮度变化，同时比原 1/16 省 3/4 的整数乘加
                int w4 = width * 4;
                for (int y = 0; y < height; y += rowStep) {
                    int rowOff = y * w4;
                    for (int x = 0; x < width; x += colStep) {
                        int idx = rowOff + x * 4;
                        int r = pixelBytes[idx] & 0xFF;
                        int g = pixelBytes[idx + 1] & 0xFF;
                        int b = pixelBytes[idx + 2] & 0xFF;
                        // 标准 CCIR 601 luma = 0.299R + 0.587G + 0.114B（用整数乘避免浮点）
                        int l = (299 * r + 587 * g + 114 * b) / 1000;
                        hist[l >> 3]++;  // 0~255 → 32 桶
                        sampleCount++;
                    }
                }
                if (sampleCount > 0) {
                    // P90：从高到低累计到 90% 样本 → 场景真实亮度（黑物体占 <90% 时不影响）
                    int acc = 0;
                    int p90 = 128;
                    int p90Target = (int) (sampleCount * 0.9f);
                    for (int b = LUM_HIST_BINS - 1; b >= 0; b--) {
                        acc += hist[b];
                        if (acc >= p90Target) { p90 = b * 8 + 4; break; }  // 取桶中值
                    }
                    // 亮部 12% 与暗部 12% 的加权平均 → EVdiff 对比度标尺
                    long brightSum = 0;
                    int brightCnt = 0;
                    long darkSum = 0;
                    int darkCnt = 0;
                    int tailTarget = Math.max(1, (int) (sampleCount * 0.12f));
                    acc = 0;
                    for (int b = LUM_HIST_BINS - 1; b >= 0; b--) {
                        int cnt = hist[b];
                        if (cnt <= 0) continue;
                        int take = Math.min(cnt, tailTarget - acc);
                        brightSum += (long) (b * 8 + 4) * take;
                        brightCnt += take;
                        acc += take;
                        if (acc >= tailTarget) break;
                    }
                    acc = 0;
                    for (int b = 0; b < LUM_HIST_BINS; b++) {
                        int cnt = hist[b];
                        if (cnt <= 0) continue;
                        int take = Math.min(cnt, tailTarget - acc);
                        darkSum += (long) (b * 8 + 4) * take;
                        darkCnt += take;
                        acc += take;
                        if (acc >= tailTarget) break;
                    }
                    int brightAvg = brightCnt > 0 ? (int) (brightSum / brightCnt) : 128;
                    int darkAvg = darkCnt > 0 ? (int) (darkSum / darkCnt) : 0;
                    int evDiff = brightAvg - darkAvg;

                    // P90 场景亮度 EMA 平滑，防止帧间小抖动导致 GAIN 跳变
                    if (emaP90 < 0f) emaP90 = p90;
                    else emaP90 = emaP90 + LUM_EMA_ALPHA * (p90 - emaP90);
                    // 补偿倍率：想把 P90 拉到 TARGET_P90_LUMA，暗画面 mult > 1，亮画面 mult < 1
                    float mult = TARGET_P90_LUMA / (emaP90 + 8f);
                    if (mult < MIN_MULT) mult = MIN_MULT;
                    else if (mult > MAX_MULT) mult = MAX_MULT;
                    CUR_BRIGHT_GAIN = USER_BRIGHT_GAIN * mult;
                    CUR_BRIGHT_OFFSET = USER_BRIGHT_OFFSET * mult;
                    CUR_BRIGHT_OFFSET_NORM = CUR_BRIGHT_OFFSET * INV_255;

                    // 暗部抬升强度：场景亮(P90 高)且对比度大(亮背景+黑物体)时，按 EVdiff 超限比例线性增强
                    float targetShadow = 0f;
                    if (emaP90 >= SHADOW_LIFT_BRIGHT_MIN && evDiff >= SHADOW_LIFT_EVDIFF_MIN) {
                        float over = (evDiff - SHADOW_LIFT_EVDIFF_MIN) / 120f;  // EVdiff 每超 120 满格
                        if (over > 1f) over = 1f;
                        targetShadow = SHADOW_LIFT_MAX * over;
                    }
                    shadowLift = shadowLift + SHADOW_LIFT_EMA_ALPHA * (targetShadow - shadowLift);
                    if (shadowLift < 0.01f) shadowLift = 0f;  // 小值归零，避免微弱抬升常驻
                    if (shadowLift > 0f && !shadowLiftLogged) {
                        shadowLiftLogged = true;
                        Log.i(TAG, "暗部抬升激活: P90=" + (int) emaP90
                                + " brightAvg=" + brightAvg + " darkAvg=" + darkAvg
                                + " EVdiff=" + evDiff + " lift=" + shadowLift);
                    }
                }
                // 变化过小不重建（256次浮点虽快，但没必要每帧做）
                // lastRebuild* < 0 表示首次（必重建）；增益接近 0 时改用绝对差避免除零
                boolean gainChanged = (lastRebuildGain < 0f)
                        || (Math.abs(lastRebuildGain) < 1e-4f
                            ? Math.abs(CUR_BRIGHT_GAIN - lastRebuildGain) > REBUILD_GAIN_RATIO * 0.5f
                            : Math.abs(CUR_BRIGHT_GAIN / lastRebuildGain - 1f) > REBUILD_GAIN_RATIO);
                boolean offsetChanged = Math.abs(CUR_BRIGHT_OFFSET - lastRebuildOffset) > REBUILD_OFFSET_ABS;
                boolean shadowChanged = Math.abs(shadowLift - lastRebuildShadow) > SHADOW_REBUILD_ABS;
                boolean needRebuild = gainChanged || (lastRebuildOffset < 0f) || offsetChanged || shadowChanged;
                if (needRebuild) {
                    rebuildBrightLut();
                    lastRebuildGain = CUR_BRIGHT_GAIN;
                    lastRebuildOffset = CUR_BRIGHT_OFFSET;
                    lastRebuildShadow = shadowLift;
                }
            }
            dynBrightComputed = true;
        }

        int total = len * 3;
        if (inputFloats == null || inputFloats.length != total) {
            inputFloats = new float[total];
        }

        // float32，值 [0,1] 并做亮度增强：LUT 查表替代每像素浮点乘加 clamp
        // copyPixelsToBuffer 对 ARGB_8888 输出 RGBA 字节序：byte[4i]=R, [4i+1]=G, [4i+2]=B, [4i+3]=A
        if (inputNchw) {
            // NCHW [1,3,H,W]：R 全图、G 全图、B 全图
            for (int i = 0; i < len; i++) {
                int idx = i * 4;
                inputFloats[i] = brightLut[pixelBytes[idx] & 0xFF];
                inputFloats[len + i] = brightLut[pixelBytes[idx + 1] & 0xFF];
                inputFloats[len * 2 + i] = brightLut[pixelBytes[idx + 2] & 0xFF];
            }
        } else {
            // NHWC [1,H,W,3]：逐像素 R,G,B 连续
            for (int i = 0; i < len; i++) {
                int idx = i * 4;
                int base = i * 3;
                inputFloats[base] = brightLut[pixelBytes[idx] & 0xFF];
                inputFloats[base + 1] = brightLut[pixelBytes[idx + 1] & 0xFF];
                inputFloats[base + 2] = brightLut[pixelBytes[idx + 2] & 0xFF];
            }
        }

        // 一次性 bulk 写入 ByteBuffer，避免每像素多次 JNI 调用
        inputFloatBuffer.rewind();
        inputFloatBuffer.put(inputFloats);
        inputBuffer.rewind();
    }

    /** 第二路并发推理：中心放大区裁剪→缩放→独立 buffer→interpreter2 推理→parseOutput2 */
    private PersonPose runSecondInference(Bitmap source) {
        try {
            Bitmap rb2 = preprocessBitmap2(source);
            bitmapToByteBuffer2(rb2);
            long ti0 = System.nanoTime();
            if (multiOutputMode) {
                java.util.Map<Integer, Object> outputs2 = new java.util.HashMap<>();
                outputs2.put(0, outputBoxes2);
                outputs2.put(1, outputScores2);
                interpreter2.runForMultipleInputsOutputs(new Object[]{inputBuffer2}, outputs2);
            } else {
                interpreter2.run(inputBuffer2, output2);
            }
            long ti1 = System.nanoTime();
            secondInferNanos.set(ti1 - ti0);
            return parseOutput2();
        } catch (Throwable t) {
            Log.w(TAG, "[2nd] 推理异常", t);
            return null;
        }
    }

    /** 第二路预处理：从原图裁中心放大区（跟随主 ROI 中心）缩放到 640 */
    private Bitmap preprocessBitmap2(Bitmap source) {
        int sw = source.getWidth();
        int sh = source.getHeight();
        int base = Math.min(sw, sh);
        zoomSize = Math.max(16, (int) (base * ROI_SCALE * DUAL_ZOOM));
        int maxZoom = Math.min(sw, sh);
        if (zoomSize > maxZoom) zoomSize = maxZoom;
        float zoomCx = roiX + roiSize / 2f;
        float zoomCy = roiY + roiSize / 2f;
        zoomX = (int) Math.max(0, Math.min(sw - zoomSize, zoomCx - zoomSize / 2f));
        zoomY = (int) Math.max(0, Math.min(sh - zoomSize, zoomCy - zoomSize / 2f));
        scaleSrcRect2.set(zoomX, zoomY, zoomX + zoomSize, zoomY + zoomSize);
        scaleCanvas2.drawBitmap(source, scaleSrcRect2, scaleRect2, scalePaint2);
        return scaledRoi2;
    }

    /** 第二路输入转换：独立 pixels2/inputFloats2/inputBuffer2，避免并发字段冲突 */
    private void bitmapToByteBuffer2(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int len = width * height;
        if (pixels2 == null || pixels2.length != len) {
            pixels2 = new int[len];
        }
        bitmap.getPixels(pixels2, 0, width, 0, 0, width, height);
        int total = len * 3;
        if (inputFloats2 == null || inputFloats2.length != total) {
            inputFloats2 = new float[total];
        }
        if (inputNchw) {
            for (int i = 0; i < len; i++) {
                int p = pixels2[i];
                inputFloats2[i] = brighten(((p >> 16) & 0xFF) * INV_255);
                inputFloats2[len + i] = brighten(((p >> 8) & 0xFF) * INV_255);
                inputFloats2[len * 2 + i] = brighten((p & 0xFF) * INV_255);
            }
        } else {
            for (int i = 0; i < len; i++) {
                int p = pixels2[i];
                int base = i * 3;
                inputFloats2[base] = brighten(((p >> 16) & 0xFF) * INV_255);
                inputFloats2[base + 1] = brighten(((p >> 8) & 0xFF) * INV_255);
                inputFloats2[base + 2] = brighten((p & 0xFF) * INV_255);
            }
        }
        inputFloatBuffer2.rewind();
        inputFloatBuffer2.put(inputFloats2);
        inputBuffer2.rewind();
    }

    private float getOutputValue2(int channel, int anchor) {
        if (multiOutputMode) {
            if (channel < 4) {
                if (channelFirst) return outputBoxes2[0][channel][anchor];
                else return outputBoxes2[0][anchor][channel];
            } else {
                int sc = channel - 4;
                if (channelFirst) return outputScores2[0][sc][anchor];
                else return outputScores2[0][anchor][sc];
            }
        }
        if (channelFirst) {
            return output2[0][channel][anchor];
        } else {
            return output2[0][anchor][channel];
        }
    }

    /**
     * 第二路精简解析：遍历 anchors，选 conf>=VALID 的最高分（距离中心+置信度），
     * box 反变换到原图像素坐标。不做跟踪/OCM/C-BIoU（那是主路职责）。
     * 返回 PersonPose.box=[px,py,pw,ph,conf]，同时 allValidBoxes 保存所有 VALID 检测（交叉核验用）。
     */
    private PersonPose parseOutput2() {
        int numPredictions = numAnchors;
        PersonPose best = null;
        float maxScore = -1f;
        java.util.ArrayList<float[]> all = new java.util.ArrayList<>(8);
        final int VALID = (int) (VALID_DETECTION_CONF * 1000000);
        for (int i = 0; i < numPredictions; i++) {
            // OPT和非OPT的conf通道输出模式一致（均含sigmoid），统一取ch4+的max
            float boxConf = 0;
            for (int c = 4; c < numChannels; c++) {
                float conf = getOutputValue2(c, i);
                if (conf > boxConf) boxConf = conf;
            }
            if ((int) (boxConf * 1000000) < VALID) continue;
            float cx, cy, w, h;
            if (boxIsXyxy) {
                // OPT 模型：box 为 cxcywh 像素值 → ÷inputSize 归一化
                float invSz = 1f / inputSize;
                cx = getOutputValue2(0, i) * invSz;
                cy = getOutputValue2(1, i) * invSz;
                w = getOutputValue2(2, i) * invSz;
                h = getOutputValue2(3, i) * invSz;
            } else {
                cx = getOutputValue2(0, i);
                cy = getOutputValue2(1, i);
                w = getOutputValue2(2, i);
                h = getOutputValue2(3, i);
            }
            float area = w * h;
            if (area < MIN_AREA_THRESHOLD) continue;
            // 反变换到原图像素坐标（第二路 zoom 区域，与主路 cx*roiSize+roiX 同理）
            float px = cx * zoomSize + zoomX;
            float py = cy * zoomSize + zoomY;
            float pw = w * zoomSize;
            float ph = h * zoomSize;
            all.add(new float[]{px, py, pw, ph, boxConf});
            float dx = (px - originalWidth / 2f) / originalWidth;
            float dy = (py - originalHeight / 2f) / originalHeight;
            float distToCenter = (float) Math.sqrt(dx * dx + dy * dy);
            float distWeight = (float) Math.exp(-(distToCenter * distToCenter)
                    / (2 * CENTER_SIGMA * CENTER_SIGMA));
            float score = distWeight * boxConf;
            if (score > maxScore) {
                maxScore = score;
                best = new PersonPose();
                best.box = new float[]{px, py, pw, ph, boxConf};
            }
        }
        if (best != null) best.allValidBoxes = all;
        return best;
    }

    /** 重建亮度查表（GAIN/OFFSET/暗部抬升 变化时调用）：线性增益 + 可选暗部抬升分段曲线 */
    private void rebuildBrightLut() {
        if (brightLut == null || brightLut.length != 256) {
            brightLut = new float[256];
        }
        float curGain = CUR_BRIGHT_GAIN;
        float curOffNorm = CUR_BRIGHT_OFFSET_NORM;
        float lift = shadowLift;
        float invCut = lift > 0f ? 1f / SHADOW_LIFT_CUTOFF : 0f;
        for (int i = 0; i < 256; i++) {
            float w = i * INV_255;                       // 输入亮度 0~1
            float v = w * curGain + curOffNorm;          // 线性增益 + 偏移
            if (lift > 0f && w < SHADOW_LIFT_CUTOFF) {
                // 暗部抬升：w=0 抬 lift，随 w 线性衰减到 cutoff 处为 0（亮部完全不动）
                v += lift * (1f - w * invCut);
            }
            if (v < 0f) v = 0f;
            else if (v > 1f) v = 1f;
            brightLut[i] = v;
        }
    }

    private float brighten(float v) {
        float out = v * CUR_BRIGHT_GAIN + CUR_BRIGHT_OFFSET_NORM;
        // 与主路 brightLut 保持一致：暗部抬升（亮环境+黑物体时抬阴影），
        // 否则第二路输入与主路不一致，影响双路交叉核验
        float lift = shadowLift;
        if (lift > 0f && v < SHADOW_LIFT_CUTOFF) {
            out += lift * (1f - v / SHADOW_LIFT_CUTOFF);
        }
        // clamp 到 [0,1]：用户在高级选项可能输入负 BRIGHTNESS_OFFSET，
        // 不 clamp 下界会让模型输入出现负值，产生异常输出
        if (out < 0f) return 0f;
        if (out > 1f) return 1f;
        return out;
    }

    private float getOutputValue(int channel, int anchor) {
        if (multiOutputMode) {
            if (channel < 4) {
                if (channelFirst) return outputBoxes[0][channel][anchor];
                else return outputBoxes[0][anchor][channel];
            } else {
                int sc = channel - 4;
                if (channelFirst) return outputScores[0][sc][anchor];
                else return outputScores[0][anchor][sc];
            }
        }
        if (channelFirst) {
            return output[0][channel][anchor];
        } else {
            return output[0][anchor][channel];
        }
    }

    /** 日志格式化：保留4位小数，避免科学计数法淹没关键信息 */
    private static String fmt4(float v) {
        if (Float.isNaN(v) || Float.isInfinite(v)) return Float.toString(v);
        return String.format(java.util.Locale.US, "%.4f", v);
    }

    private PersonPose parseOutput() {
        int numPredictions = numAnchors;

        // ========== 真实首帧诊断：统计conf通道raw范围（所有模型，对比OPT vs 非OPT） ==========
        // probe是第1次调用(黑图)，第2次调用才是真实画面 → 用estimateCallCount跳过probe
        if (estimateCallCount >= 2 && !realStatsLogged) {
            realStatsLogged = true;
            float objMin = Float.MAX_VALUE, objMax = -Float.MAX_VALUE;
            float clsMin = Float.MAX_VALUE, clsMax = -Float.MAX_VALUE;
            float rawProdMax = 0f, sigProdMax = 0f;
            int rawPassCnt = 0, sigPassCnt = 0;
            java.util.List<String> top5 = new java.util.ArrayList<>();
            int showN = Math.min(5, numPredictions);
            for (int i = 0; i < numPredictions; i++) {
                float obj = getOutputValue(4, i);
                float cls = 0f;
                for (int c = 5; c < numChannels; c++) {
                    float cf = getOutputValue(c, i);
                    if (cf > cls) cls = cf;
                }
                if (obj < objMin) objMin = obj;  if (obj > objMax) objMax = obj;
                if (cls < clsMin) clsMin = cls;  if (cls > clsMax) clsMax = cls;
                float rp = obj * cls;
                float sigO = 1f / (1f + (float) Math.exp(-obj));
                float sigC = 1f / (1f + (float) Math.exp(-cls));
                float sp = sigO * sigC;
                if (rp > rawProdMax) rawProdMax = rp;
                if (sp > sigProdMax) sigProdMax = sp;
                if (rp >= BYTETRACK_LOW_CONF) rawPassCnt++;
                if (sp >= BYTETRACK_LOW_CONF) sigPassCnt++;
                if (i < showN) {
                    top5.add(String.format(java.util.Locale.US,
                            "#%d obj=%s cls=%s rp=%s sp=%s (sigO=%s sigC=%s)",
                            i, fmt4(obj), fmt4(cls), fmt4(rp), fmt4(sp), fmt4(sigO), fmt4(sigC)));
                }
            }
            StringBuilder sb = new StringBuilder(800);
            sb.append("[REAL_STATS_OPT] first real frame stats, anchors=").append(numPredictions)
                    .append(" BYTETRACK_LOW_CONF=").append(BYTETRACK_LOW_CONF).append('\n');
            sb.append("  obj(raw) : min=").append(fmt4(objMin)).append(" max=").append(fmt4(objMax))
                    .append("  (超出0~1 => 缺少sigmoid)\n");
            sb.append("  cls(raw) : min=").append(fmt4(clsMin)).append(" max=").append(fmt4(clsMax))
                    .append("  (超出0~1 => 缺少sigmoid)\n");
            sb.append("  raw(obj*cls) max=").append(fmt4(rawProdMax))
                    .append("  pass>=").append(BYTETRACK_LOW_CONF).append(" count=").append(rawPassCnt).append('\n');
            sb.append("  sig(obj)*sig(cls) max=").append(fmt4(sigProdMax))
                    .append("  pass>=").append(BYTETRACK_LOW_CONF).append(" count=").append(sigPassCnt).append('\n');
            sb.append("  top5 anchors: ").append(TextUtils.join(" | ", top5));
            Log.w(TAG, sb.toString());
        }

        // 收集 low conf 检测的 box（[px,py,pw,ph,conf]）用于 ByteTrack 第二阶段
        float[][] lowBoxes = null;
        int lowCount = 0;
        // 提前分配一次性缓冲：最多 N 个 low（上限 anchors，实际少得多）
        // 这里用一个动态的小列表，用数组避免 ArrayList 分配
        int lowCap = 32;
        lowBoxes = new float[lowCap][];

        // OA-SORT OAM：收集 HIGH 检测框用于 bestPose 遮挡系数计算
        // highConfs 并行存置信度：NMS 去重需要按置信度排序压低分框
        int highCap = 32;
        float[][] highBoxes = new float[highCap][];
        float[] highConfs = new float[highCap];
        int highCount = 0;

        // ===== OC-SORT OCM：先对 trackedBox 做基于速度的位置外推 =====
        // 用「预测后的 trackedBox」去跟检测框做匹配，解决突然移动（开镜）时 IoU 骤降断锁
        float[] predTracked = trackedBox;
        if (trackedBox != null && trackLostFrames < MAX_TRACK_LOST) {
            // lost 帧数作为预测步长；+1 代表当前帧再推一次
            int steps = Math.min(trackLostFrames + 1, 5); // 上限 5 帧，避免外推过远
            // 运动动力学 KF (arxiv 2505.07254)：加速度大切 CA（匀加速）模型外推更准，
            // 加速度小用 CV（匀速）。acc = lastRawVx - prevRawVx（连续两帧瞬时速度差）
            float accX = lastRawVx - prevRawVx;
            float accY = lastRawVy - prevRawVy;
            float accMag = (float) Math.sqrt(accX * accX + accY * accY);
            float pxC, pyC;
            if (accMag >= ACC_THRESHOLD) {
                // CA 模型：x = x0 + v*t + 0.5*a*t²，横向加速时预测更贴合
                // 二次项限幅：不超过线性位移。低速大目标（占满 ROI）框中心抖动
                // 会制造噪声级"加速度"，steps 放大到 5 步后二次项爆炸（数百px），
                // predTracked 漂出救援缓冲区 → 匹配断链 → 30帧计数走满丢目标
                float linX = lastRawVx * steps;
                float linY = lastRawVy * steps;
                float quaX = 0.5f * accX * steps * steps;
                float quaY = 0.5f * accY * steps * steps;
                if (Math.abs(quaX) > Math.abs(linX)) quaX = Math.abs(linX) * Math.signum(quaX);
                if (Math.abs(quaY) > Math.abs(linY)) quaY = Math.abs(linY) * Math.signum(quaY);
                pxC = trackedBox[0] + linX + quaX;
                pyC = trackedBox[1] + linY + quaY;
            } else {
                // CV 模型：匀速外推
                pxC = trackedBox[0] + lastRawVx * steps;
                pyC = trackedBox[1] + lastRawVy * steps;
            }
            predTracked = new float[]{pxC, pyC, trackedBox[2], trackedBox[3]};
        }

        // ===== C-BIoU 速度自适应 buffer：HIGH 阶段 buffer 随速度放大 =====
        // 速度归一化 = 速度 / 框宽，达到 SPEED_REF 时 buffer 从 BUF_HIGH 线性放大到 BUF_LOW
        // 静止时小 buffer 防误配，快移时大 buffer 防失配
        float dynHighBuffer = CBIoU_BUF_HIGH;
        if (CMC_ENABLED && cameraMotionActive) {
            // CMC：滑动跟随画面整体平移，帧间 IoU 骤降，buffer 直接放大到 LOW 上限防失配
            dynHighBuffer = CBIoU_BUF_LOW;
        } else if (trackedBox != null && trackLostFrames < MAX_TRACK_LOST && trackedBox[2] > 1e-3f) {
            float speed = (float) Math.sqrt(lastRawVx * lastRawVx + lastRawVy * lastRawVy);
            float speedNorm = speed / trackedBox[2];
            float t = speedNorm / CBIoU_SPEED_REF;
            if (t > 1f) t = 1f;
            dynHighBuffer = CBIoU_BUF_HIGH + (CBIoU_BUF_LOW - CBIoU_BUF_HIGH) * t;
        }

        // ===== C-BIoU 第一阶段：高置信度（conf >= HIGH=VALID_DETECTION_CONF）正常评分 =====
        PersonPose bestPose = null;
        float bestConf = 0f;
        boolean bestMatchedTrack = false;
        float topConf = 0f;   // 本帧模型输出最高 conf（诊断用：判断漏检 vs 逻辑吞掉）
        // 两路选主：匹配锁定目标的框按「匹配度 simTrack」竞争（连续性优先，绿框不横跳）；
        // 不匹配的高分新框按 score（conf×位置权重）竞争。锁定存在时匹配框永远优先，
        // 只有当锁定目标完全消失时才由高分新框接管（走切换滞后）。
        PersonPose bestMatchedPose = null;
        float bestMatchSim = -1f;
        float bestMatchedConf = 0f;
        PersonPose bestUnmatchedPose = null;
        float bestUnmatchedScore = -1f;
        float bestUnmatchedConf = 0f;

        for (int i = 0; i < numPredictions; i++) {
            // OPT和非OPT的conf通道输出模式一致（均含sigmoid），统一取ch4+的max
            float boxConf = 0;
            for (int c = 4; c < numChannels; c++) {
                float conf = getOutputValue(c, i);
                if (conf > boxConf) {
                    boxConf = conf;
                }
            }
            if (boxConf > topConf) topConf = boxConf;  // 记录模型本帧最强输出
            // ByteTrack: 第一阶段阈值用 HIGH；低于 HIGH 但 >= LOW 暂存为候选项
            if (boxConf < BYTETRACK_LOW_CONF) {
                continue;
            }

            float cx, cy, w, h;
            if (boxIsXyxy) {
                // OPT 模型：box 同为 cxcywh 格式，但单位是「像素值」(0~inputSize)，
                // 而非旧模型的归一化 0~1。直接÷inputSize 归一化即可，无需 xyxy 转换。
                // （实测：OPT 416 的 cxcywh × 1/416 == 非OPT 416 的归一化 cxcywh，精确吻合）
                float invSz = 1f / inputSize;
                cx = getOutputValue(0, i) * invSz;
                cy = getOutputValue(1, i) * invSz;
                w = getOutputValue(2, i) * invSz;
                h = getOutputValue(3, i) * invSz;
            } else {
                // 旧模型：cxcywh 格式（已归一化 0~1）
                cx = getOutputValue(0, i);
                cy = getOutputValue(1, i);
                w = getOutputValue(2, i);
                h = getOutputValue(3, i);
            }
            float area = w * h;
            if (area < MIN_AREA_THRESHOLD) {
                continue;
            }

            float px = cx * roiSize + roiX;
            float py = cy * roiSize + roiY;
            float pw = w * roiSize;
            float ph = h * roiSize;

            if (boxConf >= VALID_DETECTION_CONF) {
                // === HIGH 分支：参与综合评分选主目标 ===
                float[] box = new float[]{px, py, pw, ph};

                // OA-SORT OAM：收集 HIGH 检测框，bestPose 选出后用于计算被遮挡系数
                // highConfs 并行存置信度（NMS 去重要用）
                if (highCount >= highCap) {
                    int newHighCap = highCap * 2;
                    float[][] newHigh = new float[newHighCap][];
                    System.arraycopy(highBoxes, 0, newHigh, 0, highCap);
                    highBoxes = newHigh;
                    float[] newConfs = new float[newHighCap];
                    System.arraycopy(highConfs, 0, newConfs, 0, highCap);
                    highConfs = newConfs;
                    highCap = newHighCap;
                }
                highBoxes[highCount] = box;
                highConfs[highCount] = boxConf;
                highCount++;

                float dx = (px - originalWidth / 2f) / originalWidth;
                float dy = (py - originalHeight / 2f) / originalHeight;

                // C-BIoU：跟踪匹配用预测后的 trackedBox 做 buffered IoU（HIGH 阶段，速度自适应 buffer）
                boolean matched = predTracked != null
                        && simTrack(predTracked, box, dynHighBuffer) >= TRACK_IOU_THRESHOLD;
                if (matched) {
                    // 匹配锁定目标：按匹配度 simTrack 竞争（连续性优先）。
                    // 两个目标都匹配锁定框时不再按 conf/位置 score 竞争，避免绿框在目标间帧级横跳；
                    // trackedBox 会更新到选中框，下一帧该框匹配度最高，形成自稳定。
                    float matchSim = simTrack(predTracked, box, dynHighBuffer);
                    // OC-SORT OCM 方向一致性惩罚：候选相对预测位置的位移方向与历史速度方向
                    // 夹角越背离惩罚越重。SMOT(arXiv 2507.12087) 对齐：惩罚由 EMA 方向可信度
                    // 驱动、低速不归零（0.5 下限调制）——低速大目标正是「帧间位移噪声≈真实位移」、
                    // 最需要方向约束的区间（OC-SORT 限制1）。HIGH/LOW 救援/回收三阶段共用。
                    matchSim -= ocmDirectionPenalty(predTracked[0], predTracked[1], box[0], box[1]);
                    if (matchSim > bestMatchSim) {
                        bestMatchSim = matchSim;
                        bestMatchedConf = boxConf;
                        bestMatchedPose = new PersonPose();
                        bestMatchedPose.box = box;
                    }
                } else {
                    // 不匹配的高分新框：按 conf×位置权重 score 竞争（仅锁定消失时接管）
                    float distToCenter = (float) Math.sqrt(dx * dx + dy * dy);
                    float distWeight = (float) Math.exp(-(distToCenter * distToCenter)
                            / (2 * CENTER_SIGMA * CENTER_SIGMA));
                    // 方案3：水平偏心角度分数（Best Target Selection, Nicholas Gorski）
                    // FPS 横向瞄准是关键，垂直偏心不惩罚（敌人可能在上下半部分）
                    float angleWeight = (float) Math.exp(-(dx * dx)
                            / (2 * ANGLE_SIGMA * ANGLE_SIGMA));
                    // 组合：score = (1-W)*dist + W*angle，距离为主角度为辅
                    float posWeight = (1f - SCORE_W) * distWeight + SCORE_W * angleWeight;
                    float score = boxConf * posWeight;
                    if (score > bestUnmatchedScore) {
                        bestUnmatchedScore = score;
                        bestUnmatchedConf = boxConf;
                        bestUnmatchedPose = new PersonPose();
                        bestUnmatchedPose.box = box;
                    }
                }
            } else {
                // === LOW 分支（BYTETRACK_LOW_CONF <= conf < VALID_DETECTION_CONF）
                // 暂存，第一阶段结束后专门用于「救活」被遮挡的已跟踪目标
                if (lowCount >= lowCap) {
                    // 扩容（极少触发）
                    int newCap = lowCap * 2;
                    float[][] newLow = new float[newCap][];
                    System.arraycopy(lowBoxes, 0, newLow, 0, lowCap);
                    lowBoxes = newLow;
                    lowCap = newCap;
                }
                lowBoxes[lowCount++] = new float[]{px, py, pw, ph, boxConf};
            }
        }

        // ===== 选主：锁定目标的匹配框永远优先（连续性），无匹配框才用高分新框 =====
        if (bestMatchedPose != null) {
            bestPose = bestMatchedPose;
            bestConf = bestMatchedConf;
            bestMatchedTrack = true;
        } else if (bestUnmatchedPose != null) {
            bestPose = bestUnmatchedPose;
            bestConf = bestUnmatchedConf;
            bestMatchedTrack = false;
        }

        // ===== HIGH 检测框 NMS 去重（方案 A）=====
        // YOLO 对同一目标常产生 2~3 个重叠高分框，不处理会被 OAM 当成"多人遮挡" →
        // BAM 过度降低观测信任 → 跟踪靠预测漂移。O(n²) 两两比对：IoU≥0.8 的重叠对，
        // 压掉低分那个；最后 compact 回 highBoxes 开头，highCount 更新为去重后数量。
        // 仅影响 OAM occlusion 计算；选主逻辑（bestPose 遍历中已选好）不变。
        if (highCount > 1) {
            boolean[] suppressed = new boolean[highCount];
            for (int i = 0; i < highCount; i++) {
                if (suppressed[i]) continue;
                for (int j = i + 1; j < highCount; j++) {
                    if (suppressed[j]) continue;
                    float ov = boxIou(highBoxes[i], highBoxes[j]);
                    if (ov >= HIGH_NMS_IOU_THRESH) {
                        // IoU 够大，压掉置信度低的那个
                        if (highConfs[i] >= highConfs[j]) {
                            suppressed[j] = true;
                        } else {
                            suppressed[i] = true;
                            break; // i 被压了，不用再比 i 后面
                        }
                    }
                }
            }
            // 把未被压制的框从前往后紧凑挪
            int nmsCount = 0;
            for (int i = 0; i < highCount; i++) {
                if (!suppressed[i]) {
                    if (nmsCount != i) { // 非原地才搬（原地正确位置）
                        highBoxes[nmsCount] = highBoxes[i];
                        highConfs[nmsCount] = highConfs[i];
                    }
                    nmsCount++;
                }
            }
            // 末尾的引用置空，避免 GC 认为旧框还活着（虽然是局部栈，但安全清理）
            for (int i = nmsCount; i < highCount; i++) {
                highBoxes[i] = null;
            }
            highCount = nmsCount;
        }

        // ===== C-BIoU 第二阶段：已跟踪目标在 HIGH 阶段没匹配到，用 LOW 集合救援 =====
        // LOW 用更大 buffer(0.5) 扩展 + 阈值(0.3)：放宽匹配容许遮挡漂移
        boolean rescuedByLow = false;
        if (trackedBox != null && !bestMatchedTrack && lowCount > 0) {
            // C-BIoU LOW 阈值为正数(0.3)，初始 -1 兜底
            float bestLowIoU = -1f;
            float[] bestLowBox = null;
            for (int i = 0; i < lowCount; i++) {
                float[] lb = lowBoxes[i];
                float[] lbBox = new float[]{lb[0], lb[1], lb[2], lb[3]};
                // LOW 救援与 HIGH 一致用 simTrack：bufferedIoU + 距离补分 + 高度相似性
                // simTrack >= bufferedIoU，阈值保持 CBIoU_IOU_LOW 等效放宽，与 HIGH 策略对齐
                // OCM 方向惩罚同样生效：HIGH 失配时方向是区分真假目标的强信号
                float sim = simTrack(predTracked, lbBox, CBIoU_BUF_LOW)
                        - ocmDirectionPenalty(predTracked[0], predTracked[1], lbBox[0], lbBox[1]);
                if (sim >= CBIoU_IOU_LOW && sim > bestLowIoU) {
                    bestLowIoU = sim;
                    bestLowBox = lbBox;
                }
            }
            if (bestLowBox != null) {
                // 救援成功：把 LOW box 当作本次选中目标，替代 high 阶段的选择
                // （只在 HIGH 没匹配到跟踪时才替换；若 HIGH 已选到一个 unrelated 高分框也替换，
                //  避免「有人突然出现在画面边缘就被抢走锁定」）
                bestPose = new PersonPose();
                bestPose.box = bestLowBox;
                bestConf = VALID_DETECTION_CONF; // 视为有效目标，避免下游判定无人
                rescuedByLow = true;
                bestMatchedTrack = true;
            }
        }

        // ===== 方案 B：OATrack 三阶段回收（最近丢失恢复） =====
        // LOW 救援失败后，若 trackLostFrames ∈ [1, RECOVERY_MAX_FRAMES]（短期刚丢），
        // 用更宽缓冲(RECOVERY_BUF=0.7)+更低阈值(RECOVERY_IOU=0.2)再扫一次 LOW 集合。
        // 设计意图：目标被短暂遮挡（1~3帧），Coast 外推位置与真实检测位置错位较大，
        // LOW 阶段因阈值/buffer不够宽而漏掉；此处进一步放宽条件，优先恢复"最近还在跟踪"的目标，
        // 而不是让 MAX_TRACK_LOST(30帧) 倒计时走完或被其他高分框抢走锁定。
        if (trackedBox != null && !bestMatchedTrack && !rescuedByLow
                && trackLostFrames >= 1 && trackLostFrames <= RECOVERY_MAX_FRAMES
                && lowCount > 0) {
            float bestRecoveryIoU = -1f;
            float[] bestRecoveryBox = null;
            for (int i = 0; i < lowCount; i++) {
                float[] lb = lowBoxes[i];
                float[] lbBox = new float[]{lb[0], lb[1], lb[2], lb[3]};
                // Recovery 阶段同样用 simTrack：宽缓冲 + 距离补分 + 高度相似性，
                // 与 HIGH/LOW 策略一致，进一步提高短遮挡场景的恢复率。
                // OCM 方向惩罚：丢失 ≤RECOVERY_MAX_FRAMES 内速度方向仍可信，防误恢复他目标
                float sim = simTrack(predTracked, lbBox, RECOVERY_BUF)
                        - ocmDirectionPenalty(predTracked[0], predTracked[1], lbBox[0], lbBox[1]);
                if (sim >= RECOVERY_IOU && sim > bestRecoveryIoU) {
                    bestRecoveryIoU = sim;
                    bestRecoveryBox = lbBox;
                }
            }
            if (bestRecoveryBox != null) {
                bestPose = new PersonPose();
                bestPose.box = bestRecoveryBox;
                bestConf = VALID_DETECTION_CONF;
                rescuedByLow = true;
                bestMatchedTrack = true;
            }
        }

        // ===== 锁定保持（track-and-hold）：已锁定且本帧无匹配框时，绝不切换到其他目标。=====
        // 分两段处理，显示层与锁定层解耦：
        //   [0, LOCK_HOLD_FRAMES) 短暂漏检 → 补位显示旧目标（绿框保持，防闪烁）
        //   [LOCK_HOLD_FRAMES, MAX_TRACK_LOST) 持续丢失 → 绿框淡出（bestPose=null），
        //       但仍锁 A，强制丢弃 unmatched 新框 B，禁止 B 抢锁 → 杜绝 A/B 横跳
        // 只有当 trackLostFrames 达到 MAX_TRACK_LOST（A 彻底消失）才释放锁定，B 才能接管。
        boolean heldByLock = false;   // 本帧是否由锁定保持补位（非真实检测）
        boolean coastPose = false;    // 锁定保持补位是否用了速度外推
        if (trackedBox != null && trackLostFrames < MAX_TRACK_LOST && !bestMatchedTrack) {
            // 接管阈值：新目标置信度 ≥ TAKEOVER_CONF 时绕过锁定保持，立即切换锁定
            // （解决「人物突然入场」被压制 MAX_TRACK_LOST 帧的延迟；远处小目标 conf 通常 < 0.5 仍受保护）
            // 目标锁定开启时禁用即时接管：只允许当前目标彻底消失（MAX_TRACK_LOST）后换锁
            boolean takeover = !TARGET_LOCK_ENABLED && bestPose != null && bestConf >= TAKEOVER_CONF;
            if (takeover) {
                // 高分新目标即时接管：释放旧锁（trackedBox=null），让后续通用路径按「首次锁定」处理 B；
                // 否则 BAM 混合块会用旧目标 A 的 predTracked 外推位置污染 B（A/B 相距远时 bam≈0 → trackedBox 退回 A）。
                trackedBox = null;
                trackLostFrames = 0;
                trackVelX = 0f;
                trackVelY = 0f;
                lastRawVx = 0f;
                lastRawVy = 0f;
                prevRawVx = 0f;
                prevRawVy = 0f;
                lastTrackTimeNanos = 0L;
                adaptiveLeadSec = PREDICT_SECONDS;
                leadErrEma = 0f;
                lastLeadDx = 0f;
                lastLeadDy = 0f;
                leadValid = false;
                coastEngaged = false;   // 防 A 的外推记忆延续到 B
            } else if (trackLostFrames < LOCK_HOLD_FRAMES) {
                // 短暂漏检：补位显示旧目标（绿框保持，防闪烁）
                float[] holdBox = trackedBox;
                float speedSq = lastRawVx * lastRawVx + lastRawVy * lastRawVy;
                // 滞回：进入需 ≥1.2×COAST_MIN_VEL，激活后 ≥0.8× 即可保持
                float enterT = COAST_MIN_VEL * 1.2f;
                float holdT = COAST_MIN_VEL * 0.8f;
                coastEngaged = coastEngaged ? (speedSq >= holdT * holdT)
                        : (speedSq >= enterT * enterT);
                if (coastEngaged && trackLostFrames <= COAST_MAX_FRAMES) {
                    // 外推量逐帧衰减（SocialTrack 2025 加速衰减思想）：
                    // 增量 1 → 0.5 → 0.25 帧位移，总位置渐近 2×速度，补位轨迹自然减速停车；
                    // 替代旧恒定步长 min(lost+1,2)：旧版第2帧起框「悬停」在 2×速度处再骤停
                    float steps = 2f - 1f / (float) (1 << trackLostFrames);
                    holdBox = new float[] {
                            trackedBox[0] + lastRawVx * steps,
                            trackedBox[1] + lastRawVy * steps,
                            trackedBox[2],
                            trackedBox[3]
                    };
                    coastPose = true;
                }
                bestPose = new PersonPose();
                bestPose.box = holdBox.clone();
                bestConf = 0f;
                heldByLock = true;
            } else {
                // 持续丢失：绿框淡出，但仍锁 A，丢弃 unmatched 新框 B 防止抢锁
                bestPose = null;
                bestConf = 0f;
            }
        }

        // ===== OA-SORT OAM：计算当前 bestPose 被其他 HIGH 检测框遮挡的系数 =====
        // 本帧 BAM 直接取用（OA-SORT 2026 修订版：BAM 由当前帧遮挡图驱动，消除一帧滞后；
        // 旧实现用上一帧 lastBestOcclusion，30~60fps 下遮挡刚发生的头几帧 BAM 错误地信了观测）
        float currentOcclusion = 0f;
        if (bestPose != null && highCount > 1) {
            currentOcclusion = computeOcclusion(bestPose.box, highBoxes, highCount);
        }

        // ===== 更新速度估计 & 跟踪状态 =====
        long nowNanos = System.nanoTime();
        if (bestPose != null && !heldByLock) {
            // OA-SORT BAM (Bias-Aware Momentum)：
            // BAM = IoU(predTracked, Z) · (1 - Oc_cur)
            // Z' = BAM · Z + (1 - BAM) · predTracked
            // 遮挡严重（Oc_cur 大）或观测与预测位置偏离（IoU 低）时，BAM 减小，
            // trackedBox 更多沿用 predTracked 速度外推，防止遮挡中漂移的检测框污染轨迹
            float[] newBox = bestPose.box.clone();
            // OC-SORT ORU 简化版：恢复首帧（trackLostFrames>0）跳过 BAM 混合，全信观测。
            // 丢失期间 predTracked 是外推位置，恢复帧把它掺进正确观测 = 把错误拖回轨迹，
            // 论文 ORU 正是为此：恢复关联后先用观测重建状态，而非信任先验估计。
            if (trackedBox != null && predTracked != null && predTracked != trackedBox
                    && trackLostFrames == 0) {
                float bam = boxIou(predTracked, newBox) * (1f - currentOcclusion);
                if (bam < 0f) bam = 0f;
                else if (bam > 1f) bam = 1f;
                // 观测与预测吻合（bam 高）时不混合：防止无遮挡帧把外推噪声注入轨迹
                if (bam < BAM_BLEND_THRESH) {
                    for (int k = 0; k < 4; k++) {
                        newBox[k] = bam * newBox[k] + (1f - bam) * predTracked[k];
                    }
                }
            }
            // 速度用混合后 newBox 的位移：BAM 混合等效位置层低通，为大型目标检测抖动
            // 提供速度估计所需的平滑；EMA 收敛、无发散风险。仅 trackLostFrames==0 的
            // 连续帧更新；救援帧（HIGH 失配但 LOW 级补回）仍更新，避免动态期速度冻结。
            if (lastTrackTimeNanos != 0L && trackedBox != null && trackLostFrames == 0
                    && !(CMC_ENABLED && cameraMotionActive)) {
                double dtSec = (nowNanos - lastTrackTimeNanos) / 1_000_000_000.0;
                if (dtSec > 1e-6) {
                    // 60fps 基准帧时长 (16.6ms) 做归一化
                    double framesElapsed = dtSec / (1.0 / 60.0);
                    if (framesElapsed < 0.5) framesElapsed = 0.5;
                    float rawVx = (newBox[0] - trackedBox[0]) / (float) framesElapsed;
                    float rawVy = (newBox[1] - trackedBox[1]) / (float) framesElapsed;
                    // 记录瞬时速度供 Coast 外推（响应变向，EMA 有滞后）
                    prevRawVx = lastRawVx;
                    prevRawVy = lastRawVy;
                    lastRawVx = rawVx;
                    lastRawVy = rawVy;
                    // 方案1：Area-Adaptive Motion Damping（AKKF, arxiv 2607.12544）
                    // 小目标面积小，几像素抖动=大比例位移噪声，增大阻尼降低 alpha
                    // 大目标面积大，速度估计可靠，alpha 保持基准值
                    float areaNorm = (trackedBox[2] * trackedBox[3])
                            / (originalWidth * originalHeight);
                    float adaptiveAlpha = VEL_ALPHA_MIN
                            + (VEL_EMA_ALPHA - VEL_ALPHA_MIN)
                            * Math.min(1f, areaNorm / AREA_NORM_REF);
                    trackVelX = (1 - adaptiveAlpha) * trackVelX + adaptiveAlpha * rawVx;
                    trackVelY = (1 - adaptiveAlpha) * trackVelY + adaptiveAlpha * rawVy;
                }
            } else if (trackedBox != null) {
                // 恢复帧（trackLostFrames>0）或 CMC 冻结（滑动跟随画面平移）：不更新速度，
                // 但刷新计时基准，否则下一帧 dtSec 跨整个丢失/滑动期 → 速度被稀释
                lastTrackTimeNanos = nowNanos;
            }

            // ===== 自适应外推阻尼反馈：残差 = 本帧观测 - 上帧(跟踪位置 + 外推量) =====
            // 此处 trackedBox 仍是「上一帧观测位置」，newBox 是「本帧观测位置」，
            // lastLeadDx/Dy 是上一帧显示外推量。若外推准确，本帧观测应落在
            // 「上一帧观测 + 上一帧外推」附近；残差沿外推方向为正表示外推不足，为负表示过冲。
            if (leadValid) {
                float predX = trackedBox[0] + lastLeadDx;
                float predY = trackedBox[1] + lastLeadDy;
                float errX = newBox[0] - predX;
                float errY = newBox[1] - predY;
                float leadLen = (float) Math.sqrt(lastLeadDx * lastLeadDx + lastLeadDy * lastLeadDy);
                float leadErr = 0f;
                if (leadLen > 1e-3f) {
                    // 残差在外推方向上的投影（带符号）
                    leadErr = (errX * lastLeadDx + errY * lastLeadDy) / leadLen;
                }
                leadErrEma = (1f - LEAD_ERR_EMA_ALPHA) * leadErrEma + LEAD_ERR_EMA_ALPHA * leadErr;
                if (leadErrEma > LEAD_DEADZONE) {
                    // 外推不足 → 增大提前量（跟得更贴）
                    adaptiveLeadSec = Math.min(LEAD_MAX, adaptiveLeadSec + LEAD_ADAPT_STEP);
                    if (++leadLogCounter % 15 == 0) {
                        Log.d(TAG, "自适应外推↑ lead=" + String.format(java.util.Locale.US, "%.4f", adaptiveLeadSec)
                                + "s errEma=" + String.format(java.util.Locale.US, "%.1f", leadErrEma) + "px");
                    }
                } else if (leadErrEma < -LEAD_DEADZONE) {
                    // 外推过度 → 减小提前量（抑制冲过头）
                    adaptiveLeadSec = Math.max(LEAD_MIN, adaptiveLeadSec - LEAD_ADAPT_STEP);
                    if (++leadLogCounter % 15 == 0) {
                        Log.d(TAG, "自适应外推↓ lead=" + String.format(java.util.Locale.US, "%.4f", adaptiveLeadSec)
                                + "s errEma=" + String.format(java.util.Locale.US, "%.1f", leadErrEma) + "px");
                    }
                }
                leadValid = false; // 本次外推量已消费
            }

            trackedBox = newBox;
            trackLostFrames = 0;
            lastTrackTimeNanos = nowNanos;
        } else if (trackedBox != null) {
            trackLostFrames++;
            // lost 期间不更新速度估计，保持最后一次速度做外推
            if (trackLostFrames >= MAX_TRACK_LOST) {
                trackedBox = null;
                trackLostFrames = 0;
                trackVelX = 0f;
                trackVelY = 0f;
                // 目标彻底丢失：清空瞬时速度/加速度状态，防止下一个目标 B 继承 A 的残留速度
                // 导致加速度误判（accX=lastRawVx-prevRawVx 出现跳变）→ 误触发 CA 外推 → 小目标匹配失败
                lastRawVx = 0f;
                lastRawVy = 0f;
                prevRawVx = 0f;
                prevRawVy = 0f;
                // 自适应外推提前量/残差 EMA 也复位，避免 A 学到的外推量延续到 B
                adaptiveLeadSec = PREDICT_SECONDS;
                leadErrEma = 0f;
                lastLeadDx = 0f;
                lastLeadDy = 0f;
                leadValid = false;
                lastTrackTimeNanos = 0L;
                coastEngaged = false;
            }
        }

        // ===== 预测未来位置（方案 A + B，外推提前量自适应学习）=====
        // 用当前速度外推 adaptiveLeadSec 秒后的位置，抵消检测→显示端到端延迟
        // 静止目标（速度低于 MIN_VEL）不外推，避免静止时绿框漂移（方案 B）
        // 注意：bestPose.box 是独立 new 出来的引用，trackedBox 已 clone，互不影响
        // trackVelX 单位：60fps 基准每帧位移，× 60 转换为"每秒位移"再 × adaptiveLeadSec
        // 补位帧（heldByLock）已由 coast 外推过（或原地补位），跳过预测外推避免双重叠加冲过头
        // LEAD_PREDICT_ENABLED=false：绿框预判关闭，不做外推，紧贴检测位置
        // CMC 生效期间速度已被冻结（trackVel 是滑动前的陈旧值），此时外推方向不可信，
        // 会污染绿框位置、并连带影响滑动跟随算出的偏差，故一并跳过外推
        if (bestPose != null && trackedBox != null && !heldByLock && LEAD_PREDICT_ENABLED
                && !(CMC_ENABLED && cameraMotionActive)) {
            float speedSq = trackVelX * trackVelX + trackVelY * trackVelY;
            if (speedSq >= MIN_VEL * MIN_VEL) {
                float predDx = trackVelX * 60f * adaptiveLeadSec;
                float predDy = trackVelY * 60f * adaptiveLeadSec;
                bestPose.box[0] += predDx;
                bestPose.box[1] += predDy;
                lastLeadDx = predDx;
                lastLeadDy = predDy;
                leadValid = true;
            } else {
                lastLeadDx = 0f;
                lastLeadDy = 0f;
                leadValid = false;
            }
        } else {
            lastLeadDx = 0f;
            lastLeadDy = 0f;
            leadValid = false;
        }

        // 检测诊断：低频打印选中目标的置信度/高宽比/面积，用于对比「背景物体 vs 真实人物」特征
        if (bestPose != null && bestPose.box != null && ++detLogCounter % 30 == 0) {
            float ar = bestPose.box[3] / Math.max(1f, bestPose.box[2]); // 高/宽比（人物竖长 >1，箱子方形 ≈1）
            float areaNorm = bestPose.box[2] * bestPose.box[3] / (originalWidth * originalHeight);
            Log.d(TAG, "选中目标 conf=" + String.format(java.util.Locale.US, "%.2f", bestConf)
                    + " 高宽比=" + String.format(java.util.Locale.US, "%.2f", ar)
                    + " 面积norm=" + String.format(java.util.Locale.US, "%.3f", areaNorm)
                    + " 尺寸=" + (int) bestPose.box[2] + "x" + (int) bestPose.box[3] + "px");
        }

        // ===== 静态图闪烁诊断（临时）：每帧累计窗口统计，满 30 帧打一条汇总 =====
        // topConf=模型本帧最强输出（含 LOW，过滤掉 conf<0.1 的噪声锚点前的最高值）
        // 判断口径：
        //   topConf 高(HIGH 附近)但 选中 少 => 检测在但被选中逻辑吞掉
        //   topConf 低(常在 0.1~0.2 以下) => 模型对静态输入间歇漏检
        //   补位>0  => 锁定保持（heldByLock）在起作用：锁定中本帧无匹配框，补位显示旧目标
        diagFrame++;
        if (topConf > diagTopMax) diagTopMax = topConf;
        if (topConf < diagTopMin) diagTopMin = topConf;
        diagTopSum += topConf;
        if (highCount > 0) diagHighFrames++;
        if (lowCount > 0) diagLowFrames++;
        if (heldByLock) diagRejected++;
        if (coastPose) diagCoast++;
        if (bestPose != null && bestPose.box != null) diagSelected++;
        if (diagFrame >= 30) {
            Log.d(TAG, "闪烁诊断 30帧: 选中=" + diagSelected
                    + " HIGH=" + diagHighFrames
                    + " LOW=" + diagLowFrames
                    + " topConf[" + fmt4(diagTopMin) + "~" + fmt4(diagTopMax)
                    + " avg=" + fmt4((float) (diagTopSum / diagFrame)) + "]"
                    + " 补位=" + diagRejected
                    + " coast=" + diagCoast
                    + " roi=(" + roiX + "," + roiY + "," + roiSize + ")"
                    + " box=" + (bestPose != null ? (int) bestPose.box[2] + "x" + (int) bestPose.box[3] : "null"));
            diagFrame = 0; diagSelected = 0; diagHighFrames = 0; diagLowFrames = 0;
            diagRejected = 0; diagCoast = 0; diagTopMax = 0f; diagTopMin = 1f; diagTopSum = 0;
        }

        // 把选中的置信度写入 PersonPose.conf，供上层降权逻辑（dual_verify/anti_crosshair）读取真实值，
        // 否则 box 只含 cxcywh 4 元素，读不到 conf（此前误读 box[4] 恒失败）
        if (bestPose != null) bestPose.conf = bestConf;
        return bestPose;
    }

    /**
     * C-BIoU（Buffered IoU）：中心不变，宽高 ×(1+2*bufRatio) 扩展后再算标准 IoU。
     * 论文 roboflow 2026 benchmark：HOTA 63.0 > OC-SORT 61.9。
     * 相比 EIoU 的优势：无中心距离惩罚，纯靠 buffer 扩展提升快速横向移动时的重叠率，
     * 避免 FPS 开镜横移时 predTracked 与 det 框中心距离大导致 EIoU 分数被 ρ²/c² 拖低而失配。
     * bufRatio 由调用方传入（HIGH 阶段 0.3，LOW 救援 0.5），实现两阶段差异化匹配。
     * 内联实现，零内存分配。
     */
    private float bufferedIoU(float[] boxA, float[] boxB, float bufRatio) {
        // 两框按 bufRatio 向外扩展（中心不变）
        float aW = boxA[2] * (1f + 2f * bufRatio);
        float aH = boxA[3] * (1f + 2f * bufRatio);
        float bW = boxB[2] * (1f + 2f * bufRatio);
        float bH = boxB[3] * (1f + 2f * bufRatio);

        float ax1 = boxA[0] - aW / 2f;
        float ay1 = boxA[1] - aH / 2f;
        float ax2 = boxA[0] + aW / 2f;
        float ay2 = boxA[1] + aH / 2f;
        float bx1 = boxB[0] - bW / 2f;
        float by1 = boxB[1] - bH / 2f;
        float bx2 = boxB[0] + bW / 2f;
        float by2 = boxB[1] + bH / 2f;

        // 交集
        float iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
        float ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
        float inter = iw * ih;

        // 并集与 IoU
        float union = aW * aH + bW * bH - inter;
        return union > 0 ? inter / union : 0;
    }

    /**
     * 距离惩罚混合相似度（YOLOv8-SMOT, arxiv 2507.12087）。
     * sim = bufferedIoU + DIST_WEIGHT×(1 - dist/diag)
     * 横向快移 IoU 骤降但目标实际位移不大时，中心距离近补分，让匹配通过防断锁。
     * 距离 0 补 DIST_WEIGHT，距离远（≥对角线）补 0。
     */
    private float simTrack(float[] boxA, float[] boxB, float bufRatio) {
        float iou = bufferedIoU(boxA, boxB, bufRatio);
        float dx = boxA[0] - boxB[0];
        float dy = boxA[1] - boxB[1];
        float dist = (float) Math.sqrt(dx * dx + dy * dy);
        float diagA = (float) Math.sqrt(boxA[2] * boxA[2] + boxA[3] * boxA[3]);
        float diagB = (float) Math.sqrt(boxB[2] * boxB[2] + boxB[3] * boxB[3]);
        float diag = Math.max(diagA, diagB);
        if (diag < 1e-6f) return iou;
        float distNorm = Math.min(1f, dist / diag);
        // 方案 A：MeMoSORT Mo-IoU 高度相似性（heightSim）。heightRatio ∈ (0,1]，
        // 越接近1（高度完全一致）heightSim=1；差异大（高度≥2倍差）heightSim=0。
        // FPS 场景人物身高相对稳定，重叠时"差不多高"是同一个人的强信号。
        float hA = boxA[3];
        float hB = boxB[3];
        float heightRatio = (hA < hB) ? hA / (hB + 1e-6f) : hB / (hA + 1e-6f);
        float heightNorm = Math.max(0f, (heightRatio - 0.5f) * 2f); // ratio∈[0.5,1]→0~1线性
        float heightSim = Math.min(1f, heightNorm);
        return iou + DIST_WEIGHT * (1f - distNorm) + HEIGHT_SIM_WEIGHT * heightSim;
    }

    /**
     * OC-SORT OCM（arXiv 2203.14360）方向一致性惩罚：
     * 候选相对预测位置的位移方向与历史速度方向夹角越背离惩罚越重（从 simTrack 得分中扣除）。
     * <p>
     * 低速不归零：速度 < OCM_SPEED_REF 仅降权至 0.5（真实低速大目标恰是「帧间位移噪声≈真实位移」、
     * 最需要方向约束的区间，OC-SORT 限制1）；静止目标速度是噪声，半额惩罚误伤可控。
     * <p>
     * 方向稳定门控：瞬时速度与 EMA 速度夹角 <60°（直线/缓弯）才启用——急转弯时
     * 正确候选位于旧速度方向后方，无条件惩罚会误伤正确框。
     * <p>
     * HIGH 匹配 / LOW 救援 / OATrack 回收三阶段共用：救援阶段恰好是 HIGH 失配、
     * 方向信息最有区分度的时刻。
     *
     * @return 应扣减的惩罚（≥0），无有效速度/方向时返回 0
     */
    private float ocmDirectionPenalty(float predCx, float predCy, float candCx, float candCy) {
        if (CMC_ENABLED && cameraMotionActive) return 0f;  // CMC：相机运动时方向信息不可靠，跳过方向惩罚
        float spdSq = trackVelX * trackVelX + trackVelY * trackVelY;
        if (spdSq <= 1e-12f) return 0f;
        float rawSpdSq = lastRawVx * lastRawVx + lastRawVy * lastRawVy;
        if (rawSpdSq <= 1e-12f) return 0f;
        // 方向稳定门控：瞬时速度与 EMA 速度夹角 <60°（cosθ > 0.5）
        if ((lastRawVx * trackVelX + lastRawVy * trackVelY)
                <= 0.5f * (float) Math.sqrt(rawSpdSq) * (float) Math.sqrt(spdSq)) return 0f;
        float relX = candCx - predCx;
        float relY = candCy - predCy;
        float relLen = (float) Math.sqrt(relX * relX + relY * relY);
        if (relLen <= 1e-6f) return 0f;
        float spd = (float) Math.sqrt(spdSq);
        float cosT = (relX * trackVelX + relY * trackVelY) / (relLen * spd);
        // 速度调制下限 0.5：静止噪声最多吃半额，真实低速（≈OCM_SPEED_REF）吃全量
        float speedMod = 0.5f + 0.5f * Math.min(1f, spd / OCM_SPEED_REF);
        return OCM_WEIGHT * (1f - cosT) * 0.5f * speedMod;
    }

    /**
     * OA-SORT OAM (Occlusion-Aware Module) 简化版（CVPR 2026, arxiv 2603.06034）：
     * 计算目标被其他 HIGH 检测框遮挡的面积比例。
     * - Depth Sorting：bbox 底部 y 越大越靠前（俯视相机成像原理）
     *   若 other_bottom > target_bottom + OAM_DEPTH_THRESHOLD，判定 other 在前面遮挡 target
     * - 遮挡系数 = 被遮挡面积 / target 自身面积
     * - 不实施 Gaussian Map（像素级权重开销大，简化版省略）
     * 内联计算，零内存分配。
     */
    private float computeOcclusion(float[] targetBox, float[][] highBoxes, int highCount) {
        float targetCx = targetBox[0];
        float targetCy = targetBox[1];
        float targetW = targetBox[2];
        float targetH = targetBox[3];
        float targetBottom = targetCy + targetH / 2f;
        float targetArea = targetW * targetH;
        if (targetArea < 1e-6f) return 0f;

        float targetX1 = targetCx - targetW / 2f;
        float targetY1 = targetCy - targetH / 2f;
        float targetX2 = targetCx + targetW / 2f;
        float targetY2 = targetCy + targetH / 2f;

        float occludedArea = 0f;
        for (int i = 0; i < highCount; i++) {
            float[] other = highBoxes[i];
            if (other == targetBox) continue;  // 引用比较跳过自身

            float otherBottom = other[1] + other[3] / 2f;
            // Depth Sorting：other 必须明显在 target 前面才算遮挡
            if (otherBottom <= targetBottom + OAM_DEPTH_THRESHOLD) continue;

            // 计算交集面积
            float ix1 = Math.max(targetX1, other[0] - other[2] / 2f);
            float iy1 = Math.max(targetY1, other[1] - other[3] / 2f);
            float ix2 = Math.min(targetX2, other[0] + other[2] / 2f);
            float iy2 = Math.min(targetY2, other[1] + other[3] / 2f);
            float interW = Math.max(0, ix2 - ix1);
            float interH = Math.max(0, iy2 - iy1);
            occludedArea += interW * interH;
        }

        // 多目标重叠时总遮挡面积不超过 target 自身
        if (occludedArea > targetArea) occludedArea = targetArea;
        return occludedArea / targetArea;
    }

    private float boxIou(float[] boxA, float[] boxB) {
        float ax1 = boxA[0] - boxA[2] / 2;
        float ay1 = boxA[1] - boxA[3] / 2;
        float ax2 = boxA[0] + boxA[2] / 2;
        float ay2 = boxA[1] + boxA[3] / 2;

        float bx1 = boxB[0] - boxB[2] / 2;
        float by1 = boxB[1] - boxB[3] / 2;
        float bx2 = boxB[0] + boxB[2] / 2;
        float by2 = boxB[1] + boxB[3] / 2;

        float x1 = Math.max(ax1, bx1);
        float y1 = Math.max(ay1, by1);
        float x2 = Math.min(ax2, bx2);
        float y2 = Math.min(ay2, by2);

        float interW = Math.max(0, x2 - x1);
        float interH = Math.max(0, y2 - y1);
        float inter = interW * interH;

        float areaA = boxA[2] * boxA[3];
        float areaB = boxB[2] * boxB[3];
        float union = areaA + areaB - inter;

        return union > 0 ? inter / union : 0;
    }

    private MappedByteBuffer loadModelFile(Context context, String modelName) throws IOException {
        // 优先从应用私有目录加载（运行时按需下载的模型），否则回退到 assets（打包内置模型）
        File localFile = ModelManager.getLocalFile(context, modelName);
        if (localFile.exists() && localFile.length() > 0) {
            try (FileInputStream fis = new FileInputStream(localFile)) {
                return fis.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, localFile.length());
            }
        }
        try (AssetFileDescriptor afd = context.getAssets().openFd(modelName);
             FileInputStream fis = new FileInputStream(afd.getFileDescriptor())) {
            FileChannel fileChannel = fis.getChannel();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.getStartOffset(), afd.getDeclaredLength());
        }
    }

    public int getRoiX() {
        return roiX;
    }

    public int getRoiY() {
        return roiY;
    }

    public int getRoiSize() {
        return roiSize;
    }

    /** 通知推理层「滑动跟随正在注入触摸」（画面整体平移），触发 CMC 轻量替代 */
    public void setCameraMotionActive(boolean active) {
        cameraMotionActive = active;
    }

    public Backend getBackend() {
        return backend;
    }

    private void closeHardwareDelegate() {
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
    }

    public void close() {
        // 先关第二实例（含 executor），避免并发期间释放主实例引发竞态
        if (secondExecutor != null) {
            secondExecutor.shutdownNow();
            try {
                secondExecutor.awaitTermination(500, TimeUnit.MILLISECONDS);
            } catch (InterruptedException ignored) {
            }
            secondExecutor = null;
        }
        if (interpreter2 != null) {
            interpreter2.close();
            interpreter2 = null;
        }
        closeSecondHardwareDelegate();
        if (scaledRoi2 != null) {
            scaledRoi2.recycle();
            scaledRoi2 = null;
        }
        inputBuffer2 = null;
        inputFloatBuffer2 = null;
        inputFloats2 = null;
        pixels2 = null;
        output2 = null;
        outputBoxes2 = null;
        outputScores2 = null;

        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        closeHardwareDelegate();
        if (scaledRoi != null) {
            scaledRoi.recycle();
            scaledRoi = null;
        }
        // 清空所有引用类型字段，避免 GC 根链上残留（close 后 PoseEstimator 实例本就该废弃）
        inputBuffer = null;
        inputFloatBuffer = null;
        inputFloats = null;
        pixels = null;
        pixelBytes = null;
        brightLut = null;
        scaleCanvas = null;
        scalePaint = null;
        scaleRect = null;
        scaleSrcRect = null;
        output = null;
        trackedBox = null;
        // 跟踪状态数值字段统一复位（含丢失计数、速度估计、外推自适应），
        // 与 takeover / MAX_TRACK_LOST 释放块口径一致，防实例复用残留状态
        trackLostFrames = 0;
        trackVelX = 0f;
        trackVelY = 0f;
        lastRawVx = 0f;
        lastRawVy = 0f;
        prevRawVx = 0f;
        prevRawVy = 0f;
        lastTrackTimeNanos = 0L;
        coastEngaged = false;
        adaptiveLeadSec = PREDICT_SECONDS;
        leadErrEma = 0f;
        lastLeadDx = 0f;
        lastLeadDy = 0f;
        leadValid = false;
        firstInferLogged = false;
        realStatsLogged = false;
        shadowLiftLogged = false;
        dualVerifyWarned = false;
        antiCrosshairWarned = false;
        BRIGHT_AUTO_ENABLED = true;
        DUAL_INFER_ENABLED = false;
        DUAL_VERIFY_ENABLED = false;
        ANTI_CROSSHAIR_ENABLED = false;
        originalWidth = 0;
        originalHeight = 0;
        roiX = 0;
        roiY = 0;
        roiSize = 0;
        // 重置静态图闪烁诊断窗口（临时字段，随实例释放）
        diagFrame = 0; diagSelected = 0; diagHighFrames = 0; diagLowFrames = 0;
        diagRejected = 0; diagCoast = 0; diagTopMax = 0f; diagTopMin = 1f; diagTopSum = 0;
    }
}
