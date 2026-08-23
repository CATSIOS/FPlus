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
    // ROI（黄框）占短边的比例，缩小让识别区域更聚焦中心
    private float ROI_SCALE = 0.7f;
    // 目标丢失时 ROI 回中的平滑系数（每帧移动剩余距离的比例）
    private static final float RECENTER_SMOOTH = 0.2f;
    private static final float INV_255 = 1f / 255f;
    // 亮度/对比度增强：游戏场景偏暗，提亮有助于提升检测率
    // USER_* = 高级选项用户手动设置的基础值；CUR_* = 当前帧实际应用值（USER_* 乘动态补偿倍率 mult）
    private float USER_BRIGHT_GAIN = 1.3f;
    private float USER_BRIGHT_OFFSET = 25f;
    private float CUR_BRIGHT_GAIN = 1.3f;
    private float CUR_BRIGHT_OFFSET = 25f;
    private float CUR_BRIGHT_OFFSET_NORM;
    // 动态亮度配置：目标平均亮度(0~255)、EMA平滑系数、重建LUT的变化阈值、补偿倍率上下限
    private static final float TARGET_AVG_LUMA = 90f;
    private static final float LUM_EMA_ALPHA = 0.3f;
    private static final float REBUILD_GAIN_RATIO = 0.05f;  // ±5% 变化才重建 LUT
    private static final float REBUILD_OFFSET_ABS = 3.0f;    // 绝对值±3 才重建
    private static final float MIN_MULT = 0.75f;
    private static final float MAX_MULT = 1.8f;
    // 动态亮度状态字段
    private float emaAvgL = -1f;  // -1 表示未初始化
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
    // convert 零-copy 优化：copyPixelsToBuffer 一次 memcpy 拿 RGBA byte，跳过 getPixels 的 int 打包；
    // brightLut 亮度查表（256 项），省每像素浮点乘加 clamp，遍历只做数组读
    private byte[] pixelBytes;
    private ByteBuffer pixelCopyBuffer;
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
    private int estimateCallCount = 0;          // 调用计数：probe=1, 真实画面首帧=2

    // ===== PoC: 双实例并发推理（榨 GPU 并行算力）=====
    // 目的：验证 TFLite 双 GpuDelegate 实例能否真并发，榨干 GPU 空闲算力且端到端不卡。
    // 第二路跑「中心放大区」（ROI 中心 zoomSize 正方形缩放到 640），远处小目标放大后更易检出。
    // 第二套资源与主路完全隔离，避免并发时 pixels/floats/buffer/canvas 字段冲突。
    // PoC1 阶段：不融合第二路结果到主路跟踪，仅日志对比检出差异 + 并发性能数据。
    private boolean DUAL_INFER_ENABLED = false;          // 开关：默认关，保证开关关时行为与单实例逐字节一致
    private float DUAL_ZOOM = 0.5f;                       // 第二路区域相对主 ROI 边长比例（越小放大倍数越大）
    private Interpreter interpreter2;
    private GpuDelegate gpuDelegate2;
    private ByteBuffer inputBuffer2;
    private FloatBuffer inputFloatBuffer2;
    private float[] inputFloats2;
    private int[] pixels2;
    private float[][][] output2;
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
    // C-BIoU Tracker（roboflow 2026 benchmark, HOTA 63.0 > OC-SORT 61.9）
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
    // 自适应相似度度量（YOLOv8-SMOT, arxiv 2507.12087）：匹配前对两框各方向扩展 EXPAND_RATIO
    // 提升小目标重叠率，避免仅几像素位移导致 IoU 骤降而跟踪断锁
    private static final float EXPAND_RATIO = 0.15f;
    // OC-SORT OCM：跟踪目标的速度估计（像素/帧），用于匹配前按 lost 帧数做位置外推
    private float trackVelX = 0f;
    private float trackVelY = 0f;
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
    // MIN_VEL：速度低于此值视为静止，不外推（避免静止时绿框漂移）
    //   - trackVelX 单位为 60fps 基准每帧位移，0.001 相当于 60fps 下 0.1% 图像宽/帧
    private static final float MIN_VEL = 0.001f;
    // OC-SORT Coast（速度外推）参数：丢失期间用速度外推位置继续显示绿框
    //   COAST_MIN_VEL：速度低于此值视为静止丢失，不外推直接淡出（目标真没了）
    //   COAST_MAX_FRAMES：外推最多持续帧数，超过则淡出（防持续飘移）
    //   仅在「明显移动 + 短时丢失」时外推衔接，解决大幅移动瞬时丢检测的闪烁
    private float COAST_MIN_VEL = 0.01f;
    private int COAST_MAX_FRAMES = 2;
    // OA-SORT（CVPR 2026, arxiv 2603.06034）：OAM 深度排序阈值（像素），
    // bbox 底部 y 差值小于此值不判定遮挡，避免抖动误判
    private static final float OAM_DEPTH_THRESHOLD = 5f;
    // OA-SORT BAM：上一帧 bestPose 的遮挡系数 [0,1]，本帧 trackedBox 更新时降低对观测的信任
    private float lastBestOcclusion = 0f;

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
        CONFIDENCE_THRESHOLD = parseFloat(prefs, "det_conf", 0.15f);
        VALID_DETECTION_CONF = parseFloat(prefs, "det_valid", 0.2f);
        MIN_AREA_THRESHOLD = parseFloat(prefs, "det_min_area", 0.01f);
        TRACK_IOU_THRESHOLD = parseFloat(prefs, "track_iou", 0.2f);
        CBIoU_BUF_HIGH = parseFloat(prefs, "track_buf_high", 0.3f);
        CBIoU_BUF_LOW = parseFloat(prefs, "track_buf_low", 0.5f);
        CBIoU_IOU_LOW = parseFloat(prefs, "track_iou_low", 0.3f);
        CBIoU_SPEED_REF = parseFloat(prefs, "track_speed_ref", 0.25f);
        DIST_WEIGHT = parseFloat(prefs, "track_dist_weight", 0.25f);
        ACC_THRESHOLD = parseFloat(prefs, "track_acc_threshold", 0.002f);
        MAX_TRACK_LOST = parseInt(prefs, "track_max_lost", 30);
        TAKEOVER_CONF = parseFloat(prefs, "track_takeover", 0.5f);
        PREDICT_SECONDS = parseFloat(prefs, "pred_seconds", 0.083f);
        COAST_MIN_VEL = parseFloat(prefs, "coast_min_vel", 0.01f);
        COAST_MAX_FRAMES = parseInt(prefs, "coast_max_frames", 2);
        CENTER_SIGMA = parseFloat(prefs, "score_center_sigma", 0.2f);
        SCORE_W = parseFloat(prefs, "score_w", 0.3f);
        ROI_SCALE = parseFloat(prefs, "roi_scale", 0.7f);
        USER_BRIGHT_GAIN = parseFloat(prefs, "bright_gain", 1.3f);
        USER_BRIGHT_OFFSET = parseFloat(prefs, "bright_offset", 25f);
        CUR_BRIGHT_GAIN = USER_BRIGHT_GAIN;
        CUR_BRIGHT_OFFSET = USER_BRIGHT_OFFSET;
        CUR_BRIGHT_OFFSET_NORM = CUR_BRIGHT_OFFSET * INV_255;
        emaAvgL = -1f;                 // 重置 EMA，下一帧按新场景重新收敛
        lastRebuildGain = -1f;         // 下一次必触发 rebuildBrightLut（用新 CUR_*）
        lastRebuildOffset = -1f;
        rebuildBrightLut();  // 亮度参数变化后重建 LUT，convert 遍历用查表替代浮点乘加
        // 双实例并发 PoC 开关（"1"=开），zoom 比例
        DUAL_INFER_ENABLED = "1".equals(prefs.getString("dual_infer", "0"));
        DUAL_ZOOM = parseFloat(prefs, "dual_zoom", 0.5f);
    }

    private float parseFloat(SharedPreferences prefs, String key, float def) {
        try {
            return Float.parseFloat(prefs.getString(key, Float.toString(def)));
        } catch (NumberFormatException e) {
            Log.w(TAG, "参数 " + key + " 解析失败，使用默认值 " + def);
            return def;
        }
    }

    private int parseInt(SharedPreferences prefs, String key, int def) {
        try {
            return Integer.parseInt(prefs.getString(key, Integer.toString(def)));
        } catch (NumberFormatException e) {
            Log.w(TAG, "参数 " + key + " 解析失败，使用默认值 " + def);
            return def;
        }
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
        output2 = new float[output.length][output[0].length][output[0][0].length];
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

        // 第二路结果收集 + 并发性能对比日志（PoC1 阶段不融合，主路 pose 不变）
        if (f2 != null) {
            try {
                PersonPose pose2 = f2.get();
                long dualEnd = System.nanoTime();
                long mainInferMs = (t3 - t2) / 1_000_000;
                long secondInferMs = secondInferNanos.get() / 1_000_000;
                long dualTotalMs = (dualEnd - dualStart) / 1_000_000;
                dualMainInferAccum += mainInferMs;
                dual2ndInferAccum += secondInferMs;
                dualTotalAccum += dualTotalMs;
                dualFrames++;
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

        if (pose != null && pose.box != null) {
            // parseOutput 返回原图像素坐标 [cx, cy, w, h]
            float px = pose.box[0];
            float py = pose.box[1];
            float pw = pose.box[2];
            float ph = pose.box[3];

            // ROI 跟随：下一帧 ROI 中心移到目标中心，避免主角移出 ROI 而丢失
            updateRoiCenter(px, py);

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

    private void updateRoiCenter(float targetCx, float targetCy) {
        float curCx = roiX + roiSize / 2f;
        float curCy = roiY + roiSize / 2f;
        float dx = targetCx - curCx;
        float dy = targetCy - curCy;
        float dist = (float) Math.sqrt(dx * dx + dy * dy);

        // 死区：目标在 ROI 中心 8% 范围内不移动，减少黄框抖动
        float deadZone = roiSize * 0.08f;
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
        if (pixelBytes == null || pixelBytes.length != byteLen || pixelCopyBuffer == null) {
            pixelBytes = new byte[byteLen];
            pixelCopyBuffer = ByteBuffer.allocateDirect(byteLen);
            pixelCopyBuffer.order(ByteOrder.nativeOrder());
        }
        // copyPixelsToBuffer 一次 memcpy 拿 RGBA byte，跳过 getPixels 的逐像素 JNI int 打包
        pixelCopyBuffer.rewind();
        bitmap.copyPixelsToBuffer(pixelCopyBuffer);
        pixelCopyBuffer.rewind();
        pixelCopyBuffer.get(pixelBytes, 0, byteLen);

        // ===== 动态亮度增益：抽样算平均亮度 → EMA → mult → 超阈值重建 LUT（每帧主路只算一次）=====
        if (!dynBrightComputed) {
            long sumL = 0;
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
                    sumL += (299 * r + 587 * g + 114 * b) / 1000;
                    sampleCount++;
                }
            }
            int avgL = sampleCount > 0 ? (int) (sumL / sampleCount) : 128;
            // EMA 平滑，防止帧间小抖动导致 GAIN 跳变
            if (emaAvgL < 0f) emaAvgL = avgL;
            else emaAvgL = emaAvgL + LUM_EMA_ALPHA * (avgL - emaAvgL);
            // 补偿倍率：想把平均亮度拉到 TARGET_AVG_LUMA，暗画面 mult > 1，亮画面 mult < 1
            float mult = TARGET_AVG_LUMA / (emaAvgL + 8f);
            if (mult < MIN_MULT) mult = MIN_MULT;
            else if (mult > MAX_MULT) mult = MAX_MULT;
            CUR_BRIGHT_GAIN = USER_BRIGHT_GAIN * mult;
            CUR_BRIGHT_OFFSET = USER_BRIGHT_OFFSET * mult;
            CUR_BRIGHT_OFFSET_NORM = CUR_BRIGHT_OFFSET * INV_255;
            // 变化过小不重建（256次浮点虽快，但没必要每帧做）
            // lastRebuildGain < 0 表示首次（必重建）；增益接近 0 时改用绝对差避免除零
            boolean gainChanged = (lastRebuildGain < 0f)
                    || (Math.abs(lastRebuildGain) < 1e-4f
                        ? Math.abs(CUR_BRIGHT_GAIN - lastRebuildGain) > REBUILD_GAIN_RATIO * 0.5f
                        : Math.abs(CUR_BRIGHT_GAIN / lastRebuildGain - 1f) > REBUILD_GAIN_RATIO);
            boolean offsetChanged = Math.abs(CUR_BRIGHT_OFFSET - lastRebuildOffset) > REBUILD_OFFSET_ABS;
            boolean needRebuild = gainChanged || (lastRebuildOffset < 0f) || offsetChanged;
            if (needRebuild) {
                rebuildBrightLut();
                lastRebuildGain = CUR_BRIGHT_GAIN;
                lastRebuildOffset = CUR_BRIGHT_OFFSET;
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
            interpreter2.run(inputBuffer2, output2);
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
        if (channelFirst) {
            return output2[0][channel][anchor];
        } else {
            return output2[0][anchor][channel];
        }
    }

    /**
     * 第二路精简解析：遍历 anchors，选 conf>=VALID 的最高分（距离中心+置信度），
     * box 反变换到原图像素坐标。不做跟踪/OCM/C-BIoU（那是主路职责）。
     * 返回 PersonPose.box=[px,py,pw,ph,conf]，仅用于 PoC1 日志检出对比。
     */
    private PersonPose parseOutput2() {
        int numPredictions = numAnchors;
        PersonPose best = null;
        float maxScore = -1f;
        for (int i = 0; i < numPredictions; i++) {
            // OPT和非OPT的conf通道输出模式一致（均含sigmoid），统一取ch4+的max
            float boxConf = 0;
            for (int c = 4; c < numChannels; c++) {
                float conf = getOutputValue2(c, i);
                if (conf > boxConf) boxConf = conf;
            }
            if (boxConf < VALID_DETECTION_CONF) continue;
            float cx, cy, w, h;
            if (boxIsXyxy) {
                // OPT 模型：x1y1x2y2 corner（像素值）→ ÷inputSize 归一化 → cxcywh
                float x1 = getOutputValue2(0, i);
                float y1 = getOutputValue2(1, i);
                float x2 = getOutputValue2(2, i);
                float y2 = getOutputValue2(3, i);
                float invSz = 1f / inputSize;
                cx = (x1 + x2) * 0.5f * invSz;
                cy = (y1 + y2) * 0.5f * invSz;
                w = (x2 - x1) * invSz;
                h = (y2 - y1) * invSz;
                if (w <= 0f || h <= 0f) continue;
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
        return best;
    }

    /** 重建亮度查表（GAIN/OFFSET 变化时调用）：brightLut[v]=clamp(v*INV_255*CUR_GAIN+CUR_OFFSET_NORM) */
    private void rebuildBrightLut() {
        if (brightLut == null || brightLut.length != 256) {
            brightLut = new float[256];
        }
        float curGain = CUR_BRIGHT_GAIN;
        float curOffNorm = CUR_BRIGHT_OFFSET_NORM;
        for (int i = 0; i < 256; i++) {
            float v = i * INV_255;
            v = v * curGain + curOffNorm;
            if (v < 0f) v = 0f;
            else if (v > 1f) v = 1f;
            brightLut[i] = v;
        }
    }

    private float brighten(float v) {
        v = v * CUR_BRIGHT_GAIN + CUR_BRIGHT_OFFSET_NORM;
        // clamp 到 [0,1]：用户在高级选项可能输入负 BRIGHTNESS_OFFSET，
        // 不 clamp 下界会让模型输入出现负值，产生异常输出
        if (v < 0f) return 0f;
        if (v > 1f) return 1f;
        return v;
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
                pxC = trackedBox[0] + lastRawVx * steps + 0.5f * accX * steps * steps;
                pyC = trackedBox[1] + lastRawVy * steps + 0.5f * accY * steps * steps;
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
        if (trackedBox != null && trackLostFrames < MAX_TRACK_LOST && trackedBox[2] > 1e-3f) {
            float speed = (float) Math.sqrt(lastRawVx * lastRawVx + lastRawVy * lastRawVy);
            float speedNorm = speed / trackedBox[2];
            float t = speedNorm / CBIoU_SPEED_REF;
            if (t > 1f) t = 1f;
            dynHighBuffer = CBIoU_BUF_HIGH + (CBIoU_BUF_LOW - CBIoU_BUF_HIGH) * t;
        }

        // ===== C-BIoU 第一阶段：高置信度（conf >= HIGH=VALID_DETECTION_CONF）正常评分 =====
        PersonPose bestPose = null;
        float maxScore = -1f;
        float bestConf = 0f;
        boolean bestMatchedTrack = false;

        for (int i = 0; i < numPredictions; i++) {
            // OPT和非OPT的conf通道输出模式一致（均含sigmoid），统一取ch4+的max
            float boxConf = 0;
            for (int c = 4; c < numChannels; c++) {
                float conf = getOutputValue(c, i);
                if (conf > boxConf) {
                    boxConf = conf;
                }
            }
            // ByteTrack: 第一阶段阈值用 HIGH；低于 HIGH 但 >= LOW 暂存为候选项
            if (boxConf < BYTETRACK_LOW_CONF) {
                continue;
            }

            float cx, cy, w, h;
            if (boxIsXyxy) {
                // OPT 模型：前四通道是 x1,y1,x2,y2（**像素值** corner）→ 先÷inputSize 归一化，再转 cxcywh
                float x1 = getOutputValue(0, i);
                float y1 = getOutputValue(1, i);
                float x2 = getOutputValue(2, i);
                float y2 = getOutputValue(3, i);
                float invSz = 1f / inputSize;
                cx = (x1 + x2) * 0.5f * invSz;
                cy = (y1 + y2) * 0.5f * invSz;
                w = (x2 - x1) * invSz;
                h = (y2 - y1) * invSz;
                // corner 格式数值非法时跳过（防止 x2<x1 产生负面积）
                if (w <= 0f || h <= 0f) continue;
            } else {
                // 旧模型：默认 cxcywh 格式（已归一化 0~1）
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
                float distToCenter = (float) Math.sqrt(dx * dx + dy * dy);
                float distWeight = (float) Math.exp(-(distToCenter * distToCenter)
                        / (2 * CENTER_SIGMA * CENTER_SIGMA));

                // 方案3：水平偏心角度分数（Best Target Selection, Nicholas Gorski）
                // FPS 横向瞄准是关键，垂直偏心不惩罚（敌人可能在上下半部分）
                float angleWeight = (float) Math.exp(-(dx * dx)
                        / (2 * ANGLE_SIGMA * ANGLE_SIGMA));
                // 组合：score = (1-W)*dist + W*angle，距离为主角度为辅
                float posWeight = (1f - SCORE_W) * distWeight + SCORE_W * angleWeight;

                // C-BIoU：跟踪加分用预测后的 trackedBox 做 buffered IoU（HIGH 阶段，速度自适应 buffer）
                float trackBonus = 1.0f;
                boolean matched = predTracked != null
                        && simTrack(predTracked, box, dynHighBuffer) >= TRACK_IOU_THRESHOLD;
                if (matched) {
                    trackBonus = TRACK_BONUS;
                }

                float score = boxConf * posWeight * trackBonus;
                if (score > maxScore) {
                    maxScore = score;
                    bestConf = boxConf;
                    bestPose = new PersonPose();
                    bestPose.box = box;
                    bestMatchedTrack = matched;
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
                float sim = simTrack(predTracked, lbBox, CBIoU_BUF_LOW);
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
                // 与 HIGH/LOW 策略一致，进一步提高短遮挡场景的恢复率
                float sim = simTrack(predTracked, lbBox, RECOVERY_BUF);
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

        // HIGH 阶段若已选到"非跟踪目标"的高分框（例如画面边缘另一个人突然出现），
        // 但当前仍在锁定中，我们不应该抢走锁定。逻辑：在 trackedBox 仍有效（lostFrames 不大）
        // 情况下，如果 bestMatchedTrack == false 说明 HIGH 选的不是当前目标，保持旧 trackedBox
        // 例外：若新目标置信度 ≥ TAKEOVER_CONF（清晰人物突然入场），允许立即接管，
        // 避免被压制 MAX_TRACK_LOST 帧造成的 0.5s+ 入场延迟
        if (trackedBox != null && trackLostFrames < MAX_TRACK_LOST && !bestMatchedTrack
                && bestPose != null && !rescuedByLow && bestConf < TAKEOVER_CONF) {
            // 不接受这个新目标（把 bestPose 置空，交给 lost++ 逻辑，让跟踪继续沿用旧轨迹）
            bestPose = null;
            bestConf = 0f;
        }

        // ===== OA-SORT OAM：计算当前 bestPose 被其他 HIGH 检测框遮挡的系数 =====
        // 用于下一帧 BAM 加权：遮挡时降低对观测的信任，更多依赖 predTracked
        float currentOcclusion = 0f;
        if (bestPose != null && highCount > 1) {
            currentOcclusion = computeOcclusion(bestPose.box, highBoxes, highCount);
        }

        // ===== 更新速度估计 & 跟踪状态 =====
        long nowNanos = System.nanoTime();
        if (bestPose != null) {
            // OA-SORT BAM (Bias-Aware Momentum)：
            // BAM = IoU(predTracked, Z) · (1 - Oc_prev)
            // Z' = BAM · Z + (1 - BAM) · predTracked
            // 遮挡严重（Oc_prev 大）或观测与预测位置偏离（IoU 低）时，BAM 减小，
            // trackedBox 更多沿用 predTracked 速度外推，防止遮挡中漂移的检测框污染轨迹
            float[] newBox = bestPose.box.clone();
            if (trackedBox != null && predTracked != null && predTracked != trackedBox) {
                float bam = boxIou(predTracked, newBox) * (1f - lastBestOcclusion);
                if (bam < 0f) bam = 0f;
                else if (bam > 1f) bam = 1f;
                if (bam < 1f) {
                    for (int k = 0; k < 4; k++) {
                        newBox[k] = bam * newBox[k] + (1f - bam) * predTracked[k];
                    }
                }
            }
            // 速度估计基于混合后的 newBox，保持与 trackedBox 一致性
            if (lastTrackTimeNanos != 0L && trackedBox != null) {
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
            }
            trackedBox = newBox;
            lastBestOcclusion = currentOcclusion;
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
                lastTrackTimeNanos = 0L;
                lastBestOcclusion = 0f;
            }
        }

        // ===== 预测未来位置（方案 A + B）=====
        // 用当前速度外推 PREDICT_SECONDS 秒后的位置，抵消检测→显示端到端延迟
        // 静止目标（速度低于 MIN_VEL）不外推，避免静止时绿框漂移（方案 B）
        // 注意：bestPose.box 是独立 new 出来的引用，trackedBox 已 clone，互不影响
        // trackVelX 单位：60fps 基准每帧位移，× 60 转换为"每秒位移"再 × PREDICT_SECONDS
        if (bestPose != null && trackedBox != null) {
            float speedSq = trackVelX * trackVelX + trackVelY * trackVelY;
            if (speedSq >= MIN_VEL * MIN_VEL) {
                float predDx = trackVelX * 60f * PREDICT_SECONDS;
                float predDy = trackVelY * 60f * PREDICT_SECONDS;
                bestPose.box[0] += predDx;
                bestPose.box[1] += predDy;
            }
        }

        // OC-SORT Coast（速度外推）：丢失期间用瞬时速度外推位置继续显示绿框，
        // 仅在「明显移动（速度≥COAST_MIN_VEL）+ 短时丢失（lost≤COAST_MAX_FRAMES）」时外推，
        // 解决大幅移动瞬时丢检测的闪烁；静止丢失或持续丢失直接 return null 触发淡出。
        // 用 lastRawVx/Y（瞬时）而非 trackVelX（EMA）：EMA 变向滞后，会导致角色变向丢失时
        // 绿框沿旧方向反向滑动；瞬时速度方向正确
        if (bestPose == null && trackedBox != null
                && trackLostFrames <= COAST_MAX_FRAMES) {
            float speedSq = lastRawVx * lastRawVx + lastRawVy * lastRawVy;
            if (speedSq >= COAST_MIN_VEL * COAST_MIN_VEL) {
                float steps = Math.min(trackLostFrames + 1, COAST_MAX_FRAMES);
                bestPose = new PersonPose();
                bestPose.box = new float[] {
                        trackedBox[0] + lastRawVx * steps,
                        trackedBox[1] + lastRawVy * steps,
                        trackedBox[2],
                        trackedBox[3]
                };
            }
        }

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
     * @deprecated 已由 C-BIoU {@link #bufferedIoU} 替代，保留以便回退对比。
     * 自适应相似度度量（YOLOv8-SMOT, arxiv 2507.12087）：
     * 在 EIoU 基础上对两框各方向扩展 EXPAND_RATIO 后计算 IoU，再叠加中心距离惩罚。
     * - bbox 扩展提升小目标重叠率（远处小人仅几像素位移 IoU 不再骤降为 0）
     * - 中心距离惩罚 ρ²/c² 对近邻但不重叠的框仍给出非零相似度，避免跟踪断锁
     * 整体内联实现，零内存分配，避免 GC 压力。
     */
    @Deprecated
    private float boxEIoU(float[] boxA, float[] boxB) {
        // 扩展两框：中心不变，宽高 × (1 + 2*EXPAND_RATIO)
        float eaW = boxA[2] * (1f + 2f * EXPAND_RATIO);
        float eaH = boxA[3] * (1f + 2f * EXPAND_RATIO);
        float ebW = boxB[2] * (1f + 2f * EXPAND_RATIO);
        float ebH = boxB[3] * (1f + 2f * EXPAND_RATIO);

        // 扩展后两框的角点
        float ax1 = boxA[0] - eaW / 2f;
        float ay1 = boxA[1] - eaH / 2f;
        float ax2 = boxA[0] + eaW / 2f;
        float ay2 = boxA[1] + eaH / 2f;
        float bx1 = boxB[0] - ebW / 2f;
        float by1 = boxB[1] - ebH / 2f;
        float bx2 = boxB[0] + ebW / 2f;
        float by2 = boxB[1] + ebH / 2f;

        // 交集
        float x1 = Math.max(ax1, bx1);
        float y1 = Math.max(ay1, by1);
        float x2 = Math.min(ax2, bx2);
        float y2 = Math.min(ay2, by2);
        float interW = Math.max(0, x2 - x1);
        float interH = Math.max(0, y2 - y1);
        float inter = interW * interH;

        // 并集与 IoU
        float areaA = eaW * eaH;
        float areaB = ebW * ebH;
        float union = areaA + areaB - inter;
        float iou = union > 0 ? inter / union : 0;

        // EIoU 中心距离惩罚：ρ = 原始框中心点欧氏距离
        float dcx = boxA[0] - boxB[0];
        float dcy = boxA[1] - boxB[1];
        float rho2 = dcx * dcx + dcy * dcy;

        // 最小外接矩形对角线 c（用扩展后框计算）
        float ex1 = Math.min(ax1, bx1);
        float ey1 = Math.min(ay1, by1);
        float ex2 = Math.max(ax2, bx2);
        float ey2 = Math.max(ay2, by2);
        float c2 = (ex2 - ex1) * (ex2 - ex1) + (ey2 - ey1) * (ey2 - ey1);

        if (c2 < 1e-6f) return iou;
        return iou - rho2 / c2;
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
        pixelCopyBuffer = null;
        brightLut = null;
        scaleCanvas = null;
        scalePaint = null;
        scaleRect = null;
        scaleSrcRect = null;
        output = null;
        trackedBox = null;
        originalWidth = 0;
        originalHeight = 0;
        roiX = 0;
        roiY = 0;
        roiSize = 0;
    }
}
