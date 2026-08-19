package com.example.fplus;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
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

public class PoseEstimator {

    private static final String TAG = "PoseEstimator";
    private static final int INPUT_SIZE = 640;
    private static final float CONFIDENCE_THRESHOLD = 0.15f;
    private static final float MIN_AREA_THRESHOLD = 0.01f;
    // 距离准星（屏幕中心）的高斯权重标准差（归一化，越大锁定范围越宽）
    private static final float CENTER_SIGMA = 0.2f;
    // 当前锁定目标的连续性加成倍数（防止多目标间来回跳）
    private static final float TRACK_BONUS = 2.0f;
    // ROI（黄框）占短边的比例，缩小让识别区域更聚焦中心
    private static final float ROI_SCALE = 0.7f;
    // 目标丢失时 ROI 回中的平滑系数（每帧移动剩余距离的比例）
    private static final float RECENTER_SMOOTH = 0.2f;
    private static final float INV_255 = 1f / 255f;
    // 亮度/对比度增强：游戏场景偏暗，提亮有助于提升检测率
    private static final float BRIGHTNESS_GAIN = 1.3f;   // 对比度增益
    private static final float BRIGHTNESS_OFFSET = 25f;  // 亮度偏移（0~255）
    private static final float BRIGHTNESS_OFFSET_NORM = BRIGHTNESS_OFFSET * INV_255;

    private Interpreter interpreter;
    private GpuDelegate gpuDelegate;
    private NnApiDelegate nnApiDelegate;
    private Backend backend;
    private ByteBuffer inputBuffer;
    private FloatBuffer inputFloatBuffer;
    private float[] inputFloats;
    private int[] pixels;
    private Bitmap scaledRoi;
    private Canvas scaleCanvas;
    private Paint scalePaint;
    private Rect scaleRect;
    private Rect scaleSrcRect;
    private float[][][] output;

    // 输出维度顺序：true 表示 [1, channels, anchors]，false 表示 [1, anchors, channels]
    private boolean channelFirst = true;
    private int numChannels = 0;
    private int numAnchors = 0;

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
    private static final int MAX_TRACK_LOST = 20;
    private static final float TRACK_IOU_THRESHOLD = 0.2f;

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

        int[] inputShape = interpreter.getInputTensor(0).shape();
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
        Log.d(TAG, "input dtype=" + inputTensor.dataType()
                + ", shape: " + Arrays.toString(inputShape)
                + ", quant scale=" + inQ.getScale() + " zeroPoint=" + inQ.getZeroPoint());

        int[] outputShape = interpreter.getOutputTensor(0).shape();
        org.tensorflow.lite.Tensor outputTensor = interpreter.getOutputTensor(0);
        org.tensorflow.lite.Tensor.QuantizationParams outQ = outputTensor.quantizationParams();
        Log.d(TAG, "Output shape: " + Arrays.toString(outputShape)
                + ", dtype=" + outputTensor.dataType()
                + ", quant scale=" + outQ.getScale() + " zeroPoint=" + outQ.getZeroPoint());
        if (outputShape.length != 3) {
            interpreter.close();
            throw new IOException("Unexpected output shape: " + Arrays.toString(outputShape));
        }

        // 判断维度顺序：通道维（如 6）通常远小于 anchor 维（如 2100）
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
        Log.d(TAG, "channelFirst=" + channelFirst + ", channels=" + numChannels + ", anchors=" + numAnchors);

        output = new float[outputShape[0]][outputShape[1]][outputShape[2]];
        inputBuffer = ByteBuffer.allocateDirect(1 * 3 * inputSize * inputSize * 4);
        inputBuffer.order(ByteOrder.nativeOrder());
        inputBuffer.rewind();
        inputFloatBuffer = inputBuffer.asFloatBuffer();
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

        if (backend == Backend.GPU) {
            try {
                CompatibilityList compatibilityList = new CompatibilityList();
                if (!compatibilityList.isDelegateSupportedOnThisDevice()) {
                    Log.w(TAG, "GPU delegate 不受支持");
                    return false;
                }
                gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
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

    public PersonPose estimate(Bitmap bitmap) {
        originalWidth = bitmap.getWidth();
        originalHeight = bitmap.getHeight();

        long t0 = System.nanoTime();
        Bitmap resizedBitmap = preprocessBitmap(bitmap);
        long t1 = System.nanoTime();
        bitmapToByteBuffer(resizedBitmap);
        long t2 = System.nanoTime();
        interpreter.run(inputBuffer, output);
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
        int centerX = originalWidth / 2;
        int centerY = originalHeight / 2;
        int curCx = roiX + roiSize / 2;
        int curCy = roiY + roiSize / 2;
        int newCx = curCx + (int) ((centerX - curCx) * RECENTER_SMOOTH);
        int newCy = curCy + (int) ((centerY - curCy) * RECENTER_SMOOTH);
        roiX = Math.max(0, Math.min(originalWidth - roiSize, newCx - roiSize / 2));
        roiY = Math.max(0, Math.min(originalHeight - roiSize, newCy - roiSize / 2));
    }

    private void updateRoiCenter(float targetCx, float targetCy) {
        int half = roiSize / 2;
        int newX = (int) (targetCx - half);
        int newY = (int) (targetCy - half);
        roiX = Math.max(0, Math.min(originalWidth - roiSize, newX));
        roiY = Math.max(0, Math.min(originalHeight - roiSize, newY));
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
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int len = width * height;
        if (pixels == null || pixels.length != len) {
            pixels = new int[len];
        }
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        int total = len * 3;
        if (inputFloats == null || inputFloats.length != total) {
            inputFloats = new float[total];
        }

        // float32，值 [0, 1]（* 1/255），并做亮度/对比度增强
        if (inputNchw) {
            // NCHW [1,3,H,W]：R 全图、G 全图、B 全图
            for (int i = 0; i < len; i++) {
                int p = pixels[i];
                inputFloats[i] = brighten(((p >> 16) & 0xFF) * INV_255);
                inputFloats[len + i] = brighten(((p >> 8) & 0xFF) * INV_255);
                inputFloats[len * 2 + i] = brighten((p & 0xFF) * INV_255);
            }
        } else {
            // NHWC [1,H,W,3]：逐像素 R,G,B 连续
            for (int i = 0; i < len; i++) {
                int p = pixels[i];
                int base = i * 3;
                inputFloats[base] = brighten(((p >> 16) & 0xFF) * INV_255);
                inputFloats[base + 1] = brighten(((p >> 8) & 0xFF) * INV_255);
                inputFloats[base + 2] = brighten((p & 0xFF) * INV_255);
            }
        }

        // 一次性 bulk 写入 ByteBuffer，避免每像素多次 JNI 调用
        inputFloatBuffer.rewind();
        inputFloatBuffer.put(inputFloats);
        inputBuffer.rewind();
    }

    private static float brighten(float v) {
        v = v * BRIGHTNESS_GAIN + BRIGHTNESS_OFFSET_NORM;
        return v < 1f ? v : 1f;
    }

    private float getOutputValue(int channel, int anchor) {
        if (channelFirst) {
            return output[0][channel][anchor];
        } else {
            return output[0][anchor][channel];
        }
    }

    private PersonPose parseOutput() {
        PersonPose bestPose = null;
        float maxScore = -1f;
        int numPredictions = numAnchors;

        for (int i = 0; i < numPredictions; i++) {
            // 取所有类别（channel 4 到 numChannels-1）的最大置信度
            float boxConf = 0;
            for (int c = 4; c < numChannels; c++) {
                float conf = getOutputValue(c, i);
                if (conf > boxConf) {
                    boxConf = conf;
                }
            }
            if (boxConf < CONFIDENCE_THRESHOLD) {
                continue;
            }

            float cx = getOutputValue(0, i);
            float cy = getOutputValue(1, i);
            float w = getOutputValue(2, i);
            float h = getOutputValue(3, i);
            float area = w * h;

            if (area < MIN_AREA_THRESHOLD) {
                continue;
            }

            // 映射到原图像素坐标，跟踪与 ROI 跟随都基于绝对坐标
            float px = cx * roiSize + roiX;
            float py = cy * roiSize + roiY;
            float pw = w * roiSize;
            float ph = h * roiSize;
            float[] box = new float[]{px, py, pw, ph};

            // 距离准星（屏幕中心）权重：高斯衰减，离准星越近越优先（主导因素）
            float dx = (px - originalWidth / 2f) / originalWidth;
            float dy = (py - originalHeight / 2f) / originalHeight;
            float distToCenter = (float) Math.sqrt(dx * dx + dy * dy);
            float distWeight = (float) Math.exp(-(distToCenter * distToCenter)
                    / (2 * CENTER_SIGMA * CENTER_SIGMA));

            // 跟踪连续性：当前锁定目标加分，防止多目标间来回跳
            float trackBonus = 1.0f;
            if (trackedBox != null && boxIou(trackedBox, box) >= TRACK_IOU_THRESHOLD) {
                trackBonus = TRACK_BONUS;
            }

            float score = boxConf * distWeight * trackBonus;

            if (score > maxScore) {
                maxScore = score;
                bestPose = new PersonPose();
                bestPose.box = box;
            }
        }

        // 更新跟踪状态
        if (bestPose != null) {
            trackedBox = bestPose.box.clone();
            trackLostFrames = 0;
        } else if (trackedBox != null) {
            trackLostFrames++;
            if (trackLostFrames >= MAX_TRACK_LOST) {
                trackedBox = null;
                trackLostFrames = 0;
            }
        }

        return bestPose;
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
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        closeHardwareDelegate();
        if (scaledRoi != null) {
            scaledRoi.recycle();
            scaledRoi = null;
        }
    }
}
