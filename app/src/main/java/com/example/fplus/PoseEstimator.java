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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PoseEstimator {

    private static final String TAG = "PoseEstimator";
    private static final int INPUT_SIZE = 640;
    private static final float CONFIDENCE_THRESHOLD = 0.2f;
    private static final float MIN_AREA_THRESHOLD = 0.01f;
    private static final float CENTER_BIAS_WEIGHT = 0.5f;
    // ROI（黄框）占短边的比例，缩小让识别区域更聚焦中心
    private static final float ROI_SCALE = 0.7f;
    private static final float INV_255 = 1f / 255f;

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
        inputBuffer = ByteBuffer.allocateDirect(1 * 3 * INPUT_SIZE * INPUT_SIZE * 4);
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
                    "阶段耗时 pre=%.1fms convert=%.1fms infer=%.1fms parse=%.1fms 合计=%.1fms",
                    preprocessAccum / (double) timingFrames / 1_000_000,
                    convertAccum / (double) timingFrames / 1_000_000,
                    inferAccum / (double) timingFrames / 1_000_000,
                    parseAccum / (double) timingFrames / 1_000_000,
                    (preprocessAccum + convertAccum + inferAccum + parseAccum) / (double) timingFrames / 1_000_000));
            preprocessAccum = 0;
            convertAccum = 0;
            inferAccum = 0;
            parseAccum = 0;
            timingFrames = 0;
        }

        if (pose != null && pose.box != null) {
            boolean isPixelCoord = pose.box[0] > 1.5f || pose.box[1] > 1.5f ||
                    pose.box[2] > 1.5f || pose.box[3] > 1.5f;

            float bCx = pose.box[0];
            float bCy = pose.box[1];
            float bW = pose.box[2];
            float bH = pose.box[3];

            if (isPixelCoord) {
                bCx /= INPUT_SIZE;
                bCy /= INPUT_SIZE;
                bW /= INPUT_SIZE;
                bH /= INPUT_SIZE;
            }

            // 中心 ROI 按 1:1 缩放到输入尺寸，归一化坐标直接映射回 ROI，再平移到原图
            float origCx = bCx * roiSize + roiX;
            float origCy = bCy * roiSize + roiY;
            float origW = bW * roiSize;
            float origH = bH * roiSize;

            pose.box[0] = origCx / originalWidth;
            pose.box[1] = origCy / originalHeight;
            pose.box[2] = origW / originalWidth;
            pose.box[3] = origH / originalHeight;
        }

        return pose;
    }

    private Bitmap preprocessBitmap(Bitmap source) {
        int sw = source.getWidth();
        int sh = source.getHeight();
        roiSize = (int) (Math.min(sw, sh) * ROI_SCALE);

        // 固定中心 ROI（FPS 目标通常在准星附近，固定更稳，避免 ROI 移动导致的坐标系错位）
        roiX = (sw - roiSize) / 2;
        roiY = (sh - roiSize) / 2;

        if (scaledRoi == null) {
            scaledRoi = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
            scaleCanvas = new Canvas(scaledRoi);
            scalePaint = new Paint(Paint.FILTER_BITMAP_FLAG);
            scaleRect = new Rect(0, 0, INPUT_SIZE, INPUT_SIZE);
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

        // float32，值 [0, 1]（* 1/255）
        if (inputNchw) {
            // NCHW [1,3,H,W]：R 全图、G 全图、B 全图
            for (int i = 0; i < len; i++) {
                int p = pixels[i];
                inputFloats[i] = ((p >> 16) & 0xFF) * INV_255;
                inputFloats[len + i] = ((p >> 8) & 0xFF) * INV_255;
                inputFloats[len * 2 + i] = (p & 0xFF) * INV_255;
            }
        } else {
            // NHWC [1,H,W,3]：逐像素 R,G,B 连续
            for (int i = 0; i < len; i++) {
                int p = pixels[i];
                int base = i * 3;
                inputFloats[base] = ((p >> 16) & 0xFF) * INV_255;
                inputFloats[base + 1] = ((p >> 8) & 0xFF) * INV_255;
                inputFloats[base + 2] = (p & 0xFF) * INV_255;
            }
        }

        // 一次性 bulk 写入 ByteBuffer，避免每像素多次 JNI 调用
        inputFloatBuffer.rewind();
        inputFloatBuffer.put(inputFloats);
        inputBuffer.rewind();
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
        int numCandidates = 0;

        int numPredictions = numAnchors;

        List<float[]> candidateBoxes = new ArrayList<>();

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

            numCandidates++;
            float areaScore = area;
            float centerDist = (float) Math.sqrt(
                    (cx - 0.5f) * (cx - 0.5f) + (cy - 0.5f) * (cy - 0.5f));
            float centerBonus = 1.0f + CENTER_BIAS_WEIGHT * Math.max(0, 1.0f - centerDist * 2.0f);
            float score = boxConf * areaScore * centerBonus;

            float[] box = new float[]{cx, cy, w, h};
            candidateBoxes.add(box);

            if (score > maxScore) {
                maxScore = score;
                bestPose = new PersonPose();
                bestPose.box = box;
            }
        }

        // 跟踪逻辑
        if (trackedBox != null && numCandidates > 0) {
            int bestMatchIdx = -1;
            float bestIou = 0;
            for (int i = 0; i < candidateBoxes.size(); i++) {
                float iou = boxIou(trackedBox, candidateBoxes.get(i));
                if (iou > bestIou) {
                    bestIou = iou;
                    bestMatchIdx = i;
                }
            }

            if (bestIou >= TRACK_IOU_THRESHOLD && bestMatchIdx >= 0) {
                bestPose = new PersonPose();
                bestPose.box = candidateBoxes.get(bestMatchIdx);
                trackedBox = candidateBoxes.get(bestMatchIdx).clone();
                trackLostFrames = 0;
            } else {
                trackLostFrames++;
                if (trackLostFrames >= MAX_TRACK_LOST) {
                    trackedBox = null;
                    trackLostFrames = 0;
                } else {
                    bestPose = null;
                }
            }
        } else if (bestPose != null) {
            trackedBox = bestPose.box.clone();
            trackLostFrames = 0;
        }

        if (bestPose == null && trackedBox != null) {
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
