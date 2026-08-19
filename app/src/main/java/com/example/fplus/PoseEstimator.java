package com.example.fplus;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

    private Interpreter interpreter;
    private GpuDelegate gpuDelegate;
    private boolean useGpu;
    private ByteBuffer inputBuffer;
    private float[][][] output;

    // 输出维度顺序：true 表示 [1, channels, anchors]，false 表示 [1, anchors, channels]
    private boolean channelFirst = true;
    private int numChannels = 0;
    private int numAnchors = 0;

    private int roiX;
    private int roiY;
    private int roiSize;

    private int originalWidth;
    private int originalHeight;

    private float[] trackedBox;
    private int trackLostFrames = 0;
    private static final int MAX_TRACK_LOST = 20;
    private static final float TRACK_IOU_THRESHOLD = 0.2f;

    public static class PersonPose {
        public float[] box;
    }

    public PoseEstimator(Context context, boolean useGpu) throws IOException {
        MappedByteBuffer modelBuffer = loadModelFile(context, "sunxds_0.8.0.tflite");

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);

        GpuDelegate gpuDelegate = null;
        boolean gpuActive = false;

        if (useGpu) {
            try {
                CompatibilityList compatibilityList = new CompatibilityList();
                if (compatibilityList.isDelegateSupportedOnThisDevice()) {
                    gpuDelegate = new GpuDelegate();
                    options.addDelegate(gpuDelegate);
                    gpuActive = true;
                } else {
                    Log.w(TAG, "GPU delegate 不受支持，回退 CPU");
                }
            } catch (Throwable t) {
                Log.w(TAG, "创建 GPU delegate 失败，回退 CPU", t);
                if (gpuDelegate != null) {
                    gpuDelegate.close();
                    gpuDelegate = null;
                }
                gpuActive = false;
            }
        }

        this.gpuDelegate = gpuDelegate;
        this.useGpu = gpuActive;

        try {
            interpreter = new Interpreter(modelBuffer, options);
        } catch (Throwable t) {
            if (gpuActive) {
                // 模型含 GPU 不支持的算子（如 YOLOv12 注意力结构），回退 CPU
                Log.w(TAG, "GPU 模式下模型加载失败，回退 CPU", t);
                gpuDelegate.close();
                this.gpuDelegate = null;
                this.useGpu = false;
                Interpreter.Options cpuOptions = new Interpreter.Options();
                cpuOptions.setNumThreads(4);
                modelBuffer.rewind();
                interpreter = new Interpreter(modelBuffer, cpuOptions);
            } else {
                throw new IOException("模型加载失败", t);
            }
        }

        Log.d(TAG, (this.useGpu ? "GPU" : "CPU") + " interpreter initialized");

        int[] outputShape = interpreter.getOutputTensor(0).shape();
        Log.d(TAG, "Output shape: " + Arrays.toString(outputShape));
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
    }

    public PersonPose estimate(Bitmap bitmap) {
        originalWidth = bitmap.getWidth();
        originalHeight = bitmap.getHeight();

        Bitmap resizedBitmap = preprocessBitmap(bitmap);
        bitmapToByteBuffer(resizedBitmap);
        resizedBitmap.recycle();
        interpreter.run(inputBuffer, output);
        PersonPose pose = parseOutput();

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
        // 只识别中心正方形区域（FPS 目标通常在准星附近）
        roiSize = Math.min(sw, sh);
        roiX = (sw - roiSize) / 2;
        roiY = (sh - roiSize) / 2;

        Bitmap roi = Bitmap.createBitmap(source, roiX, roiY, roiSize, roiSize);
        Bitmap scaled = Bitmap.createScaledBitmap(roi, INPUT_SIZE, INPUT_SIZE, true);
        roi.recycle();
        return scaled;
    }

    private void bitmapToByteBuffer(Bitmap bitmap) {
        inputBuffer.rewind();
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        // tflite 输入为 NCHW [1, 3, 640, 640]，float32，值 [0, 1]（/255）
        // 先填 R 全图，再 G 全图，再 B 全图
        for (int i = 0; i < pixels.length; i++) {
            inputBuffer.putFloat(((pixels[i] >> 16) & 0xFF) / 255.0f);
        }
        for (int i = 0; i < pixels.length; i++) {
            inputBuffer.putFloat(((pixels[i] >> 8) & 0xFF) / 255.0f);
        }
        for (int i = 0; i < pixels.length; i++) {
            inputBuffer.putFloat((pixels[i] & 0xFF) / 255.0f);
        }
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

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
    }
}
