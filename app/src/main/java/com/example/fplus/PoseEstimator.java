package com.example.fplus;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

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
    private static final int INPUT_SIZE = 320;
    private static final float CONFIDENCE_THRESHOLD = 0.2f;
    private static final float MIN_AREA_THRESHOLD = 0.01f;
    private static final float CENTER_BIAS_WEIGHT = 0.5f;

    private Interpreter interpreter;
    private ByteBuffer inputBuffer;
    private float[][][] output;

    // 输出维度顺序：true 表示 [1, channels, anchors]，false 表示 [1, anchors, channels]
    private boolean channelFirst = true;
    private int numChannels = 0;
    private int numAnchors = 0;

    private float letterboxScale;
    private int letterboxX;
    private int letterboxY;

    private int originalWidth;
    private int originalHeight;

    private float[] trackedBox;
    private int trackLostFrames = 0;
    private static final int MAX_TRACK_LOST = 20;
    private static final float TRACK_IOU_THRESHOLD = 0.2f;

    public static class Keypoint {
        public float x;
        public float y;
        public float confidence;

        public Keypoint(float x, float y, float confidence) {
            this.x = x;
            this.y = y;
            this.confidence = confidence;
        }
    }

    public static class PersonPose {
        public List<Keypoint> keypoints;
        public float boxConfidence;
        public float[] box;

        public PersonPose() {
            keypoints = new ArrayList<>();
        }
    }

    public PoseEstimator(Context context) throws IOException {
        // YOLOv12 模型含注意力机制等结构，TFLite GPU delegate 支持不佳导致推理错乱，
        // 改用 CPU 推理（模型仅 2.56M 参数 + 320 输入，CPU 足够快）
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(loadModelFile(context, "sunxds_0.8.0.tflite"), options);
        Log.d(TAG, "CPU interpreter initialized (4 threads)");

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

            float pixelCx = bCx * INPUT_SIZE;
            float pixelCy = bCy * INPUT_SIZE;
            float pixelW = bW * INPUT_SIZE;
            float pixelH = bH * INPUT_SIZE;

            float origCx = (pixelCx - letterboxX) / letterboxScale;
            float origCy = (pixelCy - letterboxY) / letterboxScale;
            float origW = pixelW / letterboxScale;
            float origH = pixelH / letterboxScale;

            pose.box[0] = origCx / originalWidth;
            pose.box[1] = origCy / originalHeight;
            pose.box[2] = origW / originalWidth;
            pose.box[3] = origH / originalHeight;

            Log.d(TAG, "Final box: cx=" + pose.box[0] + ", cy=" + pose.box[1] +
                    ", w=" + pose.box[2] + ", h=" + pose.box[3]);
            Log.d(TAG, "Letterbox: scale=" + letterboxScale + ", x=" + letterboxX + ", y=" + letterboxY);
            Log.d(TAG, "Original size: " + originalWidth + "x" + originalHeight);
        }

        return pose;
    }

    private Bitmap preprocessBitmap(Bitmap source) {
        float scale = Math.min((float) INPUT_SIZE / source.getWidth(), (float) INPUT_SIZE / source.getHeight());
        int newWidth = Math.round(source.getWidth() * scale);
        int newHeight = Math.round(source.getHeight() * scale);

        letterboxScale = scale;
        letterboxX = (INPUT_SIZE - newWidth) / 2;
        letterboxY = (INPUT_SIZE - newHeight) / 2;

        Bitmap scaled = Bitmap.createScaledBitmap(source, newWidth, newHeight, true);
        Bitmap letterboxed = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(letterboxed);
        canvas.drawColor(Color.rgb(114, 114, 114));
        canvas.drawBitmap(scaled, letterboxX, letterboxY, null);
        return letterboxed;
    }

    private void bitmapToByteBuffer(Bitmap bitmap) {
        inputBuffer.rewind();
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        // tflite 输入为 NCHW [1, 3, 320, 320]，float32，值 [0, 1]（/255）
        // 先填 R 全图，再 G 全图，再 B 全图
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                inputBuffer.putFloat(Color.red(pixels[y * width + x]) / 255.0f);
            }
        }
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                inputBuffer.putFloat(Color.green(pixels[y * width + x]) / 255.0f);
            }
        }
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                inputBuffer.putFloat(Color.blue(pixels[y * width + x]) / 255.0f);
            }
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

        float bestConf = 0;
        float bestArea = 0;

        List<float[]> candidateBoxes = new ArrayList<>();
        List<Float> candidateConfs = new ArrayList<>();

        // 调试：打印第一个 anchor 的原始值，确认输出格式
        if (numPredictions > 0) {
            Log.d(TAG, "Debug anchor[0]: cx=" + getOutputValue(0, 0) +
                    ", cy=" + getOutputValue(1, 0) +
                    ", w=" + getOutputValue(2, 0) +
                    ", h=" + getOutputValue(3, 0) +
                    ", personConf=" + getOutputValue(4, 0));
        }

        // 调试：扫描所有 anchor 所有类别，找出全局最大类别分数
        {
            float globalMax = 0;
            int globalMaxClass = -1;
            int globalMaxAnchor = -1;
            for (int a = 0; a < numAnchors; a++) {
                for (int c = 4; c < numChannels; c++) {
                    float conf = getOutputValue(c, a);
                    if (conf > globalMax) {
                        globalMax = conf;
                        globalMaxClass = c - 4;
                        globalMaxAnchor = a;
                    }
                }
            }
            if (globalMaxAnchor >= 0) {
                Log.d(TAG, "Global max: conf=" + globalMax + ", class=" + globalMaxClass +
                        ", box(cx=" + getOutputValue(0, globalMaxAnchor) +
                        ", cy=" + getOutputValue(1, globalMaxAnchor) +
                        ", w=" + getOutputValue(2, globalMaxAnchor) +
                        ", h=" + getOutputValue(3, globalMaxAnchor) + ")");
            }
        }

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
            candidateConfs.add(boxConf);

            if (score > maxScore) {
                maxScore = score;
                bestConf = boxConf;
                bestArea = area;
                bestPose = new PersonPose();
                bestPose.boxConfidence = boxConf;
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
                bestPose.boxConfidence = candidateConfs.get(bestMatchIdx);
                bestPose.box = candidateBoxes.get(bestMatchIdx);
                trackedBox = candidateBoxes.get(bestMatchIdx).clone();
                trackLostFrames = 0;
                Log.d(TAG, "Track matched: IoU=" + bestIou + ", conf=" + bestPose.boxConfidence);
            } else {
                trackLostFrames++;
                Log.d(TAG, "Track lost: " + trackLostFrames + "/" + MAX_TRACK_LOST);
                if (trackLostFrames >= MAX_TRACK_LOST) {
                    trackedBox = null;
                    trackLostFrames = 0;
                    Log.d(TAG, "Track reset");
                } else {
                    bestPose = null;
                }
            }
        } else if (bestPose != null) {
            trackedBox = bestPose.box.clone();
            trackLostFrames = 0;
            Log.d(TAG, "New track acquired, conf=" + bestConf);
        }

        if (bestPose != null) {
            Log.d(TAG, "Candidates: " + numCandidates + ", best conf=" + bestConf +
                    ", area=" + bestArea + ", score=" + maxScore);
            Log.d(TAG, "Raw box: [" + bestPose.box[0] + ", " + bestPose.box[1] +
                    ", " + bestPose.box[2] + ", " + bestPose.box[3] + "]");
        } else {
            Log.d(TAG, "No detection");
            if (trackedBox != null) {
                trackLostFrames++;
                if (trackLostFrames >= MAX_TRACK_LOST) {
                    trackedBox = null;
                    trackLostFrames = 0;
                }
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

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}
