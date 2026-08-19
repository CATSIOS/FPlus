package com.example.fplus;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

public class OverlayView extends View {

    private Paint boxPaint;
    private Paint roiPaint;

    private float[] smoothedBox;
    // 中心坐标用高响应，快速跟上人物；尺寸用低响应，避免框大小抖动
    private float centerSmoothingFactor = 0.7f;
    private float sizeSmoothingFactor = 0.4f;
    // 中心位移超过该阈值（归一化）时直接跳到位，消除大幅移动/换目标的滞后
    private static final float JUMP_DISTANCE = 0.12f;

    private int missingFrameCount = 0;
    private static final int MAX_MISSING_FRAMES = 8;

    private int captureWidth = 0;
    private int captureHeight = 0;

    // 中心裁剪范围（相对 capture 帧的像素坐标）
    private int roiX = 0;
    private int roiY = 0;
    private int roiSize = 0;

    public OverlayView(Context context) {
        super(context);
        initPaints();
        setWillNotDraw(false);
    }

    private void initPaints() {
        boxPaint = new Paint();
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(8f);
        boxPaint.setAntiAlias(true);

        roiPaint = new Paint();
        roiPaint.setColor(Color.YELLOW);
        roiPaint.setStyle(Paint.Style.STROKE);
        roiPaint.setStrokeWidth(3f);
        roiPaint.setAntiAlias(true);
    }

    public void updatePose(PoseEstimator.PersonPose pose, int captureWidth, int captureHeight,
                           int roiX, int roiY, int roiSize) {
        this.captureWidth = captureWidth;
        this.captureHeight = captureHeight;
        this.roiX = roiX;
        this.roiY = roiY;
        this.roiSize = roiSize;

        if (pose == null || pose.box == null) {
            missingFrameCount++;
            if (missingFrameCount >= MAX_MISSING_FRAMES) {
                smoothedBox = null;
                invalidate();
            }
            return;
        }

        missingFrameCount = 0;

        if (smoothedBox == null) {
            smoothedBox = new float[4];
            System.arraycopy(pose.box, 0, smoothedBox, 0, 4);
        } else {
            float dcx = pose.box[0] - smoothedBox[0];
            float dcy = pose.box[1] - smoothedBox[1];
            float dist = (float) Math.hypot(dcx, dcy);

            if (dist > JUMP_DISTANCE) {
                // 大幅移动或换目标：直接跳到位
                smoothedBox[0] = pose.box[0];
                smoothedBox[1] = pose.box[1];
            } else {
                smoothedBox[0] += centerSmoothingFactor * dcx;
                smoothedBox[1] += centerSmoothingFactor * dcy;
            }

            smoothedBox[2] += sizeSmoothingFactor * (pose.box[2] - smoothedBox[2]);
            smoothedBox[3] += sizeSmoothingFactor * (pose.box[3] - smoothedBox[3]);
        }

        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int viewWidth = getWidth();
        int viewHeight = getHeight();
        if (viewWidth == 0 || viewHeight == 0) return;
        if (captureWidth == 0 || captureHeight == 0) return;

        float scale = Math.min((float) viewWidth / captureWidth, (float) viewHeight / captureHeight);
        int drawWidth = (int) (captureWidth * scale);
        int drawHeight = (int) (captureHeight * scale);
        int offsetX = (viewWidth - drawWidth) / 2;
        int offsetY = (viewHeight - drawHeight) / 2;

        // 画中心裁剪范围（黄色框）
        if (roiSize > 0) {
            float roiLeft = offsetX + (roiX / (float) captureWidth) * drawWidth;
            float roiTop = offsetY + (roiY / (float) captureHeight) * drawHeight;
            float roiRight = offsetX + ((roiX + roiSize) / (float) captureWidth) * drawWidth;
            float roiBottom = offsetY + ((roiY + roiSize) / (float) captureHeight) * drawHeight;
            canvas.drawRect(roiLeft, roiTop, roiRight, roiBottom, roiPaint);
        }

        if (smoothedBox != null) {
            float cx = smoothedBox[0];
            float cy = smoothedBox[1];
            float w = smoothedBox[2];
            float h = smoothedBox[3];
            float left = offsetX + (cx - w / 2) * drawWidth;
            float top = offsetY + (cy - h / 2) * drawHeight;
            float right = offsetX + (cx + w / 2) * drawWidth;
            float bottom = offsetY + (cy + h / 2) * drawHeight;
            canvas.drawRect(left, top, right, bottom, boxPaint);
        }
    }
}
