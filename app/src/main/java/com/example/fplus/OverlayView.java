package com.example.fplus;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

public class OverlayView extends View {

    private Paint boxPaint;
    private Paint roiPaint;

    // 当前显示框（归一化：cx, cy, w, h），位置与尺寸做低通平滑，无速度预测
    private float[] displayBox;
    // 1€ 滤波器（x/y 独立）
    private OneEuroFilter filterX;
    private OneEuroFilter filterY;

    // 1€ 滤波器参数：最小截止频率（Hz，静止平滑度）与速度系数（快速移动响应度）
    private static final double MIN_CUTOFF = 1.5;
    private static final double BETA = 5.0;
    // 跳变阈值：位移超过该值视为目标切换/误检，限制最大移动量，避免框瞬移
    private static final float MAX_JUMP = 0.2f;
    // 尺寸平滑系数
    private static final float SIZE_SMOOTH = 0.6f;

    // 检测丢失后的淡出透明度（255=不透明，0=消失）
    private int fadeAlpha = 255;
    // 丢失后淡出时长（纳秒）：固定 100ms 内完全消失，避免绿框原地滞留
    private static final long FADE_DURATION_NANOS = 100_000_000L;
    // 首次丢失帧的时间戳，用于按真实时长而非帧数淡出
    private long lostAtNanos = 0;

    private int captureWidth = 0;
    private int captureHeight = 0;
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

    /**
     * 检测帧调用：用检测框做低通平滑。
     *
     * @param box 归一化 [cx, cy, w, h]，null 表示本帧未检测到目标
     */
    public void updateDetection(float[] box, int captureWidth, int captureHeight,
                                int roiX, int roiY, int roiSize, long timestampNanos) {
        this.captureWidth = captureWidth;
        this.captureHeight = captureHeight;
        this.roiX = roiX;
        this.roiY = roiY;
        this.roiSize = roiSize;

        if (box == null) {
            if (displayBox != null) {
                if (lostAtNanos == 0) {
                    lostAtNanos = timestampNanos;
                }
                long elapsed = timestampNanos - lostAtNanos;
                if (elapsed >= FADE_DURATION_NANOS) {
                    displayBox = null;
                    fadeAlpha = 255;
                    lostAtNanos = 0;
                } else {
                    // 按真实时长线性淡出：从 255 到 0
                    fadeAlpha = 255 - (int) (255L * elapsed / FADE_DURATION_NANOS);
                }
            } else {
                lostAtNanos = 0;
            }
            // 无论绿框是否还在淡出，都要重绘：黄圈（ROI）回正需要持续刷新
            invalidate();
            return;
        }

        fadeAlpha = 255;
        lostAtNanos = 0;

        if (displayBox == null) {
            displayBox = new float[4];
            displayBox[0] = box[0];
            displayBox[1] = box[1];
            displayBox[2] = box[2];
            displayBox[3] = box[3];
            filterX = new OneEuroFilter(MIN_CUTOFF, BETA);
            filterY = new OneEuroFilter(MIN_CUTOFF, BETA);
            filterX.filter(box[0], timestampNanos);
            filterY.filter(box[1], timestampNanos);
        } else {
            // 跳变抑制：限制目标位置的最大移动量，防止切换/误检导致框瞬移
            float dcx = box[0] - displayBox[0];
            float dcy = box[1] - displayBox[1];
            float dist = (float) Math.hypot(dcx, dcy);
            float targetCx = box[0];
            float targetCy = box[1];
            if (dist > MAX_JUMP) {
                float scale = MAX_JUMP / dist;
                targetCx = displayBox[0] + dcx * scale;
                targetCy = displayBox[1] + dcy * scale;
            }

            // 1€ 滤波器平滑中心（自适应：静止强平滑滤抖、快速移动弱平滑跟手）
            displayBox[0] = (float) filterX.filter(targetCx, timestampNanos);
            displayBox[1] = (float) filterY.filter(targetCy, timestampNanos);

            // 尺寸平滑
            displayBox[2] += SIZE_SMOOTH * (box[2] - displayBox[2]);
            displayBox[3] += SIZE_SMOOTH * (box[3] - displayBox[3]);
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

        // 画中心裁剪范围（黄色，圆形仅作视觉提示，实际 ROI 仍为方形）
        if (roiSize > 0) {
            float roiCx = offsetX + ((roiX + roiSize / 2f) / (float) captureWidth) * drawWidth;
            float roiCy = offsetY + ((roiY + roiSize / 2f) / (float) captureHeight) * drawHeight;
            float roiRadius = (roiSize / 2f) / (float) captureWidth * drawWidth;
            canvas.drawCircle(roiCx, roiCy, roiRadius, roiPaint);
        }

        if (displayBox != null) {
            float cx = displayBox[0];
            float cy = displayBox[1];
            float w = displayBox[2];
            float h = displayBox[3];
            float left = offsetX + (cx - w / 2) * drawWidth;
            float top = offsetY + (cy - h / 2) * drawHeight;
            float right = offsetX + (cx + w / 2) * drawWidth;
            float bottom = offsetY + (cy + h / 2) * drawHeight;
            boxPaint.setAlpha(fadeAlpha);
            canvas.drawRect(left, top, right, bottom, boxPaint);
        }
    }
}
