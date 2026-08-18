package com.example.fplus;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import android.view.View;

public class OverlayView extends View {

    private static final String TAG = "OverlayView";

    private Paint boxPaint;
    private Paint debugPaint;

    private float[] smoothedBox;
    private float boxSmoothingFactor = 0.25f;

    private int missingFrameCount = 0;
    private static final int MAX_MISSING_FRAMES = 8;

    private int captureWidth = 0;
    private int captureHeight = 0;

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

        debugPaint = new Paint();
        debugPaint.setColor(Color.RED);
        debugPaint.setStyle(Paint.Style.STROKE);
        debugPaint.setStrokeWidth(3f);
        debugPaint.setAntiAlias(true);
    }

    public void updatePose(PoseEstimator.PersonPose pose, int captureWidth, int captureHeight) {
        this.captureWidth = captureWidth;
        this.captureHeight = captureHeight;

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
            for (int i = 0; i < 4; i++) {
                smoothedBox[i] = smoothedBox[i] + boxSmoothingFactor * (pose.box[i] - smoothedBox[i]);
            }
        }

        Log.d(TAG, "Box normalized: cx=" + pose.box[0] + ", cy=" + pose.box[1] +
                ", w=" + pose.box[2] + ", h=" + pose.box[3] +
                ", conf=" + pose.boxConfidence);

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

        // 调试：画 letterbox 区域边框（蓝色）
        debugPaint.setColor(Color.BLUE);
        canvas.drawRect(offsetX, offsetY, offsetX + drawWidth, offsetY + drawHeight, debugPaint);

        // 调试：画归一化中心 (0.5, 0.5) 对应的十字
        debugPaint.setColor(Color.RED);
        float centerX = offsetX + 0.5f * drawWidth;
        float centerY = offsetY + 0.5f * drawHeight;
        canvas.drawLine(centerX - 50, centerY, centerX + 50, centerY, debugPaint);
        canvas.drawLine(centerX, centerY - 50, centerX, centerY + 50, debugPaint);

        // 调试：画屏幕中心十字
        debugPaint.setColor(Color.CYAN);
        float screenCx = viewWidth / 2f;
        float screenCy = viewHeight / 2f;
        canvas.drawLine(screenCx - 30, screenCy, screenCx + 30, screenCy, debugPaint);
        canvas.drawLine(screenCx, screenCy - 30, screenCx, screenCy + 30, debugPaint);

        // 绘制检测框
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

            Log.d(TAG, "Draw box: left=" + left + ", top=" + top +
                    ", right=" + right + ", bottom=" + bottom +
                    ", view=" + viewWidth + "x" + viewHeight +
                    ", drawArea=" + drawWidth + "x" + drawHeight +
                    ", offset=(" + offsetX + "," + offsetY + ")");
        }
    }
}
