package com.example.fplus;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import android.view.View;

public class OverlayView extends View {

    private static final String TAG = "OverlayView";
    private Paint boxPaint;
    private Paint roiPaint;

    // 当前显示框（归一化：cx, cy, w, h），位置与尺寸做低通平滑，无速度预测
    private float[] displayBox;
    // 淡出绘制框：目标丢失后 50ms 内继续绘制的旧框，与跟踪状态 displayBox 解耦。
    // 关键：目标丢失时立即清空 displayBox（跟踪状态），B 到来时走全新落位，
    // 避免淡出窗口内 B 被旧目标 A 的位置/速度缓存污染（导致 A/B 横跳）。
    private float[] fadingBox;
    // 1€ 滤波器（x/y 独立）
    private OneEuroFilter filterX;
    private OneEuroFilter filterY;

    // 1€ 滤波器参数：最小截止频率（Hz，静止平滑度）与速度系数（快速移动响应度）
    // 可调参数（从 SharedPreferences 读取，详见 AdvancedOptionsActivity）
    private double MIN_CUTOFF = 2.0;
    private double BETA = 8.0;
    // 跳变阈值：位移超过该值视为目标切换/误检，限制最大移动量，避免框瞬移
    private float MAX_JUMP = 0.4f;
    // 尺寸平滑系数：越小越平滑（0=完全锁死尺寸，1=不平滑）
    private float SIZE_SMOOTH = 0.45f;
    // 二阶 ease（慢快慢）：一阶 1€ 输出后再串一层 EMA，二阶低通阶跃响应为 S 形
    // （慢起→快中→慢收 = ease-in-out），比纯 1€ 的"快起慢收"更丝滑
    // EASE_KEEP 为二阶保留比例：0=关闭（纯一阶行为），越大越平滑但延迟增加
    private float EASE_KEEP = 0.4f;

    // 检测丢失后的淡出透明度（255=不透明，0=消失）
    private int fadeAlpha = 255;
    // 丢失后淡出时长（纳秒）：固定 100ms 内完全消失，避免绿框原地滞留
    private static final long FADE_DURATION_NANOS = 50_000_000L;
    // 首次丢失帧的时间戳，用于按真实时长而非帧数淡出
    private long lostAtNanos = 0;

    private int captureWidth = 0;
    private int captureHeight = 0;
    private int roiX = 0;
    private int roiY = 0;
    private int roiSize = 0;

    public OverlayView(Context context) {
        super(context);
        loadPrefs(context);
        initPaints();
        setWillNotDraw(false);
    }

    /**
     * 从 SharedPreferences 读取高级参数，无值则保留默认
     */
    private void loadPrefs(Context context) {
        SharedPreferences prefs = context.getSharedPreferences("fplus_settings", Context.MODE_PRIVATE);
        // 每个参数单独 try-catch：避免一个坏值连累其他参数丢失配置
        try {
            MIN_CUTOFF = Double.parseDouble(prefs.getString("overlay_min_cutoff", "2.0"));
        } catch (NumberFormatException e) {
            Log.w(TAG, "overlay_min_cutoff 解析失败，使用默认 2.0");
            MIN_CUTOFF = 2.0;
        }
        try {
            BETA = Double.parseDouble(prefs.getString("overlay_beta", "8.0"));
        } catch (NumberFormatException e) {
            Log.w(TAG, "overlay_beta 解析失败，使用默认 8.0");
            BETA = 8.0;
        }
        try {
            MAX_JUMP = Float.parseFloat(prefs.getString("overlay_max_jump", "0.4"));
        } catch (NumberFormatException e) {
            Log.w(TAG, "overlay_max_jump 解析失败，使用默认 0.4");
            MAX_JUMP = 0.4f;
        }
        try {
            SIZE_SMOOTH = Float.parseFloat(prefs.getString("overlay_size_smooth", "0.45"));
        } catch (NumberFormatException e) {
            Log.w(TAG, "overlay_size_smooth 解析失败，使用默认 0.45");
            SIZE_SMOOTH = 0.45f;
        }
        try {
            EASE_KEEP = Float.parseFloat(prefs.getString("overlay_ease", "0.4"));
        } catch (NumberFormatException e) {
            Log.w(TAG, "overlay_ease 解析失败，使用默认 0.4");
            EASE_KEEP = 0.4f;
        }
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
            // 目标丢失：先把当前跟踪框转存为淡出框，再立即清空跟踪状态
            if (displayBox != null) {
                if (fadingBox == null) fadingBox = new float[4];
                fadingBox[0] = displayBox[0];
                fadingBox[1] = displayBox[1];
                fadingBox[2] = displayBox[2];
                fadingBox[3] = displayBox[3];
                displayBox = null;   // 关键：立即清空跟踪状态，B 到来时走全新落位，无缓存残留
            }
            // 目标消失：清除 1€ 滤波器缓存（xHat/dxHat/xPrev），
            // 否则下一个目标（B）到来时 filter 会携带旧目标（A）的位置/速度惯性，
            // 绿框从 A 平滑滑到 B（表现为横跳/滑动）。reset 后 B 首次滤波直接落到 B 位置。
            if (filterX != null) filterX.reset();
            if (filterY != null) filterY.reset();

            // 淡出计时（基于 fadingBox，与跟踪状态无关）
            if (fadingBox != null) {
                if (lostAtNanos == 0) {
                    lostAtNanos = timestampNanos;
                }
                long elapsed = timestampNanos - lostAtNanos;
                if (elapsed <= 0) {
                    if (fadeAlpha > 255) fadeAlpha = 255;
                    if (fadeAlpha < 0) fadeAlpha = 0;
                } else if (elapsed >= FADE_DURATION_NANOS) {
                    fadingBox = null;
                    fadeAlpha = 255;
                    lostAtNanos = 0;
                } else {
                    int alpha = 255 - (int) (255L * elapsed / FADE_DURATION_NANOS);
                    if (alpha < 0) alpha = 0;
                    if (alpha > 255) alpha = 255;
                    fadeAlpha = alpha;
                }
            } else {
                lostAtNanos = 0;
            }
            // 无论绿框是否还在淡出，都要重绘：黄圈（ROI）回正需要持续刷新
            invalidate();
            return;
        }

        // 有目标：清除淡出框
        fadingBox = null;
        fadeAlpha = 255;
        lostAtNanos = 0;

        if (displayBox == null) {
            displayBox = new float[4];
            displayBox[0] = box[0];
            displayBox[1] = box[1];
            displayBox[2] = box[2];
            displayBox[3] = box[3];
            filterX = new OneEuroFilter(MIN_CUTOFF, BETA, 6.0);
            filterY = new OneEuroFilter(MIN_CUTOFF, BETA, 6.0);
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

            // 一阶 1€ 滤波器平滑中心（自适应：静止强平滑滤抖、快速移动弱平滑跟手）
            float oneEuroX = (float) filterX.filter(targetCx, timestampNanos);
            float oneEuroY = (float) filterY.filter(targetCy, timestampNanos);
            // 二阶 ease（慢快慢）：一阶输出后再串 EMA，阶跃响应呈 S 形 ease-in-out
            // displayBox 朝 oneEuro 推进 (1-EASE_KEEP)，保留 EASE_KEEP → 慢起快中慢收
            displayBox[0] = oneEuroX + EASE_KEEP * (displayBox[0] - oneEuroX);
            displayBox[1] = oneEuroY + EASE_KEEP * (displayBox[1] - oneEuroY);

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

        // 优先绘制跟踪框（displayBox），无跟踪框时绘制淡出框（fadingBox）
        float[] drawSrc = (displayBox != null) ? displayBox : fadingBox;
        if (drawSrc != null) {
            float cx = drawSrc[0];
            float cy = drawSrc[1];
            float w = drawSrc[2];
            float h = drawSrc[3];
            float left = offsetX + (cx - w / 2) * drawWidth;
            float top = offsetY + (cy - h / 2) * drawHeight;
            float right = offsetX + (cx + w / 2) * drawWidth;
            float bottom = offsetY + (cy + h / 2) * drawHeight;
            boxPaint.setAlpha(fadeAlpha);
            canvas.drawRect(left, top, right, bottom, boxPaint);
        }
    }
}
