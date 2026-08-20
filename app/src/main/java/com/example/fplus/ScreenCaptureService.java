package com.example.fplus;

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.view.Display;
import android.view.Surface;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.os.Process;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.WindowManager;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

public class ScreenCaptureService extends Service {

    private static final String TAG = "ScreenCaptureService";
    private static final String CHANNEL_ID = "screen_capture_channel";
    private static final int NOTIFICATION_ID = 1;
    private static final int VIRTUAL_DISPLAY_FLAGS = DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR;

    private static ScreenCaptureService instance;

    private MediaProjectionManager mediaProjectionManager;
    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;

    private HandlerThread captureThread;
    private Handler captureHandler;
    private HandlerThread inferenceThread;
    private Handler inferenceHandler;
    private final AtomicBoolean inferenceBusy = new AtomicBoolean(false);
    private Handler mainHandler;

    private PoseEstimator poseEstimator;
    private OverlayView overlayView;
    private WindowManager windowManager;
    private boolean isCapturing = false;

    private MediaProjection.Callback projectionCallback;

    // 动态捕获分辨率
    private int captureWidth;
    private int captureHeight;

    // 双缓冲 Bitmap：Capture 线程和 Inference 线程交替使用，避免读写冲突
    private final Bitmap[] captureBitmaps = new Bitmap[2];
    private int captureBitmapIdx = 0;
    // 复用像素缓冲，避免每帧分配
    private int[] capturePixels;

    // 帧耗时统计
    private long estimateTimeAccum;
    private int frameCount;

    public static ScreenCaptureService getInstance() {
        return instance;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
        mainHandler = new Handler(Looper.getMainLooper());
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        mediaProjectionManager = (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        int resultCode = Activity.RESULT_CANCELED;
        Intent resultData = null;
        if (intent != null) {
            resultCode = intent.getIntExtra("resultCode", Activity.RESULT_CANCELED);
            resultData = intent.getParcelableExtra("resultData");
        }

        // 屏幕捕获授权已在 MainActivity 中获得，此时才能以 mediaProjection 类型启动前台服务
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, buildNotification(),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIFICATION_ID, buildNotification());
        }

        SharedPreferences prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);
        String backendName = prefs.getString("backend", PoseEstimator.Backend.GPU.name());
        PoseEstimator.Backend backend;
        try {
            backend = PoseEstimator.Backend.valueOf(backendName);
        } catch (IllegalArgumentException e) {
            backend = PoseEstimator.Backend.CPU;
        }
        String modelName = prefs.getString("model_name", "sunxds_0.8.0.tflite");
        try {
            poseEstimator = new PoseEstimator(this, modelName, backend);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load pose estimator", e);
            stopSelf();
            return START_NOT_STICKY;
        }

        addOverlayView();

        if (resultCode == Activity.RESULT_OK && resultData != null) {
            mediaProjection = mediaProjectionManager.getMediaProjection(resultCode, resultData);
            projectionCallback = new MediaProjection.Callback() {
                @Override
                public void onStop() {
                    stopSelf();
                }
            };
            mediaProjection.registerCallback(projectionCallback, mainHandler);
            startCapture();
        } else {
            Log.e(TAG, "No capture permission result");
            stopSelf();
        }
        return START_NOT_STICKY;
    }

    private void startCapture() {
        if (mediaProjection == null || isCapturing) return;

        // 获取屏幕实际尺寸（自然方向，不随旋转变）
        DisplayMetrics metrics = new DisplayMetrics();
        windowManager.getDefaultDisplay().getRealMetrics(metrics);
        int screenWidth = metrics.widthPixels;
        int screenHeight = metrics.heightPixels;

        // 根据宽高比是否与当前方向匹配来决定是否交换，
        // 兼容 getRealMetrics 返回「自然方向」或「当前方向」的不同设备
        int rotation = getScreenRotation();
        boolean isLandscape = (rotation == Surface.ROTATION_90 || rotation == Surface.ROTATION_270);
        boolean sizeIsLandscape = (screenWidth > screenHeight);
        if (isLandscape != sizeIsLandscape) {
            int tmp = screenWidth;
            screenWidth = screenHeight;
            screenHeight = tmp;
        }
        Log.d(TAG, "Screen real=" + metrics.widthPixels + "x" + metrics.heightPixels +
                ", rotation=" + rotation + ", capture size=" + screenWidth + "x" + screenHeight);

        // 设置捕获分辨率，保持比例，宽度最大 720（模型输入 640，720 足够且省性能）
        float scale = Math.min(720f / screenWidth, 1280f / screenHeight);
        captureWidth = Math.round(screenWidth * scale);
        captureHeight = Math.round(screenHeight * scale);

        imageReader = ImageReader.newInstance(captureWidth, captureHeight, PixelFormat.RGBA_8888, 3);
        virtualDisplay = mediaProjection.createVirtualDisplay(
                "ScreenCapture",
                captureWidth, captureHeight, metrics.densityDpi,
                VIRTUAL_DISPLAY_FLAGS,
                imageReader.getSurface(),
                null, null);

        // 双线程流水线：
        //   CaptureThread（默认优先级）：acquire + imageToBitmap（像素拷贝）
        //   InferenceThread（URGENT_DISPLAY）：estimate 推理 + post UI
        //   双 Bitmap 交替：连续两帧返回不同 Bitmap，Inference 读完上一帧后 Capture 才回写
        inferenceThread = new HandlerThread("InferenceThread",
                Process.THREAD_PRIORITY_URGENT_DISPLAY);
        inferenceThread.start();
        inferenceHandler = new Handler(inferenceThread.getLooper());

        captureThread = new HandlerThread("CaptureThread");
        captureThread.start();
        captureHandler = new Handler(captureThread.getLooper());
        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image != null) {
                Bitmap frame = imageToBitmap(image);
                image.close();
                if (frame != null && poseEstimator != null) {
                    // 推理忙则丢帧（保持低延迟，不积压旧帧）
                    if (inferenceBusy.compareAndSet(false, true)) {
                        final Bitmap fFrame = frame;
                        inferenceHandler.post(() -> {
                            processFrame(fFrame);
                            inferenceBusy.set(false);
                        });
                    }
                }
            }
        }, captureHandler);

        isCapturing = true;
        Log.d(TAG, "Screen capture started: " + captureWidth + "x" + captureHeight);
    }

    private void processFrame(Bitmap frame) {
        long t0 = System.nanoTime();
        PoseEstimator.PersonPose pose = poseEstimator.estimate(frame);
        long detectTime = System.nanoTime(); // 检测完成时间戳
        long elapsed = detectTime - t0;
        int frameWidth = frame.getWidth();
        int frameHeight = frame.getHeight();
        // frame 是复用的 captureBitmap，不 recycle

        // 每 30 帧统计一次平均耗时
        estimateTimeAccum += elapsed;
        frameCount++;
        if (frameCount >= 30) {
            long avgMs = estimateTimeAccum / frameCount / 1_000_000;
            Log.d(TAG, "avg estimate=" + avgMs + "ms (~" + (1000 / Math.max(1, avgMs)) + "fps)");
            frameCount = 0;
            estimateTimeAccum = 0;
        }

        if (overlayView != null) {
            int roiX = poseEstimator.getRoiX();
            int roiY = poseEstimator.getRoiY();
            int roiSize = poseEstimator.getRoiSize();
            // pose 为 null 也要调用，让 OverlayView 的丢失/衰减逻辑生效
            float[] box = (pose != null && pose.box != null) ? pose.box.clone() : null;
            int w = frameWidth;
            int h = frameHeight;
            final float[] fBox = box;
            final long fTime = detectTime;
            mainHandler.post(() -> overlayView.updateDetection(fBox, w, h, roiX, roiY, roiSize, fTime));
        }
    }

    private int getScreenRotation() {
        Display display = windowManager.getDefaultDisplay();
        return display.getRotation();
    }

    private Bitmap imageToBitmap(Image image) {
        Image.Plane plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int pixelStride = plane.getPixelStride();
        int rowStride = plane.getRowStride();
        int rowPadding = rowStride - pixelStride * captureWidth;

        buffer.rewind();
        int len = captureWidth * captureHeight;
        if (capturePixels == null || capturePixels.length != len) {
            capturePixels = new int[len];
        }
        int[] pixels = capturePixels;
        int outIndex = 0;

        if (pixelStride == 4) {
            for (int y = 0; y < captureHeight; y++) {
                for (int x = 0; x < captureWidth; x++) {
                    int r = buffer.get() & 0xFF;
                    int g = buffer.get() & 0xFF;
                    int b = buffer.get() & 0xFF;
                    int a = buffer.get() & 0xFF;
                    pixels[outIndex++] = (a << 24) | (r << 16) | (g << 8) | b;
                }
                if (rowPadding > 0) {
                    buffer.position(buffer.position() + rowPadding);
                }
            }
        } else {
            for (int y = 0; y < captureHeight; y++) {
                int rowStart = y * rowStride;
                for (int x = 0; x < captureWidth; x++) {
                    int index = rowStart + x * pixelStride;
                    int r = buffer.get(index) & 0xFF;
                    int g = buffer.get(index + 1) & 0xFF;
                    int b = buffer.get(index + 2) & 0xFF;
                    int a = buffer.get(index + 3) & 0xFF;
                    pixels[outIndex++] = (a << 24) | (r << 16) | (g << 8) | b;
                }
            }
        }

        // 双缓冲：交替返回 bitmap A/B，连续两帧返回不同实例
        // Inference 读上一帧的 3~5ms 远小于两帧间隔（~32ms），无读写冲突
        int idx = captureBitmapIdx;
        captureBitmapIdx = (idx + 1) % 2;
        Bitmap bmp = captureBitmaps[idx];
        if (bmp == null
                || bmp.getWidth() != captureWidth
                || bmp.getHeight() != captureHeight) {
            if (bmp != null) bmp.recycle();
            bmp = Bitmap.createBitmap(captureWidth, captureHeight, Bitmap.Config.ARGB_8888);
            captureBitmaps[idx] = bmp;
        }
        bmp.setPixels(pixels, 0, captureWidth, 0, 0, captureWidth, captureHeight);
        return bmp;
    }

    private void addOverlayView() {
        if (overlayView != null) return;
        overlayView = new OverlayView(this);
        WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE |
                        WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE |
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT);
        windowManager.addView(overlayView, params);
    }

    private void removeOverlayView() {
        if (overlayView != null) {
            windowManager.removeView(overlayView);
            overlayView = null;
        }
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "Screen Capture Service",
                    NotificationManager.IMPORTANCE_LOW);
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(channel);
        }
    }

    private Notification buildNotification() {
        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("FPlus 骨骼绑定")
                .setContentText("正在运行，捕获屏幕并分析姿态...")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setOngoing(true)
                .setAutoCancel(false)
                .build();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        isCapturing = false;
        if (mediaProjection != null && projectionCallback != null) {
            mediaProjection.unregisterCallback(projectionCallback);
            projectionCallback = null;
        }
        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
        if (mediaProjection != null) {
            mediaProjection.stop();
            mediaProjection = null;
        }
        if (captureThread != null) {
            captureThread.quitSafely();
            captureThread = null;
        }
        if (inferenceThread != null) {
            inferenceThread.quitSafely();
            inferenceThread = null;
        }
        if (poseEstimator != null) {
            poseEstimator.close();
            poseEstimator = null;
        }
        for (int i = 0; i < captureBitmaps.length; i++) {
            if (captureBitmaps[i] != null) {
                captureBitmaps[i].recycle();
                captureBitmaps[i] = null;
            }
        }
        removeOverlayView();
        instance = null;
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}