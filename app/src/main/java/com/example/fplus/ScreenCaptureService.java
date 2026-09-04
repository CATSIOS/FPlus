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
import android.os.PowerManager;
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
    private PowerManager.WakeLock cpuWakeLock;  // PARTIAL_WAKE_LOCK：保持 CPU 不进入 idle 降压
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
    private byte[] captureByteArray;  // bulk get 缓冲，跳过逐字节 buffer.get 的 JNI 开销

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
        // minSdk=29 ≥ Q，直接带 FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION 启动
        startForeground(NOTIFICATION_ID, buildNotification(),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);

        SharedPreferences prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);
        String backendName = prefs.getString("backend", PoseEstimator.Backend.GPU.name());
        PoseEstimator.Backend backend;
        try {
            backend = PoseEstimator.Backend.valueOf(backendName);
        } catch (IllegalArgumentException e) {
            backend = PoseEstimator.Backend.CPU;
        }
        String modelName = prefs.getString("model_name", ModelManager.DEFAULT_MODEL);
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

        // PARTIAL_WAKE_LOCK：只保持 CPU 运行（屏幕关闭也没用，因为我们要触控/屏显）
        // 作用：避免系统调度器在用户短暂不触控时进入轻度 idle 并立即降频降压
        // ON_AFTER_RELEASE：释放时恢复用户手动点亮后的亮度调节逻辑；不计数（不 setCounted）
        // 安全策略：拿不到时静默失败，不影响核心功能
        try {
            PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
            if (pm != null) {
                cpuWakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK
                        | PowerManager.ON_AFTER_RELEASE, "FPlus:cpu_lock");
                cpuWakeLock.setReferenceCounted(false);
                cpuWakeLock.acquire(10 * 60 * 1000L /* 10 min 超时兜底，防止泄漏 */);
                Log.d(TAG, "PARTIAL_WAKE_LOCK acquired (timeout=10min)");
            }
        } catch (Throwable t) {
            Log.w(TAG, "WakeLock acquire failed, degrade: " + t.getMessage());
            cpuWakeLock = null;
        }

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

        // CaptureThread（像素拷贝+Canvas缩放）是 pre/convert 耗时主体，从默认(0)升到 DISPLAY(-4)，
        // 温控调度器优先降级低优先级线程，提升后能降低被降频概率，同时不抢 URGENT_DISPLAY(-10) 的推理线程
        captureThread = new HandlerThread("CaptureThread", Process.THREAD_PRIORITY_DISPLAY);
        captureThread.start();
        captureHandler = new Handler(captureThread.getLooper());
        imageReader.setOnImageAvailableListener(reader -> {
            // 停止中或 reader 已关闭：直接返回，避免在已 close 的 reader 上 acquire 抛 IllegalStateException
            if (!isCapturing) return;
            Image image;
            try {
                image = reader.acquireLatestImage();
            } catch (IllegalStateException e) {
                return;
            }
            if (image == null) return;
            try {
                Bitmap frame = imageToBitmap(image);
                final PoseEstimator estimator = poseEstimator;
                if (frame != null && estimator != null) {
                    // 推理忙则丢帧（保持低延迟，不积压旧帧）
                    if (inferenceBusy.compareAndSet(false, true)) {
                        final Bitmap fFrame = frame;
                        inferenceHandler.post(() -> {
                            try {
                                processFrame(fFrame);
                            } finally {
                                inferenceBusy.set(false);
                            }
                        });
                    }
                }
            } finally {
                try { image.close(); } catch (IllegalStateException ignored) {}
            }
        }, captureHandler);

        isCapturing = true;
        Log.d(TAG, "Screen capture started: " + captureWidth + "x" + captureHeight);
    }

    private void processFrame(Bitmap frame) {
        // 停止中：直接返回，避免 onDestroy 并发 null 字段时 NPE
        if (!isCapturing) return;
        // 局部引用：onDestroy 可能在 estimate 期间并发 null 化字段
        final PoseEstimator estimator = poseEstimator;
        final OverlayView view = overlayView;
        if (estimator == null) return;

        long t0 = System.nanoTime();
        PoseEstimator.PersonPose pose;
        try {
            pose = estimator.estimate(frame);
        } catch (Exception e) {
            // estimator 可能正在被 close，吞掉异常退出
            Log.w(TAG, "estimate interrupted during teardown", e);
            return;
        }
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

        if (view != null) {
            int roiX = estimator.getRoiX();
            int roiY = estimator.getRoiY();
            int roiSize = estimator.getRoiSize();
            // pose 为 null 也要调用，让 OverlayView 的丢失/衰减逻辑生效
            float[] box = (pose != null && pose.box != null) ? pose.box.clone() : null;
            int w = frameWidth;
            int h = frameHeight;
            final float[] fBox = box;
            final long fTime = detectTime;
            final int fRoiX = roiX, fRoiY = roiY, fRoiSize = roiSize;
            final OverlayView fView = view;
            // 局部引用 fView 不会被并发 null 化，lambda 安全
            mainHandler.post(() -> {
                if (!isCapturing) return;
                try {
                    fView.updateDetection(fBox, w, h, fRoiX, fRoiY, fRoiSize, fTime);
                } catch (Exception ignored) {
                    // view 已被 removeView，忽略
                }
            });
        }
    }

    private int getScreenRotation() {
        Display display = windowManager.getDefaultDisplay();
        return display.getRotation();
    }

    private Bitmap imageToBitmap(Image image) {
        // 兜底：imageReader.close 后 image 的 DirectByteBuffer 可能变 inaccessible，
        // bulk get 会抛 IllegalStateException；此时返回 null 跳过该帧，避免崩溃
        try {
            Image.Plane plane = image.getPlanes()[0];
            ByteBuffer buffer = plane.getBuffer();
            int pixelStride = plane.getPixelStride();
            int rowStride = plane.getRowStride();

            buffer.rewind();
            int totalBytes = rowStride * captureHeight;
            if (captureByteArray == null || captureByteArray.length < totalBytes) {
                captureByteArray = new byte[totalBytes];
            }
            // 一次 bulk get 拷整个 buffer 到 byte[]，跳过逐字节 buffer.get() 的 JNI 开销
            buffer.get(captureByteArray, 0, totalBytes);

            int len = captureWidth * captureHeight;
            if (capturePixels == null || capturePixels.length != len) {
                capturePixels = new int[len];
            }
            int[] pixels = capturePixels;
            int outIndex = 0;

            // 遍历 byte[]（内存无 JNI），按 rowStride 步长提取 RGBA 打包成 ARGB int
            if (pixelStride == 4) {
                for (int y = 0; y < captureHeight; y++) {
                    int rowStart = y * rowStride;
                    for (int x = 0; x < captureWidth; x++) {
                        int idx = rowStart + x * 4;
                        int r = captureByteArray[idx] & 0xFF;
                        int g = captureByteArray[idx + 1] & 0xFF;
                        int b = captureByteArray[idx + 2] & 0xFF;
                        pixels[outIndex++] = 0xFF000000 | (r << 16) | (g << 8) | b;
                    }
                }
            } else {
                // pixelStride != 4 兜底（罕见，保留兼容）
                for (int y = 0; y < captureHeight; y++) {
                    int rowStart = y * rowStride;
                    for (int x = 0; x < captureWidth; x++) {
                        int index = rowStart + x * pixelStride;
                        int r = captureByteArray[index] & 0xFF;
                        int g = captureByteArray[index + 1] & 0xFF;
                        int b = captureByteArray[index + 2] & 0xFF;
                        pixels[outIndex++] = 0xFF000000 | (r << 16) | (g << 8) | b;
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
        } catch (Exception e) {
            // buffer 不可访问（imageReader 关闭）或帧数据不完整（竞态），跳过该帧
            return null;
        }
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
        // minSdk=29 ≥ O，NotificationChannel 必填
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Screen Capture Service",
                NotificationManager.IMPORTANCE_LOW);
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(channel);
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
        // 1. 先置停止标志，让 listener / processFrame 快速短路退出
        isCapturing = false;

        // 2. 先注销 projectionCallback，避免 mediaProjection.stop() 触发 stopSelf 重入
        if (mediaProjection != null && projectionCallback != null) {
            try {
                mediaProjection.unregisterCallback(projectionCallback);
            } catch (Exception ignored) {}
            projectionCallback = null;
        }

        // 3. 清空 imageReader 监听器，确保 captureThread 上不再有「新」回调入队
        //    （已在队列中或正在执行的不受影响，需要靠下面 join 等待其完成）
        if (imageReader != null) {
            try {
                imageReader.setOnImageAvailableListener(null, null);
            } catch (Exception ignored) {}
        }

        // 4. 先退出 captureThread 并 join 等待当前正在执行的 imageToBitmap 完成，
        //    必须在 imageReader.close() 之前——否则 image 的 DirectByteBuffer 会被释放，
        //    导致正在跑的 imageToBitmap 内 buffer.get() 抛 IllegalStateException: buffer is inaccessible
        if (captureThread != null) {
            captureThread.quitSafely();
            try { captureThread.join(500); } catch (InterruptedException ignored) {}
            captureThread = null;
            captureHandler = null;
        }

        // 5. captureThread 已停止，此时安全释放 VirtualDisplay / ImageReader / MediaProjection
        if (virtualDisplay != null) {
            try { virtualDisplay.release(); } catch (Exception ignored) {}
            virtualDisplay = null;
        }
        if (imageReader != null) {
            try { imageReader.close(); } catch (Exception ignored) {}
            imageReader = null;
        }
        if (mediaProjection != null) {
            try { mediaProjection.stop(); } catch (Exception ignored) {}
            mediaProjection = null;
        }

        // 6. 捕获需要释放的对象引用并立即清空字段引用，
        //    这样若 onDestroy 期间 onStartCommand 重入，新启动创建的对象不会被旧清理误释放
        final PoseEstimator estimatorToClose = poseEstimator;
        final Bitmap bmp0 = captureBitmaps[0];
        final Bitmap bmp1 = captureBitmaps[1];
        poseEstimator = null;
        captureBitmaps[0] = null;
        captureBitmaps[1] = null;

        // 7. 把 estimator.close() / bitmap.recycle() 排到 inferenceThread 队列末尾执行，
        //    quitSafely 会让当前正在跑的 processFrame 完成后顺序执行清理任务再退出，
        //    既保证单线程安全释放，又不阻塞主线程（避免 join 引发 ANR 与超时竞态崩溃）
        if (inferenceThread != null) {
            final Handler ih = inferenceHandler;
            boolean posted = false;
            if (ih != null) {
                try {
                    ih.post(() -> {
                        if (estimatorToClose != null) {
                            try { estimatorToClose.close(); } catch (Exception ignored) {}
                        }
                        if (bmp0 != null) try { bmp0.recycle(); } catch (Exception ignored) {}
                        if (bmp1 != null) try { bmp1.recycle(); } catch (Exception ignored) {}
                    });
                    posted = true;
                } catch (IllegalStateException ignored) {
                    // looper 已退出，回退到同步清理
                }
            }
            inferenceThread.quitSafely();
            inferenceThread = null;
            inferenceHandler = null;
            if (!posted) {
                // 无 looper 可用，同步清理
                if (estimatorToClose != null) {
                    try { estimatorToClose.close(); } catch (Exception ignored) {}
                }
                if (bmp0 != null) try { bmp0.recycle(); } catch (Exception ignored) {}
                if (bmp1 != null) try { bmp1.recycle(); } catch (Exception ignored) {}
            }
        } else {
            // 无 inferenceThread，直接同步清理
            if (estimatorToClose != null) {
                try { estimatorToClose.close(); } catch (Exception ignored) {}
            }
            if (bmp0 != null) try { bmp0.recycle(); } catch (Exception ignored) {}
            if (bmp1 != null) try { bmp1.recycle(); } catch (Exception ignored) {}
        }

        removeOverlayView();

        // 最后释放 WakeLock（即使没 acquire 成功也安全，null/Held 都会短路）
        if (cpuWakeLock != null) {
            try {
                if (cpuWakeLock.isHeld()) cpuWakeLock.release();
            } catch (Throwable ignored) { /* 重复 release 或系统已回收 */ }
            cpuWakeLock = null;
        }
        instance = null;
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}