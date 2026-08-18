package com.example.fplus;

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.Matrix;
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
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

import java.nio.ByteBuffer;

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
    private Handler mainHandler;

    private PoseEstimator poseEstimator;
    private OverlayView overlayView;
    private WindowManager windowManager;
    private boolean isCapturing = false;

    private MediaProjection.Callback projectionCallback;

    // 动态捕获分辨率
    private int captureWidth;
    private int captureHeight;

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
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, buildNotification(),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIFICATION_ID, buildNotification());
        }

        try {
            poseEstimator = new PoseEstimator(this);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load pose estimator", e);
            stopSelf();
            return START_NOT_STICKY;
        }

        addOverlayView();

        Intent captureIntent = mediaProjectionManager.createScreenCaptureIntent();
        startActivityForResult(captureIntent, REQUEST_CODE_CAPTURE);
        return START_STICKY;
    }

    private static final int REQUEST_CODE_CAPTURE = 1001;

    private void startActivityForResult(Intent intent, int requestCode) {
        Intent activityIntent = new Intent(this, CapturePermissionActivity.class);
        activityIntent.putExtra("captureIntent", intent);
        activityIntent.putExtra("requestCode", requestCode);
        activityIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(activityIntent);
    }

    public void onCapturePermissionResult(boolean granted, Intent data) {
        if (granted) {
            mediaProjection = mediaProjectionManager.getMediaProjection(Activity.RESULT_OK, data);
            projectionCallback = new MediaProjection.Callback() {
                @Override
                public void onStop() {
                    stopSelf();
                }
            };
            mediaProjection.registerCallback(projectionCallback, mainHandler);
            startCapture();
        } else {
            Toast.makeText(this, "屏幕捕获权限被拒绝", Toast.LENGTH_SHORT).show();
            stopSelf();
        }
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

        // 设置捕获分辨率，保持比例，宽度最大 720
        float scale = Math.min(720f / screenWidth, 1280f / screenHeight);
        captureWidth = Math.round(screenWidth * scale);
        captureHeight = Math.round(screenHeight * scale);

        imageReader = ImageReader.newInstance(captureWidth, captureHeight, PixelFormat.RGBA_8888, 2);
        virtualDisplay = mediaProjection.createVirtualDisplay(
                "ScreenCapture",
                captureWidth, captureHeight, metrics.densityDpi,
                VIRTUAL_DISPLAY_FLAGS,
                imageReader.getSurface(),
                null, null);

        captureThread = new HandlerThread("CaptureThread");
        captureThread.start();
        captureHandler = new Handler(captureThread.getLooper());
        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image != null) {
                Bitmap frame = imageToBitmap(image);
                image.close();
                if (frame != null && poseEstimator != null) {
                    processFrame(frame);
                }
            }
        }, captureHandler);

        isCapturing = true;
        Log.d(TAG, "Screen capture started: " + captureWidth + "x" + captureHeight);
    }

    private void processFrame(Bitmap frame) {
        int rotation = getScreenRotation();
        Log.d(TAG, "Frame size: " + frame.getWidth() + "x" + frame.getHeight() + ", rotation: " + rotation);

        PoseEstimator.PersonPose pose = poseEstimator.estimate(frame);
        if (overlayView != null) {
            // pose 为 null 也要调用，让 OverlayView 的 missingFrameCount 逻辑生效（框平滑消失）
            mainHandler.post(() -> overlayView.updatePose(pose, frame.getWidth(), frame.getHeight()));
        }
    }

    private int getScreenRotation() {
        Display display = windowManager.getDefaultDisplay();
        return display.getRotation();
    }

    private Bitmap imageToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        int pixelStride = planes[0].getPixelStride();
        int rowStride = planes[0].getRowStride();
        int rowPadding = rowStride - pixelStride * captureWidth;
        int width = captureWidth + rowPadding / pixelStride;

        buffer.rewind();
        int[] pixels = new int[width * captureHeight];
        for (int y = 0; y < captureHeight; y++) {
            for (int x = 0; x < width; x++) {
                int index = (y * width + x) * pixelStride;
                int r = buffer.get(index) & 0xFF;
                int g = buffer.get(index + 1) & 0xFF;
                int b = buffer.get(index + 2) & 0xFF;
                int a = buffer.get(index + 3) & 0xFF;
                pixels[y * width + x] = (a << 24) | (r << 16) | (g << 8) | b;
            }
        }

        Bitmap bitmap = Bitmap.createBitmap(captureWidth, captureHeight, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(pixels, 0, width, 0, 0, captureWidth, captureHeight);
        return bitmap;
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
        if (poseEstimator != null) {
            poseEstimator.close();
            poseEstimator = null;
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