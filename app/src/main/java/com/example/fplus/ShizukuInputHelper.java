package com.example.fplus;

import android.content.pm.PackageManager;
import android.os.IBinder;
import android.os.SystemClock;
import android.view.InputEvent;
import android.view.MotionEvent;

import java.lang.reflect.Method;

import rikka.shizuku.Shizuku;
import rikka.shizuku.ShizukuBinderWrapper;
import rikka.shizuku.SystemServiceHelper;

/**
 * 通过 Shizuku 反射调用 InputManager.injectInputEvent，
 * 手动注入多段 ACTION_MOVE，避免 input swipe 被识别成点击。
 */
public class ShizukuInputHelper {

    private static Object sInputManager;
    private static Method sInjectMethod;

    private static synchronized boolean init() {
        if (sInputManager != null) return true;
        try {
            IBinder binder = SystemServiceHelper.getSystemService("input");
            IBinder wrapped = new ShizukuBinderWrapper(binder);
            Class<?> stub = Class.forName("android.hardware.input.IInputManager$Stub");
            Method asInterface = stub.getDeclaredMethod("asInterface", IBinder.class);
            asInterface.setAccessible(true);
            sInputManager = asInterface.invoke(null, wrapped);
            sInjectMethod = sInputManager.getClass().getMethod(
                    "injectInputEvent", InputEvent.class, int.class);
            return true;
        } catch (Throwable t) {
            // 初始化失败（通常是 Shizuku 未就绪）：清空字段，允许下次重试，
            // 不能用永久标志锁死，否则 Shizuku 授权后滑动依然失效
            sInputManager = null;
            sInjectMethod = null;
            return false;
        }
    }

    /** Shizuku 是否就绪（服务在线 + 权限授予 + 反射初始化成功） */
    public static synchronized boolean isReady() {
        try {
            if (!Shizuku.pingBinder()) return false;
            if (Shizuku.checkSelfPermission() != PackageManager.PERMISSION_GRANTED) return false;
            return init();
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * 拟人化滑动：从 (x1, y1) 滑到 (x2, y2)，duration 毫秒。
     * 轨迹特征：smoothstep 缓动（起停慢、中间快）+ 轻微弧线弯曲 + 随机抖动，
     * 模拟真人手指的加速-巡航-减速过程，避免被识别为匀速直线。
     * 该方法会阻塞调用线程，请放在子线程执行。
     *
     * @return true 表示注入完成
     */
    public static boolean swipe(int x1, int y1, int x2, int y2, long duration) {
        if (!isReady()) return false;
        if (duration <= 0) duration = 200;
        try {
            long downTime = SystemClock.uptimeMillis();

            float dx = x2 - x1;
            float dy = y2 - y1;
            float len = (float) Math.sqrt(dx * dx + dy * dy);
            if (len < 1f) len = 1f;

            // 弧线弯曲幅度：随滑动距离增大，但封顶，方向随机（左弯/右弯）
            float arcAmp = (float) (Math.random() * 2f - 1f) * Math.min(12f, len * 0.06f);
            // 垂直于滑动方向的单位向量，用于叠加弧线偏移
            float nx = -dy / len;
            float ny = dx / len;

            // ACTION_DOWN
            MotionEvent down = MotionEvent.obtain(downTime, downTime,
                    MotionEvent.ACTION_DOWN, x1, y1, 0);
            sInjectMethod.invoke(sInputManager, down, 0);
            down.recycle();

            // 分段 MOVE：至少 20 段，每段约 10ms
            int steps = Math.max(20, (int) (duration / 10));
            long stepDelay = duration / steps;

            for (int i = 1; i < steps; i++) {
                // 时间均匀推进，位置用缓动函数映射 → 瞬时速度呈起停慢、中间快
                float t = i / (float) steps;
                float eased = t * t * (3f - 2f * t);              // smoothstep 缓动
                float arc = arcAmp * (float) Math.sin(Math.PI * t); // 弧线偏移，中点最大
                float jx = (float) (Math.random() * 4f - 2f);      // ±2px 随机抖动
                float jy = (float) (Math.random() * 4f - 2f);

                float x = x1 + dx * eased + nx * arc + jx;
                float y = y1 + dy * eased + ny * arc + jy;

                long eventTime = downTime + stepDelay * i;
                MotionEvent move = MotionEvent.obtain(downTime, eventTime,
                        MotionEvent.ACTION_MOVE, x, y, 0);
                sInjectMethod.invoke(sInputManager, move, 0);
                move.recycle();

                if (stepDelay > 0) {
                    try { Thread.sleep(stepDelay); } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return false;
                    }
                }
            }

            // ACTION_UP（终点坐标，与目标位置一致）
            long upTime = downTime + duration;
            MotionEvent up = MotionEvent.obtain(downTime, upTime,
                    MotionEvent.ACTION_UP, x2, y2, 0);
            sInjectMethod.invoke(sInputManager, up, 0);
            up.recycle();

            return true;
        } catch (Throwable t) {
            return false;
        }
    }
}
