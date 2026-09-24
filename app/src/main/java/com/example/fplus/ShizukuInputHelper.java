package com.example.fplus;

import android.content.pm.PackageManager;
import android.os.IBinder;
import android.os.SystemClock;
import android.view.InputDevice;
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

    // 复用触点属性/坐标，模拟压力与接触面积波动（pointerId=0 已验证安全）
    private static final MotionEvent.PointerProperties sProps = new MotionEvent.PointerProperties();
    static {
        sProps.id = 0;
        sProps.toolType = MotionEvent.TOOL_TYPE_FINGER;
    }

    /** 构造带压力/面积的触点事件（deviceId=0，与真实触摸同源，避免越界崩溃） */
    private static MotionEvent obtainTouch(long downTime, long eventTime, int action,
                                           float x, float y, float pressure) {
        MotionEvent.PointerCoords coords = new MotionEvent.PointerCoords();
        coords.x = x;
        coords.y = y;
        coords.pressure = pressure;
        coords.size = 0.9f + (float) Math.random() * 0.2f;  // 接触面积轻微波动
        return MotionEvent.obtain(
                downTime, eventTime, action, 1,
                new MotionEvent.PointerProperties[]{sProps},
                new MotionEvent.PointerCoords[]{coords},
                0, 0, 1f, 1f,
                0,                  // deviceId=0（默认触摸屏）
                0,                  // edgeFlags
                InputDevice.SOURCE_TOUCHSCREEN,
                0);
    }

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
     * 反作弊规避要点：
     *   - 起点/终点加随机偏移（真人不会每次从同一像素点开始）
     *   - 事件时间戳加抖动（不规律采样间隔）
     *   - 压力随按压进程波动（按下渐强 → 滑动中波动 → 抬起前衰减）
     *   - smoothstep 缓动（起停慢、中间快）+ 弧线 + 抖动
     * 该方法会阻塞调用线程，请放在子线程执行。
     *
     * @return true 表示注入完成
     */
    public static boolean swipe(int x1, int y1, int x2, int y2, long duration) {
        if (!isReady()) return false;
        if (duration <= 0) duration = 200;
        try {
            // 起点加 ±10px 随机偏移（最重要的反作弊特征：不固定落点）
            float sx = x1 + (float) (Math.random() * 20f - 10f);
            float sy = y1 + (float) (Math.random() * 20f - 10f);
            // 终点加 ±6px 随机偏移
            float ex = x2 + (float) (Math.random() * 12f - 6f);
            float ey = y2 + (float) (Math.random() * 12f - 6f);

            long downTime = SystemClock.uptimeMillis();

            float dx = ex - sx;
            float dy = ey - sy;
            float len = (float) Math.sqrt(dx * dx + dy * dy);
            if (len < 1f) len = 1f;

            // 弧线弯曲幅度：随滑动距离增大，但封顶，方向随机（左弯/右弯）
            float arcAmp = (float) (Math.random() * 2f - 1f) * Math.min(14f, len * 0.07f);
            float nx = -dy / len;
            float ny = dx / len;

            // ACTION_DOWN：压力从较低值开始（真实手指刚接触时压力较小）
            MotionEvent down = obtainTouch(downTime, downTime,
                    MotionEvent.ACTION_DOWN, sx, sy, 0.45f + (float) Math.random() * 0.15f);
            sInjectMethod.invoke(sInputManager, down, 0);
            down.recycle();

            // 分段 MOVE：至少 20 段，每段约 8~12ms
            int steps = Math.max(20, (int) (duration / 10));
            long stepDelay = duration / steps;

            for (int i = 1; i < steps; i++) {
                float t = i / (float) steps;
                float eased = t * t * (3f - 2f * t);              // smoothstep 缓动
                float arc = arcAmp * (float) Math.sin(Math.PI * t);
                float jx = (float) (Math.random() * 4f - 2f);
                float jy = (float) (Math.random() * 4f - 2f);

                float x = sx + dx * eased + nx * arc + jx;
                float y = sy + dy * eased + ny * arc + jy;

                // 压力：滑动中在 0.6~1.0 之间波动
                float pressure = 0.65f + (float) Math.random() * 0.35f;

                // 时间戳加 ±3ms 抖动，避免采样间隔完全规律
                long jitter = (long) (Math.random() * 6f - 3f);
                long eventTime = downTime + stepDelay * i + jitter;
                MotionEvent move = obtainTouch(downTime, eventTime,
                        MotionEvent.ACTION_MOVE, x, y, pressure);
                sInjectMethod.invoke(sInputManager, move, 0);
                move.recycle();

                if (stepDelay > 0) {
                    try { Thread.sleep(stepDelay); } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return false;
                    }
                }
            }

            // ACTION_UP：抬起前压力衰减
            long upTime = downTime + duration;
            MotionEvent up = obtainTouch(downTime, upTime,
                    MotionEvent.ACTION_UP, ex, ey, 0.3f + (float) Math.random() * 0.15f);
            sInjectMethod.invoke(sInputManager, up, 0);
            up.recycle();

            return true;
        } catch (Throwable t) {
            return false;
        }
    }
}
