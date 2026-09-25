package com.example.fplus;

import android.content.ComponentName;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.util.Log;

import rikka.shizuku.Shizuku;

/**
 * 触摸屏监听客户端：绑定 Shizuku UserService，查询用户是否正用手触摸屏幕。
 *
 * 性能设计：UserService 运行在独立进程，每次查询都是跨进程调用（Binder IPC）。
 * 若在推理线程每帧同步查询，会阻塞推理、拖慢跟随。因此这里用独立后台线程
 * 周期性轮询并把结果缓存到 volatile 字段，推理线程只读本地缓存（零 IPC 开销）。
 * 代价是让路判断最多滞后一个轮询周期（约 30ms），用户无感。
 */
public class TouchMonitorClient {
    private static final String TAG = "TouchMonitorClient";
    private static final long POLL_INTERVAL_MS = 30;   // 轮询周期

    private volatile ITouchMonitor service;
    private volatile boolean bound = false;
    private volatile boolean started = false;

    // ===== 后台轮询缓存（推理线程直接读，无跨进程开销）=====
    private volatile boolean userTouching = false;
    private volatile float touchX = -1f;
    private volatile float touchY = -1f;

    private Thread pollThread;
    private volatile boolean polling = false;

    private final Shizuku.UserServiceArgs args;
    private final ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder binder) {
            service = ITouchMonitor.Stub.asInterface(binder);
            if (!started) {
                started = true;
                doStart();
            }
            startPolling();
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            service = null;
            started = false;
            stopPolling();
        }
    };

    public TouchMonitorClient() {
        args = new Shizuku.UserServiceArgs(
                new ComponentName("com.example.fplus", TouchMonitorService.class.getName()))
                .daemon(false)
                .version(1)
                .processNameSuffix("touchmon");
    }

    public void bind() {
        if (bound) return;
        bound = true;
        try {
            Shizuku.bindUserService(args, connection);
        } catch (Throwable t) {
            bound = false;
            Log.w(TAG, "bindUserService failed: " + t.getMessage());
        }
    }

    public void unbind() {
        if (!bound) return;
        stopPolling();
        try {
            Shizuku.unbindUserService(args, connection, true);
        } catch (Throwable ignored) {
        }
        bound = false;
        service = null;
        started = false;
    }

    private void doStart() {
        if (service == null) return;
        try {
            service.start();
        } catch (Throwable t) {
            Log.w(TAG, "start failed", t);
        }
    }

    private void startPolling() {
        if (polling) return;
        polling = true;
        pollThread = new Thread(() -> {
            while (polling) {
                ITouchMonitor s = service;
                if (s != null) {
                    try {
                        float[] st = s.getState();
                        if (st != null && st.length >= 3) {
                            userTouching = st[0] != 0f;
                            touchX = st[1];
                            touchY = st[2];
                        }
                    } catch (Throwable ignored) {
                        // 服务重启瞬间可能抛异常，下一轮重试
                    }
                }
                try {
                    Thread.sleep(POLL_INTERVAL_MS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "TouchMonitorPoll");
        pollThread.setDaemon(true);
        pollThread.start();
    }

    private void stopPolling() {
        polling = false;
        Thread t = pollThread;
        pollThread = null;
        if (t != null) t.interrupt();
        userTouching = false;
        touchX = -1f;
        touchY = -1f;
    }

    // ===== 以下 getter 全部读本地缓存，无跨进程调用 =====

    /** 用户是否正用手触摸真实屏幕 */
    public boolean isUserTouching() {
        return userTouching;
    }

    /** 手指当前归一化 X 坐标（0~1），未触摸时无意义 */
    public float getTouchX() {
        return touchX;
    }

    /** 手指当前归一化 Y 坐标（0~1），未触摸时无意义 */
    public float getTouchY() {
        return touchY;
    }
}
