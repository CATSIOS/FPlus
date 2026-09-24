package com.example.fplus;

import android.content.ComponentName;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.util.Log;

import rikka.shizuku.Shizuku;

/**
 * 触摸屏监听客户端：绑定 Shizuku UserService，查询用户是否正用手触摸屏幕。
 * 用于"检测让路"：用户触摸时暂停自动跟随，避免与用户操作冲突。
 */
public class TouchMonitorClient {
    private static final String TAG = "TouchMonitorClient";

    private ITouchMonitor service;
    private boolean bound = false;
    private boolean started = false;

    private final Shizuku.UserServiceArgs args;
    private final ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder binder) {
            service = ITouchMonitor.Stub.asInterface(binder);
            Log.i(TAG, "touch monitor connected");
            // 服务连上后补做 start（bind 是异步的）
            if (!started) {
                started = true;
                doStart();
            }
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            service = null;
            started = false;
            Log.i(TAG, "touch monitor disconnected");
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
            Log.i(TAG, "touch monitor started");
        } catch (Throwable t) {
            Log.w(TAG, "start failed", t);
        }
    }

    /**
     * 用户是否正用手触摸真实屏幕。
     * 未连接或查询失败时返回 false（退化为不检测，不影响正常跟随）。
     */
    public boolean isUserTouching() {
        if (service == null) return false;
        try {
            return service.isUserTouching();
        } catch (Throwable t) {
            return false;
        }
    }

    /** 手指当前归一化 X 坐标（0~1），失败时返回 -1 */
    public float getTouchX() {
        if (service == null) return -1f;
        try {
            return service.getTouchX();
        } catch (Throwable t) {
            return -1f;
        }
    }

    /** 手指当前归一化 Y 坐标（0~1），失败时返回 -1 */
    public float getTouchY() {
        if (service == null) return -1f;
        try {
            return service.getTouchY();
        } catch (Throwable t) {
            return -1f;
        }
    }
}
