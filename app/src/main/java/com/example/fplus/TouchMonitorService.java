package com.example.fplus;

import android.content.Context;

import androidx.annotation.Keep;

/**
 * Shizuku UserService：运行在 shell UID 进程，通过 JNI 只读监听真实触摸屏。
 *
 * 本类不是 Android Service，而是 Shizuku 通过反射 new 出来的 IBinder
 * （继承 AIDL 生成的 ITouchMonitor.Stub）。因此：
 *   - 必须有「无参构造」或「@Keep 标注的 Context 构造」（Shizuku 反射调用）
 *   - 不需要在 AndroidManifest 声明
 *   - destroy() 由 Shizuku 用固定 transaction code(16777114) 调用
 */
public class TouchMonitorService extends ITouchMonitor.Stub {

    public TouchMonitorService() {
    }

    @Keep
    public TouchMonitorService(Context context) {
    }

    @Override
    public void destroy() {
        nativeDestroy();
        System.exit(0);
    }

    @Override
    public void start() {
        nativeStart();
    }

    @Override
    public boolean isUserTouching() {
        return nativeIsUserTouching() != 0;
    }

    @Override
    public float getTouchX() {
        return nativeGetTouchX();
    }

    @Override
    public float getTouchY() {
        return nativeGetTouchY();
    }

    static {
        System.loadLibrary("touchmonitor");
    }

    private native void nativeStart();
    private native int nativeIsUserTouching();
    private native float nativeGetTouchX();
    private native float nativeGetTouchY();
    private native void nativeDestroy();
}
