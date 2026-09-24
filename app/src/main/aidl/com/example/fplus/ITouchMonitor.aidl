package com.example.fplus;

/**
 * 触摸屏监听服务接口（运行在 Shizuku UserService 进程，shell UID）。
 * 只读打开真实触摸屏，用于判断用户是否正用手触摸屏幕，供"检测让路"使用。
 *
 * 注意：destroy() 的 transaction code 必须为 16777114（Shizuku 约定），
 * Shizuku 服务停止时会用固定 transaction code 调用它做清理。
 */
interface ITouchMonitor {

    /** Shizuku 保留的销毁方法，transaction code 固定 16777114 */
    void destroy() = 16777114;

    /** 开始监听真实触摸屏 */
    void start() = 1;

    /** 用户是否正用手触摸屏幕 */
    boolean isUserTouching() = 2;

    /** 手指当前归一化 X 坐标（0~1，相对触摸屏原始坐标系） */
    float getTouchX() = 3;

    /** 手指当前归一化 Y 坐标（0~1，相对触摸屏原始坐标系） */
    float getTouchY() = 4;
}
