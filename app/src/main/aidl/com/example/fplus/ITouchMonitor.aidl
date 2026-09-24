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

    /**
     * 一次性获取全部状态：[是否触摸, 归一化X, 归一化Y]。
     * 客户端用后台线程轮询本方法并缓存，避免每帧多次跨进程调用阻塞推理线程。
     */
    float[] getState() = 2;
}
