package com.example.fplus;

/**
 * 1€ 滤波器（One Euro Filter）：自适应低通滤波。
 * 静止/慢速时强平滑（过滤检测抖动），快速移动时弱平滑（保持响应）。
 * 截止频率随信号速度自适应，且用 dt 归一化，帧率无关。
 *
 * 参考：Casiez et al. "1€ Filter: A Simple Speed-based Low-pass Filter
 * for Noisy Input in Interactive Systems", CHI 2012.
 */
public class OneEuroFilter {

    private final double minCutoff; // 最小截止频率（Hz），静止时的平滑度
    private final double beta;      // 速度系数，快速移动时的响应度
    private static final double D_CUTOFF = 1.0; // 速度低通滤波截止频率（Hz）

    private double xHat;      // 滤波后的值
    private double dxHat;     // 滤波后的速度
    private double xPrev;     // 上一个原始值
    private long lastTimeNanos = -1;
    private boolean initialized = false;

    public OneEuroFilter(double minCutoff, double beta) {
        this.minCutoff = minCutoff;
        this.beta = beta;
    }

    public void reset() {
        initialized = false;
        lastTimeNanos = -1;
    }

    /** 滤波一个采样值，返回滤波结果 */
    public double filter(double x, long timeNanos) {
        if (!initialized) {
            xHat = x;
            dxHat = 0;
            xPrev = x;
            lastTimeNanos = timeNanos;
            initialized = true;
            return xHat;
        }

        double dt = (timeNanos - lastTimeNanos) / 1e9;
        if (dt <= 0) dt = 1.0 / 60.0;
        lastTimeNanos = timeNanos;

        // 1. 原始信号的速度（导数）
        double dx = (x - xPrev) / dt;
        xPrev = x;

        // 2. 速度低通滤波（固定截止频率，降低速度噪声）
        double alphaD = smoothingFactor(D_CUTOFF, dt);
        dxHat = dxHat + alphaD * (dx - dxHat);

        // 3. 自适应截止频率：速度越大，截止频率越高，响应越快
        double fc = minCutoff + beta * Math.abs(dxHat);

        // 4. 信号低通滤波
        double alpha = smoothingFactor(fc, dt);
        xHat = xHat + alpha * (x - xHat);

        return xHat;
    }

    /** 平滑因子：alpha = 1 / (1 + tau/dt)，tau = 1/(2*pi*fc) */
    private static double smoothingFactor(double fc, double dt) {
        double tau = 1.0 / (2.0 * Math.PI * fc);
        return 1.0 / (1.0 + tau / dt);
    }
}
