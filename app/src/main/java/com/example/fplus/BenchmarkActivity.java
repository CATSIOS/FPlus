package com.example.fplus;

import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** 自动测速界面：遍历本地已下载的「模型 × 后端」组合，选出最快并应用 */
public class BenchmarkActivity extends AppCompatActivity {

    private SharedPreferences prefs;
    private Button btnBenchmark;
    private TextView benchmarkResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_benchmark);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }
        toolbar.setNavigationOnClickListener(v -> finish());

        prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);
        btnBenchmark = findViewById(R.id.btn_benchmark);
        benchmarkResult = findViewById(R.id.benchmark_result);
        btnBenchmark.setOnClickListener(v -> runBenchmark());
    }

    private void runBenchmark() {
        btnBenchmark.setEnabled(false);
        benchmarkResult.setText("测速中…");

        // 用 applicationContext 做模型 IO、WeakReference 回传 Activity，
        // 避免后台线程持有 Activity 引用导致内存泄漏（用户中途退出时无法回收）。
        final android.content.Context appContext = getApplicationContext();
        final android.os.Handler mainHandler = new android.os.Handler(android.os.Looper.getMainLooper());
        final java.lang.ref.WeakReference<BenchmarkActivity> weakSelf =
                new java.lang.ref.WeakReference<>(this);

        new Thread(() -> {
            Bitmap testBitmap = Bitmap.createBitmap(720, 450, Bitmap.Config.ARGB_8888);
            try {
                testBitmap.eraseColor(0xFF808080);

                String[] models = ModelManager.listLocalModels(appContext).toArray(new String[0]);
                if (models.length == 0) {
                    mainHandler.post(() -> {
                        BenchmarkActivity a = weakSelf.get();
                        if (a == null || a.isDestroyed()) return;
                        a.btnBenchmark.setEnabled(true);
                        a.benchmarkResult.setText("没有已下载的模型，请先在模型管理中下载");
                    });
                    return;
                }
                // 测速只测 NPU / GPU，CPU 永远最慢，跳过以大幅缩短测速时间
                PoseEstimator.Backend[] backends = {
                        PoseEstimator.Backend.NNAPI,
                        PoseEstimator.Backend.GPU
                };

                List<BenchResult> results = new ArrayList<>();
                long bestMs = Long.MAX_VALUE;
                String bestModel = null;
                PoseEstimator.Backend bestBackend = null;

                for (String model : models) {
                    for (PoseEstimator.Backend backend : backends) {
                        long ms = benchmarkOne(appContext, testBitmap, model, backend);
                        String label = modelLabel(model) + " + " + backendLabel(backend);
                        results.add(new BenchResult(label, ms));
                        if (ms > 0 && ms < bestMs) {
                            bestMs = ms;
                            bestModel = model;
                            bestBackend = backend;
                        }
                    }
                }

                // 按耗时从快到慢排序（失败放最后）
                Collections.sort(results, (a, b) -> {
                    if (a.ms <= 0 && b.ms <= 0) return 0;
                    if (a.ms <= 0) return 1;
                    if (b.ms <= 0) return -1;
                    return Long.compare(a.ms, b.ms);
                });

                StringBuilder sb = new StringBuilder();
                for (BenchResult r : results) {
                    sb.append(r.label).append(": ")
                            .append(r.ms > 0 ? r.ms + "ms" : "失败")
                            .append("\n");
                }
                Log.d("Benchmark", "测速结果: " + sb.toString().replace("\n", " | "));

                final String resultText = sb.toString();
                final String fModel = bestModel;
                final PoseEstimator.Backend fBackend = bestBackend;

                mainHandler.post(() -> {
                    BenchmarkActivity a = weakSelf.get();
                    if (a == null || a.isDestroyed()) return;
                    a.btnBenchmark.setEnabled(true);
                    if (fModel != null && fBackend != null) {
                        a.prefs.edit()
                                .putString("model_name", fModel)
                                .putString("backend", fBackend.name())
                                .apply();
                        a.benchmarkResult.setText(resultText + "\n已自动选择: "
                                + modelLabel(fModel) + " + " + backendLabel(fBackend));
                    } else {
                        a.benchmarkResult.setText(resultText + "\n测速失败");
                    }
                });
            } finally {
                testBitmap.recycle();
            }
        }).start();
    }

    private static long benchmarkOne(android.content.Context context, Bitmap bitmap, String model, PoseEstimator.Backend backend) {
        if (!ModelManager.isDownloaded(context, model)) {
            Log.w("Benchmark", model + " 未下载，跳过测速");
            return -1;
        }
        PoseEstimator estimator = null;
        try {
            estimator = new PoseEstimator(context, model, backend);
            // 预热 2 次
            for (int i = 0; i < 2; i++) {
                estimator.estimate(bitmap);
            }
            // 计时 8 次
            int rounds = 8;
            long start = System.currentTimeMillis();
            for (int i = 0; i < rounds; i++) {
                estimator.estimate(bitmap);
            }
            long elapsed = System.currentTimeMillis() - start;
            return Math.round((float) elapsed / rounds);
        } catch (Exception e) {
            Log.w("Benchmark", model + " + " + backend + " 失败", e);
            return -1;
        } finally {
            if (estimator != null) {
                estimator.close();
            }
        }
    }

    private static String modelLabel(String model) {
        return model.endsWith(".tflite")
                ? model.substring(0, model.length() - ".tflite".length())
                : model;
    }

    private static String backendLabel(PoseEstimator.Backend backend) {
        switch (backend) {
            case NNAPI:
                return "NPU";
            case GPU:
                return "GPU";
            default:
                return "CPU";
        }
    }

    private static class BenchResult {
        final String label;
        final long ms;

        BenchResult(String label, long ms) {
            this.label = label;
            this.ms = ms;
        }
    }
}
