package com.example.fplus;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.widget.Button;
import android.widget.ExpandableListView;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_OVERLAY_PERMISSION = 2001;
    private static final int REQUEST_CODE_CAPTURE = 2002;

    private static final String MODEL_DEFAULT = "sunxds_0.8.0.tflite";
    private static final String MODEL_NHWC = "sunxds_0.8.0_float32.tflite";
    private static final String MODEL_INT8 = "sunxds_0.8.0_w8a16.tflite";
    private static final String MODEL_416 = "sunxds_0.8.0_416.tflite";
    private static final String MODEL_320_OPT = "sunxds_0.8.0_320_opt.tflite";
    private static final String MODEL_416_OPT = "sunxds_0.8.0_416_opt.tflite";

    private SharedPreferences prefs;
    private RadioGroup radioModel;
    private RadioGroup radioBackend;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);

        // 左侧：原生可折叠设置（Android Settings 风格 ExpandableListView）
        String[] groups = {getString(R.string.model_option_title),
                           getString(R.string.gpu_option_title)};
        SettingsExpandableAdapter settingsAdapter = new SettingsExpandableAdapter(this, groups);
        ExpandableListView expandable = findViewById(R.id.expandable_settings);
        expandable.setAdapter(settingsAdapter);
        expandable.expandGroup(0);
        expandable.expandGroup(1);
        // 防止折叠时父拦截点击影响 groupIndicator，保留原生折叠动画
        expandable.setOnGroupClickListener((parent, v, groupPosition, id) -> false);

        // Adapter 构造时已预加载 child view，直接取引用（不会 NPE）
        radioModel = settingsAdapter.getModelRadioGroup();
        String savedModel = prefs.getString("model_name", MODEL_DEFAULT);
        radioModel.check(checkedIdForModel(savedModel));
        radioModel.setOnCheckedChangeListener((group, checkedId) ->
                prefs.edit().putString("model_name", modelForCheckedId(checkedId)).apply());

        radioBackend = settingsAdapter.getBackendRadioGroup();

        String savedBackend = prefs.getString("backend", PoseEstimator.Backend.GPU.name());
        PoseEstimator.Backend backend;
        try {
            backend = PoseEstimator.Backend.valueOf(savedBackend);
        } catch (IllegalArgumentException e) {
            backend = PoseEstimator.Backend.GPU;
        }
        switch (backend) {
            case GPU:
                radioBackend.check(R.id.radio_gpu);
                break;
            case CPU:
                radioBackend.check(R.id.radio_cpu);
                break;
            case NNAPI:
            default:
                radioBackend.check(R.id.radio_npu);
                break;
        }
        radioBackend.setOnCheckedChangeListener((group, checkedId) -> {
            PoseEstimator.Backend b;
            if (checkedId == R.id.radio_gpu) {
                b = PoseEstimator.Backend.GPU;
            } else if (checkedId == R.id.radio_cpu) {
                b = PoseEstimator.Backend.CPU;
            } else {
                b = PoseEstimator.Backend.NNAPI;
            }
            prefs.edit().putString("backend", b.name()).apply();
        });

        Button btnBenchmark = findViewById(R.id.btn_benchmark);
        btnBenchmark.setOnClickListener(v -> runBenchmark());

        Button btnAdvanced = findViewById(R.id.btn_advanced);
        btnAdvanced.setOnClickListener(v -> {
            Intent intent = new Intent(this, AdvancedOptionsActivity.class);
            startActivity(intent);
        });

        Button btnStart = findViewById(R.id.btn_start);
        btnStart.setOnClickListener(v -> {
            // 再次按下"开始"时若服务在运行则停止（toggle 行为）
            if (ScreenCaptureService.getInstance() != null) {
                stopService(new Intent(this, ScreenCaptureService.class));
                btnStart.setText(R.string.btn_start);
            } else if (checkOverlayPermission()) {
                requestScreenCapture();
            } else {
                requestOverlayPermission();
            }
        });
    }

    /**
     * 检查悬浮窗权限
     */
    private boolean checkOverlayPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return Settings.canDrawOverlays(this);
        }
        return true; // Android 6.0 以下默认有权限
    }

    /**
     * 请求悬浮窗权限
     */
    private void requestOverlayPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + getPackageName()));
            startActivityForResult(intent, REQUEST_CODE_OVERLAY_PERMISSION);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE_OVERLAY_PERMISSION) {
            if (checkOverlayPermission()) {
                requestScreenCapture();
            } else {
                Toast.makeText(this, "需要悬浮窗权限才能显示骨骼", Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == REQUEST_CODE_CAPTURE) {
            if (resultCode == RESULT_OK && data != null) {
                startScreenCaptureService(resultCode, data);
            } else {
                Toast.makeText(this, "未授予屏幕捕获权限", Toast.LENGTH_SHORT).show();
            }
        }
    }

    /**
     * 请求屏幕捕获授权（授权须先于 mediaProjection 前台服务启动）
     */
    private void requestScreenCapture() {
        MediaProjectionManager mpm = (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        startActivityForResult(mpm.createScreenCaptureIntent(), REQUEST_CODE_CAPTURE);
    }

    /**
     * 授权成功后启动屏幕捕获服务
     */
    private void startScreenCaptureService(int resultCode, Intent resultData) {
        Intent serviceIntent = new Intent(this, ScreenCaptureService.class);
        serviceIntent.putExtra("resultCode", resultCode);
        serviceIntent.putExtra("resultData", resultData);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent);
        } else {
            startService(serviceIntent);
        }
        Toast.makeText(this, "服务已启动", Toast.LENGTH_SHORT).show();
        Button btnStart = findViewById(R.id.btn_start);
        btnStart.setText(R.string.btn_stop);
    }

    /**
     * 自动测速：遍历所有「模型 × 后端」组合，测平均推理耗时，选出最快并应用
     */
    private void runBenchmark() {
        Button btnBenchmark = findViewById(R.id.btn_benchmark);
        TextView benchmarkResult = findViewById(R.id.benchmark_result);
        btnBenchmark.setEnabled(false);
        benchmarkResult.setText("测速中…");

        new Thread(() -> {
            Bitmap testBitmap = Bitmap.createBitmap(720, 450, Bitmap.Config.ARGB_8888);
            testBitmap.eraseColor(0xFF808080);

            // 测 640/416/320opt/416opt（NHWC/INT8 实测更慢，已证伪，不再参与测速）
            String[] models = {MODEL_DEFAULT, MODEL_416, MODEL_320_OPT, MODEL_416_OPT};
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
                    long ms = benchmarkOne(testBitmap, model, backend);
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

            testBitmap.recycle();

            final String resultText = sb.toString();
            final String fModel = bestModel;
            final PoseEstimator.Backend fBackend = bestBackend;

            runOnUiThread(() -> {
                btnBenchmark.setEnabled(true);
                if (fModel != null && fBackend != null) {
                    prefs.edit()
                            .putString("model_name", fModel)
                            .putString("backend", fBackend.name())
                            .apply();
                    syncSelection(fModel, fBackend);
                    benchmarkResult.setText(resultText + "\n已自动选择: "
                            + modelLabel(fModel) + " + " + backendLabel(fBackend));
                } else {
                    benchmarkResult.setText(resultText + "\n测速失败");
                }
            });
        }).start();
    }

    private long benchmarkOne(Bitmap bitmap, String model, PoseEstimator.Backend backend) {
        PoseEstimator estimator = null;
        try {
            estimator = new PoseEstimator(this, model, backend);
            // 预热 2 次
            for (int i = 0; i < 2; i++) {
                estimator.estimate(bitmap);
            }
            // 计时 8 次（足够平均，且大幅缩短测速时间）
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

    private String modelLabel(String model) {
        if (MODEL_NHWC.equals(model)) return "640/NHWC";
        if (MODEL_INT8.equals(model)) return "640/INT8";
        if (MODEL_416.equals(model)) return "416";
        if (MODEL_320_OPT.equals(model)) return "320/OPT(原生训练)";
        if (MODEL_416_OPT.equals(model)) return "416/OPT(简化)";
        return "640/NCHW";
    }

    private String backendLabel(PoseEstimator.Backend backend) {
        switch (backend) {
            case NNAPI:
                return "NPU";
            case GPU:
                return "GPU";
            default:
                return "CPU";
        }
    }

    private void syncSelection(String model, PoseEstimator.Backend backend) {
        radioModel.check(checkedIdForModel(model));
        switch (backend) {
            case GPU:
                radioBackend.check(R.id.radio_gpu);
                break;
            case CPU:
                radioBackend.check(R.id.radio_cpu);
                break;
            case NNAPI:
            default:
                radioBackend.check(R.id.radio_npu);
                break;
        }
    }

    private int checkedIdForModel(String model) {
        if (MODEL_NHWC.equals(model)) return R.id.radio_model_nhwc;
        if (MODEL_INT8.equals(model)) return R.id.radio_model_int8;
        if (MODEL_416.equals(model)) return R.id.radio_model_416;
        if (MODEL_320_OPT.equals(model)) return R.id.radio_model_320_opt;
        if (MODEL_416_OPT.equals(model)) return R.id.radio_model_416_opt;
        return R.id.radio_model_default;
    }

    private String modelForCheckedId(int checkedId) {
        if (checkedId == R.id.radio_model_nhwc) return MODEL_NHWC;
        if (checkedId == R.id.radio_model_int8) return MODEL_INT8;
        if (checkedId == R.id.radio_model_416) return MODEL_416;
        if (checkedId == R.id.radio_model_320_opt) return MODEL_320_OPT;
        if (checkedId == R.id.radio_model_416_opt) return MODEL_416_OPT;
        return MODEL_DEFAULT;
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