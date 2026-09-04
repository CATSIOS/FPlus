package com.example.fplus;

import android.content.Intent;
import android.content.SharedPreferences;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Settings;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_OVERLAY_PERMISSION = 2001;
    private static final int REQUEST_CODE_CAPTURE = 2002;

    private SharedPreferences prefs;
    private Button btnStart;
    private TextView textConfig;

    @Override
    protected void onResume() {
        super.onResume();
        refreshConfigText();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);

        // 左侧常驻导航侧边栏：点击进入各独立设置界面
        String[] navItems = {
                getString(R.string.model_option_title),
                getString(R.string.gpu_option_title),
                getString(R.string.models_option_title),
                getString(R.string.benchmark_option_title),
                getString(R.string.btn_advanced)
        };
        ListView navList = findViewById(R.id.nav_list);
        navList.setAdapter(new ArrayAdapter<>(this, R.layout.settings_nav_item, navItems));
        navList.setOnItemClickListener((parent, view, position, id) -> {
            Intent intent;
            switch (position) {
                case 0: intent = new Intent(this, ModelSettingsActivity.class); break;
                case 1: intent = new Intent(this, BackendSettingsActivity.class); break;
                case 2: intent = new Intent(this, ModelManagerActivity.class); break;
                case 3: intent = new Intent(this, BenchmarkActivity.class); break;
                default: intent = new Intent(this, AdvancedOptionsActivity.class); break;
            }
            startActivity(intent);
        });

        btnStart = findViewById(R.id.btn_start);
        textConfig = findViewById(R.id.text_config);
        btnStart.setOnClickListener(v -> {
            // 再次按下"开始"时若服务在运行则停止（toggle 行为）
            if (ScreenCaptureService.getInstance() != null) {
                stopService(new Intent(this, ScreenCaptureService.class));
                btnStart.setText(R.string.btn_start);
            } else if (!ModelManager.isDownloaded(this, currentModelName())) {
                Toast.makeText(this, "模型未下载，请先在模型管理中下载", Toast.LENGTH_SHORT).show();
                startActivity(new Intent(this, ModelManagerActivity.class));
            } else if (checkOverlayPermission()) {
                requestScreenCapture();
            } else {
                requestOverlayPermission();
            }
        });
    }

    private String currentModelName() {
        return prefs.getString("model_name", ModelManager.DEFAULT_MODEL);
    }

    /** 在主界面显示当前选择的模型与推理后端（onResume 时刷新，设置页返回后同步） */
    private void refreshConfigText() {
        if (textConfig == null) return;
        String model = currentModelName();
        textConfig.setText(getString(R.string.main_config_format, model, backendLabel()));
    }

    private String backendLabel() {
        String backend = prefs.getString("backend", PoseEstimator.Backend.GPU.name());
        try {
            switch (PoseEstimator.Backend.valueOf(backend)) {
                case NNAPI: return getString(R.string.backend_npu);
                case GPU:   return getString(R.string.backend_gpu);
                default:    return getString(R.string.backend_cpu);
            }
        } catch (IllegalArgumentException e) {
            return getString(R.string.backend_gpu);
        }
    }

    /**
     * 检查悬浮窗权限（minSdk=29 ≥ M，无版本分支判断）
     */
    private boolean checkOverlayPermission() {
        return Settings.canDrawOverlays(this);
    }

    /**
     * 请求悬浮窗权限
     */
    private void requestOverlayPermission() {
        Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:" + getPackageName()));
        startActivityForResult(intent, REQUEST_CODE_OVERLAY_PERMISSION);
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
        // minSdk=29 ≥ O，直接 startForegroundService
        startForegroundService(serviceIntent);
        Toast.makeText(this, "服务已启动", Toast.LENGTH_SHORT).show();
        btnStart.setText(R.string.btn_stop);
    }
}
