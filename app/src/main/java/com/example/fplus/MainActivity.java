package com.example.fplus;

import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
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

import rikka.shizuku.Shizuku;

public class MainActivity extends AppCompatActivity {

    private static final int SHIZUKU_PERMISSION_REQUEST_CODE = 1001;
    private static final int REQUEST_CODE_OVERLAY_PERMISSION = 2001;
    private static final int REQUEST_CODE_CAPTURE = 2002;

    private SharedPreferences prefs;
    private Button btnStart;
    private TextView textConfig;

    private boolean updating = false;           // 是否正在检查/下载更新（防重复点击）

    private Shizuku.OnRequestPermissionResultListener permissionListener;
    private Shizuku.OnBinderReceivedListener binderReceivedListener;
    private Shizuku.OnBinderDeadListener binderDeadListener;

    @Override
    protected void onResume() {
        super.onResume();
        refreshConfigText();
        checkShizukuStatus();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);

        String[] navItems = {
                getString(R.string.model_option_title),
                getString(R.string.gpu_option_title),
                getString(R.string.models_option_title),
                getString(R.string.update_option_title),
                getString(R.string.benchmark_option_title),
                getString(R.string.btn_advanced)
        };
        ListView navList = findViewById(R.id.nav_list);
        navList.setAdapter(new ArrayAdapter<>(this, R.layout.settings_nav_item, navItems));
        navList.setOnItemClickListener((parent, view, position, id) -> {
            if (position == 3) {
                checkForAppUpdate();
                return;
            }
            Intent intent;
            switch (position) {
                case 0: intent = new Intent(this, ModelSettingsActivity.class); break;
                case 1: intent = new Intent(this, BackendSettingsActivity.class); break;
                case 2: intent = new Intent(this, ModelManagerActivity.class); break;
                case 4: intent = new Intent(this, BenchmarkActivity.class); break;
                default: intent = new Intent(this, AdvancedOptionsActivity.class); break;
            }
            startActivity(intent);
        });

        btnStart = findViewById(R.id.btn_start);
        textConfig = findViewById(R.id.text_config);
        btnStart.setOnClickListener(v -> {
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

        setupShizuku();
    }

    /** 初始化 Shizuku：注册 Binder 就绪和权限回调 */
    private void setupShizuku() {
        // 权限申请结果回调
        permissionListener = (requestCode, grantResult) -> {
            if (requestCode == SHIZUKU_PERMISSION_REQUEST_CODE) {
                if (grantResult == PackageManager.PERMISSION_GRANTED) {
                    runOnUiThread(() -> Toast.makeText(this,
                            "Shizuku 权限已授予", Toast.LENGTH_SHORT).show());
                } else {
                    runOnUiThread(() -> Toast.makeText(this,
                            "Shizuku 权限被拒绝", Toast.LENGTH_SHORT).show());
                }
            }
        };
        Shizuku.addRequestPermissionResultListener(permissionListener);

        // Binder 就绪回调
        binderReceivedListener = () -> {
            if (!Shizuku.pingBinder()) return;
            if (Shizuku.checkSelfPermission() != PackageManager.PERMISSION_GRANTED) {
                Shizuku.requestPermission(SHIZUKU_PERMISSION_REQUEST_CODE);
            }
        };
        Shizuku.addBinderReceivedListenerSticky(binderReceivedListener);

        // Binder 断开回调
        binderDeadListener = () -> runOnUiThread(() ->
                Toast.makeText(this, "Shizuku 服务已断开", Toast.LENGTH_SHORT).show());
        Shizuku.addBinderDeadListener(binderDeadListener);
    }

    /**
     * 主动检测 Shizuku 状态：onResume 时调用。
     * binderReceivedListener 只在 Shizuku 已运行时才触发，若服务从未启动则无任何提示；
     * 此处用 pingBinder 主动探测，在开启滑动跟随但 Shizuku 未运行时给出提示。
     */
    private void checkShizukuStatus() {
        boolean swipeEnabled = "1".equals(prefs.getString("swipe_enabled", "0"));
        if (!swipeEnabled) return;  // 未开启滑动跟随，无需 Shizuku

        if (!Shizuku.pingBinder()) {
            Toast.makeText(this, "Shizuku 服务未运行，滑动跟随无法使用", Toast.LENGTH_LONG).show();
        } else if (Shizuku.checkSelfPermission() != PackageManager.PERMISSION_GRANTED) {
            Shizuku.requestPermission(SHIZUKU_PERMISSION_REQUEST_CODE);
        }
    }

    /**
     * 一键更新 APK：后台检查 GitHub Releases 最新版本。
     * 有新版则弹出对话框展示更新日志，点「更新」后用浏览器打开 APK 直接下载链接（走用户代理）。
     */
    private void checkForAppUpdate() {
        if (updating) {
            Toast.makeText(this, "正在更新中，请稍候…", Toast.LENGTH_SHORT).show();
            return;
        }
        updating = true;
        Toast.makeText(this, "正在检查更新…", Toast.LENGTH_SHORT).show();
        new Thread(() -> UpdateManager.checkLatest(this, new UpdateManager.CheckCallback() {
            @Override
            public void onNoUpdate() {
                updating = false;
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "已是最新版本", Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onUpdateAvailable(UpdateManager.LatestVersion version) {
                updating = false;
                runOnUiThread(() -> showUpdateDialog(version));
            }

            @Override
            public void onError(String message) {
                updating = false;
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "检查失败：" + message, Toast.LENGTH_LONG).show());
            }
        })).start();
    }

    /** 弹出更新对话框：展示版本号和更新日志，点「更新」用浏览器下载 APK */
    private void showUpdateDialog(UpdateManager.LatestVersion version) {
        new androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("发现新版本 " + version.versionName)
                .setMessage(version.changelog.isEmpty() ? "有新版本可用" : version.changelog)
                .setPositiveButton("更新", (dialog, which) -> openDownloadUrl(version.downloadUrl))
                .setNegativeButton("取消", null)
                .show();
    }

    /** 用浏览器打开 APK 直接下载链接（走用户代理） */
    private void openDownloadUrl(String downloadUrl) {
        try {
            Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(downloadUrl));
            startActivity(intent);
        } catch (Exception e) {
            Toast.makeText(this, "无法打开浏览器：" + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private String currentModelName() {
        return prefs.getString("model_name", ModelManager.DEFAULT_MODEL);
    }

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

    private boolean checkOverlayPermission() {
        return Settings.canDrawOverlays(this);
    }

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

    private void requestScreenCapture() {
        MediaProjectionManager mpm = (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        startActivityForResult(mpm.createScreenCaptureIntent(), REQUEST_CODE_CAPTURE);
    }

    private void startScreenCaptureService(int resultCode, Intent resultData) {
        Intent serviceIntent = new Intent(this, ScreenCaptureService.class);
        serviceIntent.putExtra("resultCode", resultCode);
        serviceIntent.putExtra("resultData", resultData);
        startForegroundService(serviceIntent);
        Toast.makeText(this, "服务已启动", Toast.LENGTH_SHORT).show();
        btnStart.setText(R.string.btn_stop);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (permissionListener != null) Shizuku.removeRequestPermissionResultListener(permissionListener);
        if (binderReceivedListener != null) Shizuku.removeBinderReceivedListener(binderReceivedListener);
        if (binderDeadListener != null) Shizuku.removeBinderDeadListener(binderDeadListener);
    }
}