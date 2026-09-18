package com.example.fplus;

import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
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

    private long downloadId = -1;               // 当前下载任务的 ID
    private boolean updating = false;           // 是否正在检查/下载更新（防重复点击）
    private BroadcastReceiver downloadReceiver;

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
        registerDownloadReceiver();

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
     * 一键更新 APK：后台检查 GitHub Releases 最新版本，有新版则交给系统 DownloadManager 下载。
     * DownloadManager 支持后台下载 + 通知栏进度，切后台/退出界面都不中断。
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
            public void onUpdateAvailable(String newVersion, String downloadUrl) {
                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this,
                            "发现新版本 " + newVersion + "，开始下载…", Toast.LENGTH_SHORT).show();
                    startDownload(newVersion, downloadUrl);
                });
            }

            @Override
            public void onError(String message) {
                updating = false;
                runOnUiThread(() -> Toast.makeText(MainActivity.this,
                        "检查失败：" + message, Toast.LENGTH_LONG).show());
            }
        })).start();
    }

    /** 用系统 DownloadManager 发起下载，下载完成通过广播回调触发安装 */
    private void startDownload(String version, String downloadUrl) {
        try {
            DownloadManager.Request request = new DownloadManager.Request(Uri.parse(downloadUrl));
            request.setTitle("FPlus " + version);
            request.setDescription("正在下载更新…");
            request.setNotificationVisibility(
                    DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
            // 下载到公共 Download 目录，文件名带版本号
            String fileName = "FPlus_" + version + ".apk";
            request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, fileName);
            // GitHub 下载要求带 User-Agent
            request.addRequestHeader("User-Agent", "FPlus-Updater");
            request.setMimeType("application/vnd.android.package-archive");

            DownloadManager dm = (DownloadManager) getSystemService(DOWNLOAD_SERVICE);
            downloadId = dm.enqueue(request);
        } catch (Exception e) {
            updating = false;
            Toast.makeText(this, "无法发起下载：" + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /** 注册下载完成广播，收到后查询状态并触发安装 */
    private void registerDownloadReceiver() {
        if (downloadReceiver != null) return;
        downloadReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                if (downloadId < 0) return;
                long id = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1);
                if (id != downloadId) return;

                DownloadManager dm = (DownloadManager) getSystemService(DOWNLOAD_SERVICE);
                DownloadManager.Query query = new DownloadManager.Query().setFilterById(downloadId);
                try (Cursor cursor = dm.query(query)) {
                    if (cursor == null || !cursor.moveToFirst()) return;
                    int status = cursor.getInt(cursor.getColumnIndex(DownloadManager.COLUMN_STATUS));
                    if (status == DownloadManager.STATUS_SUCCESSFUL) {
                        updating = false;
                        installApk(dm.getUriForDownloadedFile(downloadId));
                    } else if (status == DownloadManager.STATUS_FAILED) {
                        updating = false;
                        Toast.makeText(MainActivity.this,
                                "下载失败，请重试", Toast.LENGTH_LONG).show();
                    }
                } catch (Exception e) {
                    updating = false;
                }
            }
        };
        registerReceiver(downloadReceiver,
                new IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE),
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                        ? Context.RECEIVER_EXPORTED
                        : 0);
    }

    /** 调起系统安装器安装 content:// URI 的 APK（DownloadManager 下载完成后的产物） */
    private void installApk(Uri apkUri) {
        Intent intent = new Intent(Intent.ACTION_VIEW);
        intent.setDataAndType(apkUri, "application/vnd.android.package-archive");
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        try {
            startActivity(intent);
        } catch (Exception e) {
            Toast.makeText(this, "无法打开安装器：" + e.getMessage(), Toast.LENGTH_LONG).show();
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
        if (downloadReceiver != null) {
            unregisterReceiver(downloadReceiver);
            downloadReceiver = null;
        }
    }
}