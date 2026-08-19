package com.example.fplus;

import android.content.Intent;
import android.content.SharedPreferences;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.materialswitch.MaterialSwitch;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_OVERLAY_PERMISSION = 2001;
    private static final int REQUEST_CODE_CAPTURE = 2002;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        SharedPreferences prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);
        MaterialSwitch switchGpu = findViewById(R.id.switch_gpu);
        switchGpu.setChecked(prefs.getBoolean("use_gpu", false));
        switchGpu.setOnCheckedChangeListener((buttonView, isChecked) ->
                prefs.edit().putBoolean("use_gpu", isChecked).apply());

        Button btnStart = findViewById(R.id.btn_start);
        btnStart.setOnClickListener(v -> {
            if (checkOverlayPermission()) {
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
    }
}