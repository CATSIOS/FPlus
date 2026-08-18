package com.example.fplus;

import android.app.Activity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;
import android.util.Log;

/**
 * 透明 Activity，用于请求屏幕捕获权限并回调给 ScreenCaptureService
 */
public class CapturePermissionActivity extends Activity {

    private static final String TAG = "CapturePermissionActivity";
    private static final int REQUEST_CODE_CAPTURE = 1001;

    private MediaProjectionManager mediaProjectionManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // 设置透明，不显示界面
        setContentView(android.R.layout.activity_list_item);

        mediaProjectionManager = (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);

        Intent captureIntent = getIntent().getParcelableExtra("captureIntent");
        if (captureIntent == null) {
            Log.e(TAG, "captureIntent is null");
            finish();
            return;
        }

        // 直接启动授权请求
        startActivityForResult(captureIntent, REQUEST_CODE_CAPTURE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE_CAPTURE) {
            boolean granted = resultCode == RESULT_OK;
            // 将结果传递给 ScreenCaptureService
            ScreenCaptureService service = ScreenCaptureService.getInstance();
            if (service != null) {
                service.onCapturePermissionResult(granted, data);
            } else {
                Log.e(TAG, "ScreenCaptureService instance is null");
            }
            finish();
        }
    }
}