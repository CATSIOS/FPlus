package com.example.fplus;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * 模型按需下载：模型文件托管在 Cloudflare Pages，首次使用/切换模型时下载到应用私有目录，
 * 之后直接从本地加载，减小 APK 体积并支持独立更新模型。
 */
public class ModelManager {

    private static final String TAG = "ModelManager";
    private static final String BASE_URL = "https://fplus-models.pages.dev/";

    public interface ProgressListener {
        void onProgress(long downloaded, long total);
    }

    public interface SuccessListener {
        void onSuccess(File file);
    }

    public interface ErrorListener {
        void onError(String message);
    }

    private ModelManager() {
    }

    /** 模型在应用私有目录中的本地文件路径 */
    public static File getLocalFile(Context context, String modelName) {
        return new File(new File(context.getFilesDir(), "models"), modelName);
    }

    public static boolean isDownloaded(Context context, String modelName) {
        File f = getLocalFile(context, modelName);
        return f.exists() && f.length() > 0;
    }

    public static String getUrl(String modelName) {
        return BASE_URL + modelName;
    }

    /**
     * 后台下载模型到本地，完成后回调。临时文件 .tmp 下载完成再 rename，避免半截文件被加载。
     */
    public static void download(Context context, String modelName,
                                ProgressListener progress, SuccessListener success, ErrorListener error) {
        File dir = new File(context.getFilesDir(), "models");
        if (!dir.exists() && !dir.mkdirs()) {
            error.onError("无法创建模型目录");
            return;
        }
        File target = getLocalFile(context, modelName);
        File tmp = new File(dir, modelName + ".tmp");

        new Thread(() -> {
            HttpURLConnection conn = null;
            try {
                URL url = new URL(getUrl(modelName));
                conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(30000);
                conn.setRequestMethod("GET");
                conn.connect();

                int code = conn.getResponseCode();
                if (code != HttpURLConnection.HTTP_OK) {
                    error.onError("HTTP " + code);
                    return;
                }

                long total = conn.getContentLength();
                try (InputStream in = conn.getInputStream();
                     FileOutputStream out = new FileOutputStream(tmp)) {
                    byte[] buf = new byte[64 * 1024];
                    long downloaded = 0;
                    int n;
                    while ((n = in.read(buf)) != -1) {
                        out.write(buf, 0, n);
                        downloaded += n;
                        if (progress != null) {
                            progress.onProgress(downloaded, total);
                        }
                    }
                    out.flush();
                }

                if (tmp.length() > 0) {
                    if (target.exists()) {
                        target.delete();
                    }
                    tmp.renameTo(target);
                    success.onSuccess(target);
                } else {
                    error.onError("下载内容为空");
                }
            } catch (Exception e) {
                Log.e(TAG, "模型下载失败: " + modelName, e);
                if (tmp.exists()) {
                    tmp.delete();
                }
                error.onError(e.getMessage());
            } finally {
                if (conn != null) {
                    conn.disconnect();
                }
            }
        }).start();
    }
}
