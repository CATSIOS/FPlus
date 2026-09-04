package com.example.fplus;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * 模型按需下载：模型文件托管在 Cloudflare Pages，首次使用/切换模型时下载到应用私有目录，
 * 之后直接从本地加载，减小 APK 体积并支持独立更新模型。
 * <p>
 * 模型列表动态获取：云端清单 {@code models.json} 列出所有可用模型，本地目录扫描已下载模型，
 * 两者合并（云端在前、本地补充），模型名直接使用文件名。
 */
public class ModelManager {

    private static final String TAG = "ModelManager";
    private static final String BASE_URL = "https://fplus-models.pages.dev/";
    private static final String MANIFEST = "models.json";

    /** prefs 中 model_name 为空时使用的默认模型（须与云端 models.json 中的文件名一致） */
    public static final String DEFAULT_MODEL = "DeltaForce_640.tflite";

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
        try {
            return BASE_URL + URLEncoder.encode(modelName, "UTF-8").replace("+", "%20");
        } catch (Exception e) {
            return BASE_URL + modelName;
        }
    }

    /**
     * 拉取云端模型清单（models.json），返回所有云端可用模型文件名。
     * 同步网络请求，须在后台线程调用；失败返回 null。
     */
    public static List<String> fetchCloudModels() {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(BASE_URL + MANIFEST);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(10000);
            conn.setRequestMethod("GET");
            conn.connect();

            if (conn.getResponseCode() != HttpURLConnection.HTTP_OK) {
                return null;
            }
            try (InputStream in = conn.getInputStream();
                 ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                byte[] buf = new byte[4096];
                int n;
                while ((n = in.read(buf)) != -1) {
                    baos.write(buf, 0, n);
                }
                return parseManifest(baos.toString("UTF-8"));
            }
        } catch (Exception e) {
            Log.e(TAG, "拉取云端模型清单失败", e);
            return null;
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }
    }

    /** 解析 models.json（字符串数组），过滤出 .tflite 文件名 */
    private static List<String> parseManifest(String json) {
        List<String> list = new ArrayList<>();
        try {
            JSONArray arr = new JSONArray(json);
            for (int i = 0; i < arr.length(); i++) {
                String name = arr.getString(i);
                if (name != null && name.endsWith(".tflite")) {
                    list.add(name);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "models.json 解析失败", e);
        }
        return list;
    }

    /** 扫描本地已下载的模型文件名（.tflite，非空文件） */
    public static List<String> listLocalModels(Context context) {
        List<String> list = new ArrayList<>();
        File dir = new File(context.getFilesDir(), "models");
        if (dir.exists() && dir.isDirectory()) {
            File[] files = dir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.getName().endsWith(".tflite") && f.length() > 0) {
                        list.add(f.getName());
                    }
                }
            }
        }
        return list;
    }

    /**
     * 合并云端 + 本地模型列表（去重，云端在前、本地补充）。
     * 包含网络请求，须在后台线程调用。
     */
    public static List<String> listAllModels(Context context) {
        LinkedHashSet<String> set = new LinkedHashSet<>();
        List<String> cloud = fetchCloudModels();
        if (cloud != null) {
            set.addAll(cloud);
        }
        set.addAll(listLocalModels(context));
        return new ArrayList<>(set);
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
