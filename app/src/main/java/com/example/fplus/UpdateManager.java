package com.example.fplus;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * APK 自更新：从 GitHub Releases 检查最新版本。
 * 参照 AndroidEasterEggs 的做法：检查到新版本后弹出对话框展示更新日志，
 * 用户点击「更新」后用浏览器打开 APK 的直接下载链接（走用户代理）。
 */
public class UpdateManager {

    private static final String TAG = "UpdateManager";
    private static final String API_URL =
            "https://api.github.com/repos/CATSIOS/FPlus/releases/latest";

    /** 最新版本信息 */
    public static class LatestVersion {
        public final String versionName;
        public final String changelog;
        public final String downloadUrl;

        public LatestVersion(String versionName, String changelog, String downloadUrl) {
            this.versionName = versionName;
            this.changelog = changelog;
            this.downloadUrl = downloadUrl;
        }
    }

    /** 检查结果回调（后台线程调用） */
    public interface CheckCallback {
        void onNoUpdate();
        void onUpdateAvailable(LatestVersion version);
        void onError(String message);
    }

    private UpdateManager() {
    }

    /** 本地已安装版本名，如 "1.3.5"；失败返回 "0" */
    public static String localVersionName(Context context) {
        try {
            PackageInfo pi = context.getPackageManager()
                    .getPackageInfo(context.getPackageName(), 0);
            return pi.versionName;
        } catch (PackageManager.NameNotFoundException e) {
            return "0";
        }
    }

    /**
     * 检查最新版本（同步网络请求，须在后台线程调用）。
     * 对比 GitHub 最新 tag 与本地版本，仅当远程版本号更大时回调 onUpdateAvailable。
     */
    public static void checkLatest(Context context, CheckCallback callback) {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(API_URL);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(10000);
            conn.setRequestMethod("GET");
            // GitHub API 要求带 User-Agent，否则返回 403
            conn.setRequestProperty("User-Agent", "FPlus-Updater");
            conn.setRequestProperty("Accept", "application/vnd.github+json");
            conn.connect();

            int code = conn.getResponseCode();
            if (code != HttpURLConnection.HTTP_OK) {
                callback.onError("GitHub API HTTP " + code);
                return;
            }

            StringBuilder sb = new StringBuilder();
            try (InputStream in = conn.getInputStream()) {
                byte[] buf = new byte[4096];
                int n;
                while ((n = in.read(buf)) != -1) {
                    sb.append(new String(buf, 0, n, "UTF-8"));
                }
            }

            JSONObject release = new JSONObject(sb.toString());
            String tag = release.optString("tag_name", "");
            if (tag.isEmpty()) {
                callback.onError("release 缺少 tag_name");
                return;
            }
            String changelog = release.optString("body", "");

            // 找第一个 .apk 资产
            String downloadUrl = null;
            JSONArray assets = release.optJSONArray("assets");
            if (assets != null) {
                for (int i = 0; i < assets.length(); i++) {
                    JSONObject asset = assets.optJSONObject(i);
                    if (asset == null) continue;
                    String name = asset.optString("name", "");
                    if (name.toLowerCase().endsWith(".apk")) {
                        downloadUrl = asset.optString("browser_download_url", "");
                        break;
                    }
                }
            }
            if (downloadUrl == null || downloadUrl.isEmpty()) {
                callback.onError("release 中未找到 APK 资产");
                return;
            }

            String local = localVersionName(context);
            if (compareVersion(tag, local) > 0) {
                callback.onUpdateAvailable(new LatestVersion(tag, changelog, downloadUrl));
            } else {
                callback.onNoUpdate();
            }
        } catch (Exception e) {
            Log.e(TAG, "检查更新失败", e);
            callback.onError(e.getMessage());
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    /**
     * 逐段数字比较版本号（兼容 "1.3.6"、"v1.3.6"、"1.3.6-beta"）。
     * 返回 >0 表示 a 比 b 新，<0 表示 a 旧，0 相等。
     */
    private static int compareVersion(String a, String b) {
        String[] pa = normalize(a).split("\\.");
        String[] pb = normalize(b).split("\\.");
        int len = Math.max(pa.length, pb.length);
        for (int i = 0; i < len; i++) {
            int va = i < pa.length ? parsePart(pa[i]) : 0;
            int vb = i < pb.length ? parsePart(pb[i]) : 0;
            if (va != vb) return va - vb;
        }
        return 0;
    }

    /** 去掉前导 v/V 和非数字段，仅保留数字部分做对比 */
    private static String normalize(String v) {
        if (v == null) return "0";
        String s = v.trim();
        if (s.startsWith("v") || s.startsWith("V")) {
            s = s.substring(1);
        }
        // 去掉预发布后缀（如 -beta），只比主版本号
        int dash = s.indexOf('-');
        if (dash >= 0) s = s.substring(0, dash);
        return s;
    }

    /** 解析单个版本段，非法返回 0 */
    private static int parsePart(String s) {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return 0;
        }
    }
}
