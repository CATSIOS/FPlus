package com.example.fplus;

import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.google.android.material.button.MaterialButton;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 模型管理界面：动态列出所有模型，支持下载 / 删除 */
public class ModelManagerActivity extends AppCompatActivity {

    private LinearLayout container;
    private final Set<String> downloadingModels = new HashSet<>();
    private final Map<String, TextView> statusViews = new HashMap<>();
    private final Map<String, MaterialButton> actionButtons = new HashMap<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_model_manager);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }
        toolbar.setNavigationOnClickListener(v -> finish());

        container = findViewById(R.id.model_list_container);
        loadModels();
    }

    private void loadModels() {
        new Thread(() -> {
            List<String> models = ModelManager.listAllModels(this);
            runOnUiThread(() -> render(models));
        }).start();
    }

    private void render(List<String> models) {
        container.removeAllViews();
        statusViews.clear();
        actionButtons.clear();

        if (models.isEmpty()) {
            TextView empty = new TextView(this);
            empty.setText("未获取到模型（请检查网络）");
            empty.setAlpha(0.7f);
            empty.setTextSize(14);
            empty.setPadding(dp(4), dp(16), dp(4), dp(16));
            container.addView(empty);
            return;
        }

        for (String model : models) {
            container.addView(buildRow(model));
        }
    }

    private View buildRow(String model) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(android.view.Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(8), 0, dp(8));

        TextView name = new TextView(this);
        name.setText(model);
        name.setTextSize(14);
        LinearLayout.LayoutParams nameLp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        name.setLayoutParams(nameLp);

        TextView status = new TextView(this);
        status.setTextSize(12);
        status.setAlpha(0.7f);
        LinearLayout.LayoutParams statusLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        statusLp.setMarginEnd(dp(8));
        status.setLayoutParams(statusLp);

        MaterialButton button = new MaterialButton(this);
        button.setMinWidth(dp(88));
        button.setMinHeight(0);

        row.addView(name);
        row.addView(status);
        row.addView(button);

        statusViews.put(model, status);
        actionButtons.put(model, button);

        button.setOnClickListener(v -> {
            if (ModelManager.isDownloaded(this, model)) {
                ModelManager.getLocalFile(this, model).delete();
                refreshModelStatus(model);
                Toast.makeText(this, "模型已删除", Toast.LENGTH_SHORT).show();
            } else {
                downloadModel(model);
            }
        });

        refreshModelStatus(model);
        return row;
    }

    private void refreshModelStatus(String model) {
        TextView status = statusViews.get(model);
        MaterialButton button = actionButtons.get(model);
        if (status == null || button == null) return;
        if (downloadingModels.contains(model)) {
            status.setText(R.string.model_status_downloading);
            button.setEnabled(false);
        } else if (ModelManager.isDownloaded(this, model)) {
            status.setText(R.string.model_status_downloaded);
            button.setText(R.string.model_action_delete);
            button.setEnabled(true);
        } else {
            status.setText(R.string.model_status_not_downloaded);
            button.setText(R.string.model_action_download);
            button.setEnabled(true);
        }
    }

    private void downloadModel(String model) {
        if (downloadingModels.contains(model)) {
            return;
        }
        downloadingModels.add(model);
        TextView status = statusViews.get(model);
        MaterialButton button = actionButtons.get(model);
        if (status != null) status.setText(R.string.model_status_downloading);
        if (button != null) button.setEnabled(false);

        ModelManager.download(this, model,
                (downloaded, total) -> runOnUiThread(() -> {
                    TextView s = statusViews.get(model);
                    if (s != null && total > 0) {
                        int pct = (int) (downloaded * 100 / total);
                        s.setText("下载中 " + pct + "%");
                    }
                }),
                file -> runOnUiThread(() -> {
                    downloadingModels.remove(model);
                    refreshModelStatus(model);
                    Toast.makeText(this, "模型下载完成", Toast.LENGTH_SHORT).show();
                }),
                err -> runOnUiThread(() -> {
                    downloadingModels.remove(model);
                    refreshModelStatus(model);
                    Toast.makeText(this, "下载失败：" + err, Toast.LENGTH_SHORT).show();
                }));
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
