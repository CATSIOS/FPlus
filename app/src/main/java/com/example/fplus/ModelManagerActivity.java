package com.example.fplus;

import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.HashSet;
import java.util.Set;

/** 模型管理界面：下载 / 删除模型 */
public class ModelManagerActivity extends AppCompatActivity {

    private static final String MODEL_320_OPT = "sunxds_0.8.0_320_opt.tflite";
    private static final String MODEL_DELTA = "deltaforce_640.tflite";
    private final Set<String> downloadingModels = new HashSet<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_model_manager);

        for (String model : new String[]{MODEL_DELTA, MODEL_320_OPT}) {
            actionButtonFor(model).setOnClickListener(v -> {
                if (ModelManager.isDownloaded(this, model)) {
                    ModelManager.getLocalFile(this, model).delete();
                    refreshModelStatus(model);
                    Toast.makeText(this, "模型已删除", Toast.LENGTH_SHORT).show();
                } else {
                    downloadModel(model);
                }
            });
            refreshModelStatus(model);
        }
    }

    private TextView statusViewFor(String model) {
        return MODEL_320_OPT.equals(model)
                ? findViewById(R.id.status_320_opt)
                : findViewById(R.id.status_delta);
    }

    private Button actionButtonFor(String model) {
        return MODEL_320_OPT.equals(model)
                ? findViewById(R.id.btn_320_opt)
                : findViewById(R.id.btn_delta);
    }

    private void refreshModelStatus(String model) {
        TextView status = statusViewFor(model);
        Button button = actionButtonFor(model);
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
        TextView status = statusViewFor(model);
        Button button = actionButtonFor(model);
        status.setText(R.string.model_status_downloading);
        button.setEnabled(false);

        ModelManager.download(this, model,
                (downloaded, total) -> runOnUiThread(() -> {
                    if (total > 0) {
                        int pct = (int) (downloaded * 100 / total);
                        status.setText("下载中 " + pct + "%");
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
}
