package com.example.fplus;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import java.util.List;

/** 模型选择界面：动态列出云端 + 本地所有模型 */
public class ModelSettingsActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_model_settings);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }
        toolbar.setNavigationOnClickListener(v -> finish());

        SharedPreferences prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);
        RadioGroup radioModel = findViewById(R.id.radio_model);

        // 后台拉取模型列表（含网络请求），回主线程动态填充
        new Thread(() -> {
            List<String> models = ModelManager.listAllModels(this);
            runOnUiThread(() -> populate(radioModel, models, prefs));
        }).start();
    }

    private void populate(RadioGroup radioModel, List<String> models, SharedPreferences prefs) {
        radioModel.removeAllViews();

        if (models.isEmpty()) {
            TextView empty = new TextView(this);
            empty.setText(R.string.model_empty);
            empty.setAlpha(0.7f);
            empty.setTextSize(14);
            empty.setPadding(0, dp(8), 0, 0);
            radioModel.addView(empty);
            return;
        }

        String saved = prefs.getString("model_name", ModelManager.DEFAULT_MODEL);

        for (String model : models) {
            RadioButton rb = new RadioButton(this);
            rb.setText(model);
            rb.setTag(model);
            rb.setId(View.generateViewId());
            // 增大选项间垂直间距，避免两个选项贴在一起
            RadioGroup.LayoutParams lp = new RadioGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            lp.setMargins(0, dp(12), 0, dp(12));
            rb.setLayoutParams(lp);
            radioModel.addView(rb);
            if (model.equals(saved)) {
                radioModel.check(rb.getId());
            }
        }

        radioModel.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == -1) return;
            View v = group.findViewById(checkedId);
            if (v != null && v.getTag() != null) {
                prefs.edit().putString("model_name", (String) v.getTag()).apply();
            }
        });
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
