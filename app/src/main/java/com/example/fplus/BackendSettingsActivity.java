package com.example.fplus;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.RadioGroup;

import androidx.appcompat.app.AppCompatActivity;

/** 推理后端选择界面 */
public class BackendSettingsActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_backend_settings);

        SharedPreferences prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);
        RadioGroup radioBackend = findViewById(R.id.radio_backend);
        String savedBackend = prefs.getString("backend", PoseEstimator.Backend.GPU.name());
        PoseEstimator.Backend backend;
        try {
            backend = PoseEstimator.Backend.valueOf(savedBackend);
        } catch (IllegalArgumentException e) {
            backend = PoseEstimator.Backend.GPU;
        }
        switch (backend) {
            case GPU:
                radioBackend.check(R.id.radio_gpu);
                break;
            case CPU:
                radioBackend.check(R.id.radio_cpu);
                break;
            case NNAPI:
            default:
                radioBackend.check(R.id.radio_npu);
                break;
        }
        radioBackend.setOnCheckedChangeListener((group, checkedId) -> {
            PoseEstimator.Backend b;
            if (checkedId == R.id.radio_gpu) {
                b = PoseEstimator.Backend.GPU;
            } else if (checkedId == R.id.radio_cpu) {
                b = PoseEstimator.Backend.CPU;
            } else {
                b = PoseEstimator.Backend.NNAPI;
            }
            prefs.edit().putString("backend", b.name()).apply();
        });
    }
}
