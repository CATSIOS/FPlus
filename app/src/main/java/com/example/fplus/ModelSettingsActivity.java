package com.example.fplus;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.RadioGroup;

import androidx.appcompat.app.AppCompatActivity;

/** 模型选择界面 */
public class ModelSettingsActivity extends AppCompatActivity {

    private static final String MODEL_320_OPT = "sunxds_0.8.0_320_opt.tflite";
    private static final String MODEL_DELTA = "deltaforce_640.tflite";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_model_settings);

        SharedPreferences prefs = getSharedPreferences("fplus_settings", MODE_PRIVATE);
        RadioGroup radioModel = findViewById(R.id.radio_model);
        String savedModel = prefs.getString("model_name", MODEL_DELTA);
        radioModel.check(MODEL_DELTA.equals(savedModel) ? R.id.radio_model_delta : R.id.radio_model_320_opt);
        radioModel.setOnCheckedChangeListener((group, checkedId) ->
                prefs.edit().putString("model_name",
                        checkedId == R.id.radio_model_delta ? MODEL_DELTA : MODEL_320_OPT).apply());
    }
}
