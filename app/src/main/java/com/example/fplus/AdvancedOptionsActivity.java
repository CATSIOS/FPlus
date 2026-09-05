package com.example.fplus;

import android.content.SharedPreferences;
import android.graphics.Typeface;
import android.os.Bundle;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import java.util.ArrayList;
import java.util.List;

/**
 * 高级选项界面：调试详细参数 + 一键还原
 * 纯 Android 原生组件，无 Material 自定义样式
 * 参数通过 SharedPreferences 持久化，PoseEstimator/OverlayView 初始化时读取
 */
public class AdvancedOptionsActivity extends AppCompatActivity {

    private static final String PREFS_NAME = "fplus_settings";

    /** 参数项数据类 */
    private static class ParamItem {
        final String key;          // SharedPreferences 键
        final String defValue;     // 默认值（字符串形式，避免浮点/整数差异）
        final String label;        // 参数名
        final String desc;         // 简短说明
        final String group;        // 所属分组
        final double min;          // 允许最小值（含）
        final double max;          // 允许最大值（含）

        ParamItem(String key, String defValue, String label, String desc, String group,
                  double min, double max) {
            this.key = key;
            this.defValue = defValue;
            this.label = label;
            this.desc = desc;
            this.group = group;
            this.min = min;
            this.max = max;
        }
    }

    /** 分组顺序 */
    private static final String GROUP_DETECTION = "检测";
    private static final String GROUP_TRACKING = "跟踪";
    private static final String GROUP_PREDICT = "预测";
    private static final String GROUP_SCORE = "评分";
    private static final String GROUP_ROI = "ROI";
    private static final String GROUP_BRIGHT = "亮度增强";
    private static final String GROUP_OVERLAY = "Overlay 滤波";
    private static final String GROUP_DUAL = "双实例并发";

    /** 参数定义表：key 必须与 PoseEstimator/OverlayView 中读取逻辑一致 */
    private final List<ParamItem> params = new ArrayList<>();

    {
        params.add(new ParamItem("det_conf", "0.15", "置信度阈值",
                "YOLO 检测框置信度下限（0.05~0.5）", GROUP_DETECTION, 0.05, 0.5));
        params.add(new ParamItem("det_valid", "0.2", "有效检测置信度",
                "C-BIoU HIGH 检测阈值（0.1~0.5）", GROUP_DETECTION, 0.1, 0.5));
        params.add(new ParamItem("det_min_area", "0.01", "最小面积阈值",
                "过滤小 UI 元素（0.001~0.1）", GROUP_DETECTION, 0.001, 0.1));

        params.add(new ParamItem("track_iou", "0.2", "C-BIoU HIGH 阈值",
                "HIGH 阶段 buffered IoU 阈值（0~1）", GROUP_TRACKING, 0, 1));
        params.add(new ParamItem("track_buf_high", "0.3", "C-BIoU HIGH 缓冲",
                "HIGH 阶段 buffer 扩展比例（0~1）", GROUP_TRACKING, 0, 1));
        params.add(new ParamItem("track_iou_low", "0.3", "C-BIoU LOW 阈值",
                "LOW 救援 buffered IoU 阈值（0~1）", GROUP_TRACKING, 0, 1));
        params.add(new ParamItem("ocm_weight", "0.15", "OCM 方向权重",
                "方向不一致候选的最大惩罚，0 关闭（0~0.5）", GROUP_TRACKING, 0, 0.5));
        params.add(new ParamItem("ocm_speed_ref", "0.01", "OCM 速度参考",
                "速度≥此值达满权重，低速降权至半额（0.001~0.1）", GROUP_TRACKING, 0.001, 0.1));
        params.add(new ParamItem("track_buf_low", "0.5", "C-BIoU LOW 缓冲",
                "LOW 救援 buffer 扩展比例（0~1）", GROUP_TRACKING, 0, 1));
        params.add(new ParamItem("track_speed_ref", "0.25", "速度自适应参考",
                "HIGH buffer 放大触发速度（0.1~0.6 框宽/帧）", GROUP_TRACKING, 0.1, 0.6));
        params.add(new ParamItem("track_max_lost", "30", "最大跟踪丢失帧数",
                "锁定保持时长（5~60 帧）", GROUP_TRACKING, 5, 60));
        params.add(new ParamItem("track_takeover", "0.5", "接管置信度",
                "新目标即时接管阈值（0.3~0.9）", GROUP_TRACKING, 0.3, 0.9));
        params.add(new ParamItem("track_dist_weight", "0.25", "距离惩罚权重",
                "横向断连补分强度（0~0.5）", GROUP_TRACKING, 0, 0.5));
        params.add(new ParamItem("target_lock", "0", "目标锁定",
                "开启后锁定当前目标：丢失期间由新目标接管被禁止，直到该目标彻底消失才允许换锁", GROUP_TRACKING, 0, 1));
        params.add(new ParamItem("track_acc_threshold", "0.002", "加速度阈值",
                "CA 外推触发加速度（0~0.02）", GROUP_TRACKING, 0, 0.02));
        params.add(new ParamItem("anti_crosshair", "0", "抗准心误识别（实验）",
                "开镜瞄具/准心UI误绑定时启用：超小+中心+非人形比例才惩罚，不影响真人", GROUP_TRACKING, 0, 1));

        params.add(new ParamItem("lead_predict", "1", "绿框预判",
                "开启后绿框按目标速度提前外推抵消延迟；关闭则紧贴检测位置，观感更稳", GROUP_PREDICT, 0, 1));
        params.add(new ParamItem("pred_seconds", "0.083", "预测时长（秒）",
                "绿框提前量，抵消延迟（0~0.2）", GROUP_PREDICT, 0, 0.2));
        params.add(new ParamItem("coast_min_vel", "0.01", "Coast 速度阈值",
                "低于此速度不外推直接淡出（0.001~0.02）", GROUP_PREDICT, 0.001, 0.02));
        params.add(new ParamItem("coast_max_frames", "2", "Coast 最大帧数",
                "丢失外推持续上限（1~30 帧）", GROUP_PREDICT, 1, 30));

        params.add(new ParamItem("score_center_sigma", "0.2", "中心高斯标准差",
                "锁定范围（0.1~0.5）", GROUP_SCORE, 0.1, 0.5));
        params.add(new ParamItem("score_w", "0.3", "角度权重",
                "水平偏心惩罚权重（0~1）", GROUP_SCORE, 0, 1));

        params.add(new ParamItem("roi_scale", "0.7", "ROI 尺寸",
                "检测区域占比（0.3~1.0）", GROUP_ROI, 0.3, 1.0));

        params.add(new ParamItem("bright_auto", "1", "启用环境光优化",
                "根据画面亮度动态调整增益+暗部抬升（关则只使用固定增益）", GROUP_BRIGHT, 0, 1));
        params.add(new ParamItem("bright_gain", "1.3", "亮度增益",
                "对比度提升（0.5~3.0）", GROUP_BRIGHT, 0.5, 3.0));
        params.add(new ParamItem("bright_offset", "25", "亮度偏移",
                "暗景提亮（0~100）", GROUP_BRIGHT, 0, 100));

        params.add(new ParamItem("overlay_min_cutoff", "2.0", "1€ 最小截止频率",
                "静止平滑度（0.5~5）", GROUP_OVERLAY, 0.5, 5));
        params.add(new ParamItem("overlay_beta", "4.0", "1€ 速度系数",
                "快速响应度（0.1~20）", GROUP_OVERLAY, 0.1, 20));
        params.add(new ParamItem("overlay_max_jump", "0.4", "最大跳变阈值",
                "目标切换位移限制（0.1~1.0）", GROUP_OVERLAY, 0.1, 1.0));
        params.add(new ParamItem("overlay_size_smooth", "0.45", "尺寸平滑系数",
                "越小越稳（0~1）", GROUP_OVERLAY, 0, 1));
        params.add(new ParamItem("overlay_ease", "0.4", "ease 慢快慢",
                "二阶缓动保留比例（0~0.8）", GROUP_OVERLAY, 0, 0.8));

        params.add(new ParamItem("dual_infer", "0", "双实例并发",
                "开启第二路放大区推理，榨GPU算力", GROUP_DUAL, 0, 1));
        params.add(new ParamItem("dual_verify", "0", "双路交叉核验（实验）",
                "双路开时生效：放大区主路候选需第二路同位置也检出才采信，降权背景误识别", GROUP_DUAL, 0, 1));
        params.add(new ParamItem("dual_zoom", "0.5", "放大区比例",
                "第二路区域相对ROI（0.3~0.8，越小放大越大）", GROUP_DUAL, 0.3, 0.8));
    }

    private final List<EditText> editTexts = new ArrayList<>();
    private Switch dualSwitch;         // dual_infer 用开关控件，editTexts 对应槽位存 null 占位
    private Switch dualVerifySwitch;   // dual_verify 实验开关，editTexts 槽位存 null 占位
    private Switch antiCrosshairSwitch;// anti_crosshair 抗准心开关，editTexts 槽位存 null 占位
    private Switch brightAutoSwitch;   // bright_auto 环境光优化开关，editTexts 槽位存 null 占位
    private Switch targetLockSwitch;   // target_lock 目标锁定开关，editTexts 槽位存 null 占位
    private Switch leadPredictSwitch;  // lead_predict 绿框预判开关，editTexts 槽位存 null 占位
    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_advanced_options);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }
        toolbar.setNavigationOnClickListener(v -> finish());

        prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        LinearLayout container = findViewById(R.id.params_container);
        buildParamRows(container);

        Button btnReset = findViewById(R.id.btn_reset);
        btnReset.setOnClickListener(v -> resetToDefault());

        Button btnSave = findViewById(R.id.btn_save);
        btnSave.setOnClickListener(v -> saveAndExit());
    }

    /** 动态生成参数行：分组标题 + 标签 + 说明 + EditText */
    private void buildParamRows(LinearLayout container) {
        String lastGroup = "";
        for (int i = 0; i < params.size(); i++) {
            ParamItem p = params.get(i);

            // 分组标题
            if (!p.group.equals(lastGroup)) {
                // 分组间分割线（非首组）
                if (i != 0) {
                    View divider = new View(this);
                    divider.setBackgroundColor(0x1AFFFFFF);
                    LinearLayout.LayoutParams divLp = new LinearLayout.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT, dp(1));
                    divLp.topMargin = dp(16);
                    divLp.bottomMargin = dp(14);
                    container.addView(divider, divLp);
                }
                TextView groupTitle = new TextView(this);
                groupTitle.setText(p.group);
                groupTitle.setTextSize(13);
                groupTitle.setTextColor(getColor(R.color.text_secondary));   // text_secondary：组标题弱化，区分条目文字层级
                groupTitle.setTypeface(null, Typeface.BOLD);
                groupTitle.setPadding(dp(4), dp(2), 0, dp(2));
                LinearLayout.LayoutParams groupLp = new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT);
                groupLp.topMargin = i == 0 ? 0 : dp(4);
                groupLp.bottomMargin = dp(6);
                container.addView(groupTitle, groupLp);
                lastGroup = p.group;
            }

            // 参数行：左侧标签+说明，右侧 EditText
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(dp(4), dp(10), dp(4), dp(10));   // 行内上下留白，条目间不贴太紧

            // 左侧文本
            LinearLayout textCol = new LinearLayout(this);
            textCol.setOrientation(LinearLayout.VERTICAL);
            textCol.setLayoutParams(new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

            TextView label = new TextView(this);
            label.setText(p.label);
            label.setTextSize(14);
            textCol.addView(label);

            TextView desc = new TextView(this);
            desc.setText(p.desc);
            desc.setTextSize(11);
            desc.setAlpha(0.6f);
            textCol.addView(desc);

            row.addView(textCol);

            // 右侧控件：dual_infer / dual_verify / bright_auto 用开关，其余用数值输入框
            if ("dual_infer".equals(p.key)) {
                Switch sw = new Switch(this);
                sw.setChecked("1".equals(prefs.getString(p.key, p.defValue)));
                LinearLayout.LayoutParams swLp = new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT);
                swLp.setMarginStart(dp(12));
                row.addView(sw, swLp);
                dualSwitch = sw;
                editTexts.add(null);  // 占位保持与 params index 对齐
            } else if ("dual_verify".equals(p.key)) {
                Switch sw = new Switch(this);
                sw.setChecked("1".equals(prefs.getString(p.key, p.defValue)));
                LinearLayout.LayoutParams swLp = new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT);
                swLp.setMarginStart(dp(12));
                row.addView(sw, swLp);
                dualVerifySwitch = sw;
                editTexts.add(null);
            } else if ("anti_crosshair".equals(p.key)) {
                Switch sw = new Switch(this);
                sw.setChecked("1".equals(prefs.getString(p.key, p.defValue)));
                LinearLayout.LayoutParams swLp = new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT);
                swLp.setMarginStart(dp(12));
                row.addView(sw, swLp);
                antiCrosshairSwitch = sw;
                editTexts.add(null);
            } else if ("bright_auto".equals(p.key)) {
                Switch sw = new Switch(this);
                sw.setChecked(!"0".equals(prefs.getString(p.key, p.defValue)));
                LinearLayout.LayoutParams swLp = new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT);
                swLp.setMarginStart(dp(12));
                row.addView(sw, swLp);
                brightAutoSwitch = sw;
                editTexts.add(null);
            } else if ("target_lock".equals(p.key)) {
                Switch sw = new Switch(this);
                sw.setChecked("1".equals(prefs.getString(p.key, p.defValue)));
                LinearLayout.LayoutParams swLp = new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT);
                swLp.setMarginStart(dp(12));
                row.addView(sw, swLp);
                targetLockSwitch = sw;
                editTexts.add(null);
            } else if ("lead_predict".equals(p.key)) {
                // 默认打开：仅显式存 "0" 才关闭（与 bright_auto 同风格）
                Switch sw = new Switch(this);
                sw.setChecked(!"0".equals(prefs.getString(p.key, p.defValue)));
                LinearLayout.LayoutParams swLp = new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT);
                swLp.setMarginStart(dp(12));
                row.addView(sw, swLp);
                leadPredictSwitch = sw;
                editTexts.add(null);
            } else {
                EditText et = new EditText(this);
                et.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL | InputType.TYPE_NUMBER_FLAG_SIGNED);
                et.setMinEms(5);
                et.setMaxEms(6);
                et.setGravity(Gravity.CENTER);
                // 读取当前值：prefs 无值则用默认
                String cur = prefs.getString(p.key, p.defValue);
                et.setText(cur);
                LinearLayout.LayoutParams etLp = new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT);
                etLp.setMarginStart(dp(12));
                row.addView(et, etLp);
                editTexts.add(et);
            }

            LinearLayout.LayoutParams rowLp = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT);
            rowLp.bottomMargin = dp(8);
            container.addView(row, rowLp);
        }
    }

    /** 保存所有参数到 SharedPreferences，超范围或非法值回退默认并提示 */
    private void saveAndExit() {
        SharedPreferences.Editor editor = prefs.edit();
        List<String> resetNames = new ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            ParamItem p = params.get(i);
            if ("dual_infer".equals(p.key)) {
                editor.putString(p.key, dualSwitch.isChecked() ? "1" : "0");
                continue;
            }
            if ("dual_verify".equals(p.key)) {
                editor.putString(p.key, dualVerifySwitch.isChecked() ? "1" : "0");
                continue;
            }
            if ("anti_crosshair".equals(p.key)) {
                editor.putString(p.key, antiCrosshairSwitch.isChecked() ? "1" : "0");
                continue;
            }
            if ("bright_auto".equals(p.key)) {
                editor.putString(p.key, brightAutoSwitch.isChecked() ? "1" : "0");
                continue;
            }
            if ("target_lock".equals(p.key)) {
                editor.putString(p.key, targetLockSwitch.isChecked() ? "1" : "0");
                continue;
            }
            String val = editTexts.get(i).getText().toString().trim();
            String saved = val;
            if (val.isEmpty()) {
                saved = p.defValue;
            } else {
                // 校验：必须能解析为数值且落在 [min,max]
                try {
                    double d = Double.parseDouble(val);
                    if (d < p.min || d > p.max) {
                        saved = p.defValue;
                        resetNames.add(p.label);
                    }
                } catch (NumberFormatException e) {
                    saved = p.defValue;
                    resetNames.add(p.label);
                }
            }
            editor.putString(p.key, saved);
        }
        editor.apply();
        if (resetNames.isEmpty()) {
            Toast.makeText(this, "已保存，下次启动服务生效", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "以下参数超范围已回退默认: " + String.join(", ", resetNames),
                    Toast.LENGTH_LONG).show();
        }
        finish();
    }

    /** 一键还原：清除所有高级参数，恢复默认 */
    private void resetToDefault() {
        SharedPreferences.Editor editor = prefs.edit();
        for (ParamItem p : params) {
            editor.remove(p.key);
        }
        editor.apply();
        // 刷新 UI 显示默认值
        for (int i = 0; i < params.size(); i++) {
            ParamItem p = params.get(i);
            if ("dual_infer".equals(p.key)) {
                dualSwitch.setChecked("1".equals(p.defValue));
                continue;
            }
            if ("dual_verify".equals(p.key)) {
                dualVerifySwitch.setChecked("1".equals(p.defValue));
                continue;
            }
            if ("anti_crosshair".equals(p.key)) {
                antiCrosshairSwitch.setChecked("1".equals(p.defValue));
                continue;
            }
            if ("bright_auto".equals(p.key)) {
                brightAutoSwitch.setChecked(!"0".equals(p.defValue));
                continue;
            }
            if ("target_lock".equals(p.key)) {
                targetLockSwitch.setChecked("1".equals(p.defValue));
                continue;
            }
            if ("lead_predict".equals(p.key)) {
                leadPredictSwitch.setChecked(!"0".equals(p.defValue));
                continue;
            }
            editTexts.get(i).setText(p.defValue);
        }
        Toast.makeText(this, "已恢复默认值", Toast.LENGTH_SHORT).show();
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }
}
