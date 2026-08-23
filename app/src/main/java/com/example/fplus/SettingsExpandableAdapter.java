package com.example.fplus;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AbsListView;
import android.widget.BaseExpandableListAdapter;
import android.widget.RadioGroup;
import android.widget.TextView;

/**
 * 原生 ExpandableListView 设置页风格 Adapter（Android Settings 样式）。
 * <p>
 * 修正 v5.8 四处闪退根因：
 *   1) groupIndicator 用 arrow_down_float（API29+独占资源）→ 系统默认 indicator；
 *   2) 构造预 inflate 时 parent=null → getChildView 首次调用时补
 *      AbsListView.LayoutParams(MATCH_PARENT, WRAP_CONTENT)，保证测量正确；
 *   3) getChildView 返回已缓存 view 时若 view.getParent() != null 会触发
 *      IllegalStateException("child already has a parent") → 返回前先
 *      从旧父容器 removeView()，允许 ListView 重绘 / 反复展开折叠重复使用；
 *   4) 改为"延迟加载"会让 MainActivity.onCreate 里 getModelRadioGroup()
 *      返回 null 引发 NPE → 保留构造预 inflate，保证取引用非空。
 */
public class SettingsExpandableAdapter extends BaseExpandableListAdapter {

    private final Context mContext;
    private final LayoutInflater mInflater;
    private final String[] mGroups;

    // 构造预加载（保证取引用非空，LayoutParams 在 getChildView 里补）
    private final View mChildModel;
    private final View mChildBackend;

    public SettingsExpandableAdapter(Context context, String[] groups) {
        mContext = context;
        mInflater = LayoutInflater.from(context);
        mGroups = groups != null ? groups : new String[]{"模型", "推理后端"};

        // parent=null 只为能在构造时 inflate；LayoutParams 在 getChildView 补
        mChildModel = mInflater.inflate(R.layout.settings_child_model, null, false);
        mChildBackend = mInflater.inflate(R.layout.settings_child_backend, null, false);
    }

    public RadioGroup getModelRadioGroup() {
        return (RadioGroup) mChildModel.findViewById(R.id.radio_model);
    }

    public RadioGroup getBackendRadioGroup() {
        return (RadioGroup) mChildBackend.findViewById(R.id.radio_backend);
    }

    @Override public int getGroupCount() { return mGroups.length; }
    @Override public int getChildrenCount(int groupPosition) { return 1; }
    @Override public Object getGroup(int groupPosition) { return mGroups[groupPosition]; }
    @Override public Object getChild(int groupPosition, int childPosition) { return groupPosition; }
    @Override public long getGroupId(int groupPosition) { return groupPosition; }
    @Override public long getChildId(int groupPosition, int childPosition) { return groupPosition; }
    @Override public boolean hasStableIds() { return true; }

    @Override
    public View getGroupView(int groupPosition, boolean isExpanded,
                             View convertView, ViewGroup parent) {
        if (convertView == null) {
            convertView = mInflater.inflate(android.R.layout.simple_expandable_list_item_1,
                    parent, false);
        }
        TextView tv = (TextView) convertView.findViewById(android.R.id.text1);
        if (tv != null) tv.setText(mGroups[groupPosition]);
        return convertView;
    }

    @Override
    public View getChildView(int groupPosition, int childPosition,
                             boolean isLastChild, View convertView, ViewGroup parent) {
        View v = groupPosition == 0 ? mChildModel : mChildBackend;

        // 修复构造时 inflate(parent=null) 导致 LayoutParams 缺失
        ViewGroup.LayoutParams lp = v.getLayoutParams();
        if (!(lp instanceof AbsListView.LayoutParams)) {
            v.setLayoutParams(new AbsListView.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT));
        }

        // 防止 ListView 重绘 / 折叠再展开时重复 addView 崩 IllegalStateException
        if (v.getParent() != null && v.getParent() instanceof ViewGroup) {
            ((ViewGroup) v.getParent()).removeView(v);
        }
        return v;
    }

    @Override public boolean isChildSelectable(int groupPosition, int childPosition) { return true; }
}
