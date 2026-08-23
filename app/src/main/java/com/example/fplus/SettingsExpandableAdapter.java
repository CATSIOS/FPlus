package com.example.fplus;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseExpandableListAdapter;
import android.widget.RadioGroup;
import android.widget.TextView;

/**
 * 原生 ExpandableListView 设置页风格 Adapter（Android Settings 样式）。
 * <p>
 * 构造时预 inflate 两个 child view，避免在 getChildView 中复用失败导致
 * RadioGroup 引用 NPE；MainActivity 直接通过 {@link #getModelRadioGroup()}
 * / {@link #getBackendRadioGroup()} 拿引用，id 与原 layout 完全一致。
 */
public class SettingsExpandableAdapter extends BaseExpandableListAdapter {

    private final Context mContext;
    private final LayoutInflater mInflater;
    private final String[] mGroups;

    // 构造预加载，防 NPE
    private final View mChildModel;
    private final View mChildBackend;

    public SettingsExpandableAdapter(Context context, String[] groups) {
        mContext = context;
        mInflater = LayoutInflater.from(context);
        mGroups = groups != null ? groups : new String[]{"模型", "推理后端"};

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
        // 只两个 child，不复用，直接返回预加载的 view，避免 RadioGroup 状态丢失 / NPE
        return groupPosition == 0 ? mChildModel : mChildBackend;
    }

    @Override public boolean isChildSelectable(int groupPosition, int childPosition) { return true; }
}
