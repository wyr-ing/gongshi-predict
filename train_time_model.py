import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import os
import json
import hashlib
from datetime import datetime
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="工时预测系统 - 方案一",
    page_icon="📊",
    layout="wide"
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置
# ============================================================
DATA_FILE_SMT = "smt_data.xlsx"
DATA_FILE_DIP = "dip_data.xlsx"
API_KEY = "sk-fvxkdwbhjcokafftooavzvedrlmmrffotehplsnfnjupogqb"
BASE_URL = "https://api.siliconflow.cn/v1"
HISTORY_FILE = "prediction_history.json"

# ============================================================
# 列名映射
# ============================================================
def get_column_mapping(df):
    columns = df.columns.tolist()
    
    point_col = None
    point_keywords = ['单板点数', '点位数', '元件总数', '点数', '元件数', '总点数']
    
    for keyword in point_keywords:
        for col in columns:
            if keyword in col:
                point_col = col
                break
        if point_col is not None:
            break
    
    actual_col = None
    actual_keywords = ['实际工时/s', '实际工时', '实际时间', '工时']
    for keyword in actual_keywords:
        for col in columns:
            if keyword in col:
                actual_col = col
                break
        if actual_col is not None:
            break
    
    return point_col, actual_col

# ============================================================
# 数据保存/加载
# ============================================================
def get_data_file(line_type):
    if line_type == "SMT":
        return DATA_FILE_SMT
    else:
        return DATA_FILE_DIP

def save_data(df, line_type):
    data_file = get_data_file(line_type)
    df.to_excel(data_file, index=False)

def load_saved_data(line_type):
    data_file = get_data_file(line_type)
    if os.path.exists(data_file):
        try:
            df = pd.read_excel(data_file)
            point_col, actual_col = get_column_mapping(df)
            if point_col is not None and actual_col is not None:
                df_clean = df[[point_col, actual_col]].copy()
                df_clean.columns = ['点位数', '实际工时']
                df_clean = df_clean.dropna()
                df_clean = df_clean[df_clean['点位数'] > 0]
                df_clean = df_clean[df_clean['实际工时'] > 0]
                return df_clean
        except Exception as e:
            return None
    return None

# ============================================================
# 数据探索分析
# ============================================================
def data_exploration(df):
    """生成数据探索报告"""
    report = {}
    
    # 基本统计
    report['总数据量'] = len(df)
    report['点位范围'] = (df['点位数'].min(), df['点位数'].max())
    report['工时范围'] = (df['实际工时'].min(), df['实际工时'].max())
    
    # 按点位分组统计
    grouped = df.groupby('点位数').agg({
        '实际工时': ['count', 'mean', 'std', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    }).reset_index()
    grouped.columns = ['点位数', '样本数', '均值', '标准差', '最小值', '最大值', 'Q1', 'Q3']
    grouped['变异系数'] = grouped['标准差'] / grouped['均值']
    
    report['分组统计'] = grouped
    
    # 识别高离散点位
    high_cv = grouped[grouped['变异系数'] > 0.5]
    report['高离散点位'] = len(high_cv)
    report['高离散点位详情'] = high_cv.sort_values('变异系数', ascending=False).head(20)
    
    return report

# ============================================================
# 数据清洗（按点位分组剔除异常值）
# ============================================================
def clean_data_by_group(df, sigma=3.0):
    """按点位分组，剔除超出均值±n*标准差的数据"""
    clean_list = []
    outlier_list = []
    
    for point in df['点位数'].unique():
        group = df[df['点位数'] == point]
        
        if len(group) < 3:
            clean_list.append(group)
            continue
        
        mean_val = group['实际工时'].mean()
        std_val = group['实际工时'].std()
        
        if std_val == 0:
            clean_list.append(group)
            continue
        
        lower = mean_val - sigma * std_val
        upper = mean_val + sigma * std_val
        
        clean = group[(group['实际工时'] >= lower) & (group['实际工时'] <= upper)]
        outliers = group[(group['实际工时'] < lower) | (group['实际工时'] > upper)]
        
        clean_list.append(clean)
        if len(outliers) > 0:
            outlier_list.append(outliers)
    
    clean_df = pd.concat(clean_list, ignore_index=True) if clean_list else df
    outlier_df = pd.concat(outlier_list, ignore_index=True) if outlier_list else pd.DataFrame()
    
    return clean_df, outlier_df

# ============================================================
# 分段建模
# ============================================================
def segment_model(df):
    """
    按点位区间分段建模：
    - 小点位 (1-50): 线性回归
    - 中点位 (51-150): 二次多项式
    - 大点位 (151+): RANSAC 稳健回归
    """
    models = {}
    segments = {}
    
    segments_config = [
        ('small', 1, 50, '线性回归', LinearRegression()),
        ('medium', 51, 150, '二次多项式', make_pipeline(PolynomialFeatures(2), LinearRegression())),
        ('large', 151, float('inf'), 'RANSAC', RANSACRegressor(random_state=42))
    ]
    
    for name, low, high, method, model in segments_config:
        segment_data = df[(df['点位数'] >= low) & (df['点位数'] <= high)]
        
        if len(segment_data) < 3:
            continue
        
        X = segment_data[['点位数']].values
        y = segment_data['实际工时'].values
        
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        models[name] = {
            'model': model,
            'data': segment_data,
            'method': method,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'range': f'{low}-{high}点',
            'sample_count': len(segment_data)
        }
        
        segments[name] = segment_data
    
    return models, segments

# ============================================================
# 预测函数
# ============================================================
def predict_by_segment(point_count, models):
    """根据点位选择对应模型进行预测"""
    if point_count <= 50 and 'small' in models:
        model_info = models['small']
        X = np.array([[point_count]])
        pred = model_info['model'].predict(X)[0]
        return pred, model_info['method'], model_info['range']
    elif point_count <= 150 and 'medium' in models:
        model_info = models['medium']
        X = np.array([[point_count]])
        pred = model_info['model'].predict(X)[0]
        return pred, model_info['method'], model_info['range']
    elif 'large' in models:
        model_info = models['large']
        X = np.array([[point_count]])
        pred = model_info['model'].predict(X)[0]
        return pred, model_info['method'], model_info['range']
    else:
        return None, None, None

# ============================================================
# 绘图函数（纯matplotlib，不依赖seaborn）
# ============================================================
def plot_exploration(df, grouped):
    """数据探索可视化 - 使用纯matplotlib"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1：点位样本量分布
    sample_data = grouped.groupby('点位数')['样本数'].first().reset_index()
    axes[0].bar(sample_data['点位数'], sample_data['样本数'], color='steelblue', alpha=0.7, width=0.8)
    axes[0].set_xlabel('点位数', fontsize=11)
    axes[0].set_ylabel('样本数', fontsize=11)
    axes[0].set_title('各点位样本量分布', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, sample_data['点位数'].max() * 1.05)
    
    # 图2：变异系数分布
    cv_data = grouped[grouped['变异系数'] <= 2]['变异系数'].dropna()
    if len(cv_data) > 0:
        axes[1].hist(cv_data, bins=min(30, len(cv_data)), color='coral', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='CV=0.5 (高离散阈值)')
        axes[1].set_xlabel('变异系数 (CV = 标准差/均值)', fontsize=11)
        axes[1].set_ylabel('点位数量', fontsize=11)
        axes[1].set_title('各点位工时变异系数分布', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, '数据不足以绘制变异系数分布', ha='center', va='center', fontsize=14)
        axes[1].set_title('各点位工时变异系数分布', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_segment_chart(df, models, clean_df=None, outlier_df=None, 
                       point_count=None, predicted_time=None, line_type="SMT"):
    """分段建模预测图 - 使用纯matplotlib"""
    
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=100)
    
    # 原始数据（浅色）
    ax.scatter(df['点位数'], df['实际工时'], color='#cccccc', s=15, 
               alpha=0.3, label='原始数据', zorder=1)
    
    # 清洗后数据（深色）
    if clean_df is not None and len(clean_df) > 0:
        ax.scatter(clean_df['点位数'], clean_df['实际工时'], color='#1f77b4', 
                   s=30, alpha=0.6, label='清洗后数据', zorder=3)
    
    # 异常值（红色）
    if outlier_df is not None and len(outlier_df) > 0:
        ax.scatter(outlier_df['点位数'], outlier_df['实际工时'], color='red', 
                   s=50, alpha=0.5, marker='x', linewidths=2,
                   label=f'异常值 ({len(outlier_df)}个)', zorder=4)
    
    # 绘制各段的预测曲线
    colors = {'small': '#2ca02c', 'medium': '#d62728', 'large': '#9467bd'}
    labels = {'small': '小点位 (1-50点) - 线性', 
              'medium': '中点位 (51-150点) - 二次', 
              'large': '大点位 (151+点) - RANSAC'}
    
    for name, info in models.items():
        if info['model'] is None:
            continue
        
        data = info['data']
        if len(data) < 2:
            continue
        
        X_min = data['点位数'].min()
        X_max = data['点位数'].max()
        
        if X_max == X_min:
            continue
        
        X_smooth = np.linspace(X_min - 5, X_max + 5, 100).reshape(-1, 1)
        y_smooth = info['model'].predict(X_smooth)
        
        ax.plot(X_smooth, y_smooth, color=colors.get(name, '#333'), 
                linewidth=2.5, label=labels.get(name, name), zorder=5)
        
        # 标注模型性能
        mid_x = (X_min + X_max) / 2
        if mid_x > 0:
            try:
                mid_y = info['model'].predict(np.array([[mid_x]]))[0]
                ax.text(mid_x, mid_y * 1.08, f'R²={info["r2"]:.3f}', 
                        fontsize=8, color=colors.get(name, '#333'), ha='center')
            except:
                pass
    
    # 预测点
    if point_count is not None and predicted_time is not None:
        ax.scatter([point_count], [predicted_time], color='#ff6b6b', 
                   s=120, edgecolors='white', linewidth=2, zorder=6,
                   label=f'预测: {point_count}点 → {predicted_time:.1f}s')
        ax.axvline(x=point_count, color='#ff6b6b', linestyle=':', alpha=0.6)
        ax.axhline(y=predicted_time, color='#ff6b6b', linestyle=':', alpha=0.6)
    
    ax.set_xlabel('点位数', fontsize=11, fontweight='bold')
    ax.set_ylabel('工时 (秒)', fontsize=11, fontweight='bold')
    ax.set_title(f'📊 {line_type} 分段建模预测图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9.5)
    ax.grid(True, alpha=0.25)
    
    # 坐标轴范围
    x_max = df['点位数'].max() * 1.1
    y_max = df['实际工时'].max() * 1.2
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, max(y_max, 10))
    
    plt.tight_layout()
    return fig

# ============================================================
# 初始化会话状态
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "prediction_history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                st.session_state.prediction_history = json.load(f)
        except:
            st.session_state.prediction_history = []
    else:
        st.session_state.prediction_history = []

for line_type in ["SMT", "DIP"]:
    if f"df_raw_{line_type}" not in st.session_state:
        st.session_state[f"df_raw_{line_type}"] = None
    if f"df_clean_{line_type}" not in st.session_state:
        st.session_state[f"df_clean_{line_type}"] = None
    if f"models_{line_type}" not in st.session_state:
        st.session_state[f"models_{line_type}"] = None
    if f"segments_{line_type}" not in st.session_state:
        st.session_state[f"segments_{line_type}"] = None
    if f"report_{line_type}" not in st.session_state:
        st.session_state[f"report_{line_type}"] = None

if "current_line_type" not in st.session_state:
    st.session_state.current_line_type = "SMT"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_prediction_result" not in st.session_state:
    st.session_state.last_prediction_result = None

# ============================================================
# 加载数据
# ============================================================
for line_type in ["SMT", "DIP"]:
    saved_df = load_saved_data(line_type)
    if saved_df is not None and len(saved_df) > 0:
        st.session_state[f"df_raw_{line_type}"] = saved_df
        
        # 数据探索
        report = data_exploration(saved_df)
        st.session_state[f"report_{line_type}"] = report
        
        # 数据清洗
        clean_df, outlier_df = clean_data_by_group(saved_df, sigma=3.0)
        
        # 分段建模
        models, segments = segment_model(clean_df)
        
        st.session_state[f"df_clean_{line_type}"] = clean_df
        st.session_state[f"models_{line_type}"] = models
        st.session_state[f"segments_{line_type}"] = segments

# ============================================================
# 主界面
# ============================================================
st.markdown("""
<div style="text-align: center; padding: 0.5rem 0;">
    <h1>📊 方案一：数据预处理 + 分组建模</h1>
    <p style="color: #666;">按点位区间分段建模，自动剔除异常值</p>
</div>
<hr>
""", unsafe_allow_html=True)

# 产线选择
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    line_type = st.radio(
        "选择产线",
        ["SMT", "DIP"],
        index=0 if st.session_state.current_line_type == "SMT" else 1,
        horizontal=True
    )
    if line_type != st.session_state.current_line_type:
        st.session_state.current_line_type = line_type
        st.rerun()

# ============================================================
# 数据探索报告
# ============================================================
report = st.session_state.get(f"report_{line_type}")
df_raw = st.session_state.get(f"df_raw_{line_type}")
df_clean = st.session_state.get(f"df_clean_{line_type}")
models = st.session_state.get(f"models_{line_type}")

if df_raw is not None and len(df_raw) > 0:
    
    # ============================================================
    # 数据概览
    # ============================================================
    with st.expander("📋 数据概览", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("原始数据量", len(df_raw))
        with col2:
            clean_count = len(df_clean) if df_clean is not None else 0
            outlier_count = len(df_raw) - clean_count
            st.metric("清洗后数据量", clean_count, delta=f"-{outlier_count} 异常值")
        with col3:
            point_range = f"{df_raw['点位数'].min():.0f} ~ {df_raw['点位数'].max():.0f}"
            st.metric("点位范围", point_range)
        with col4:
            time_range = f"{df_raw['实际工时'].min():.1f} ~ {df_raw['实际工时'].max():.1f}"
            st.metric("工时范围", time_range)
    
    # ============================================================
    # 数据质量分析
    # ============================================================
    with st.expander("📊 数据质量分析", expanded=True):
        if report is not None:
            grouped = report['分组统计']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**各点位样本量分布**")
                sample_data = grouped.groupby('点位数')['样本数'].first().reset_index()
                sample_df = pd.DataFrame({
                    '点位数': sample_data['点位数'],
                    '样本数': sample_data['样本数']
                })
                st.dataframe(sample_df, use_container_width=True, height=250)
            
            with col2:
                st.markdown("**高离散点位 (CV > 0.5)**")
                high_cv = report['高离散点位详情']
                if len(high_cv) > 0:
                    st.dataframe(high_cv[['点位数', '样本数', '均值', '标准差', '变异系数']], 
                                 use_container_width=True, height=250)
                else:
                    st.success("✅ 所有点位变异系数均 ≤ 0.5，数据质量良好")
            
            # 变异系数分布图（纯matplotlib）
            st.markdown("**变异系数分布**")
            fig = plot_exploration(df_raw, grouped)
            st.pyplot(fig)
            plt.close(fig)
    
    # ============================================================
    # 分段建模结果
    # ============================================================
    with st.expander("🎯 分段建模结果", expanded=True):
        if models is not None and len(models) > 0:
            cols = st.columns(3)
            for idx, (name, info) in enumerate(models.items()):
                with cols[idx % 3]:
                    color = '#2ecc71' if info['r2'] > 0.7 else '#f39c12' if info['r2'] > 0.4 else '#e74c3c'
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin: 0.2rem 0; border-left: 4px solid {color};">
                        <b>{info['method']}</b><br>
                        <span style="color: #666; font-size: 0.85rem;">{info['range']}</span><br>
                        样本数: {info['sample_count']}<br>
                        R²: <b style="color: {color};">{info['r2']:.3f}</b><br>
                        MAE: {info['mae']:.2f}s<br>
                        MAPE: {info['mape']:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
    
    # ============================================================
    # 预测图表
    # ============================================================
    st.markdown("### 📈 分段建模预测图")
    
    if df_clean is not None:
        # 找出异常值（原始数据中不在清洗后数据中的部分）
        outlier_df = df_raw[~df_raw.index.isin(df_clean.index)]
    else:
        outlier_df = pd.DataFrame()
    
    fig = plot_segment_chart(
        df_raw,
        models,
        clean_df=df_clean,
        outlier_df=outlier_df if len(outlier_df) > 0 else None,
        line_type=line_type
    )
    st.pyplot(fig)
    plt.close(fig)
    
    # ============================================================
    # 预测区域
    # ============================================================
    st.markdown("---")
    st.markdown("### 🔮 工时预测")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        max_point = int(df_raw['点位数'].max() * 1.5)
        point_input = st.number_input(
            "输入点位数",
            min_value=1,
            max_value=max_point,
            value=min(50, max_point),
            step=5
        )
        
        if st.button("🚀 预测", use_container_width=True):
            if models is not None:
                pred, method, range_str = predict_by_segment(point_input, models)
                if pred is not None:
                    st.session_state.last_prediction = {
                        "point_count": point_input,
                        "predicted": pred
                    }
                    st.session_state.last_prediction_result = {
                        "point_count": point_input,
                        "predicted": pred,
                        "method": method,
                        "range": range_str
                    }
                    st.rerun()
    
    with col2:
        if st.session_state.last_prediction_result is not None:
            result = st.session_state.last_prediction_result
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%); 
                        padding: 1rem 1.5rem; 
                        border-radius: 10px; 
                        border-left: 4px solid #4a6cf7;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 0.85rem; color: #888;">预测结果</span>
                        <div style="font-size: 2rem; font-weight: 700; color: #1f77b4;">
                            {result['predicted']:.1f} 秒
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.85rem; color: #888;">点位数</span>
                        <div style="font-size: 1.2rem; font-weight: 600;">{result['point_count']}</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.85rem; color: #888;">模型</span>
                        <div style="font-size: 0.9rem; font-weight: 600; color: #4a6cf7;">
                            {result.get('method', 'Unknown')}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.warning(f"⚠️ 暂无 {line_type} 数据，请先上传数据")
    
    # 上传区域
    with st.expander("📤 上传数据", expanded=True):
        uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx", "xls"])
        if uploaded_file:
            df_raw = pd.read_excel(uploaded_file)
            point_col, actual_col = get_column_mapping(df_raw)
            
            if point_col is not None and actual_col is not None:
                df = df_raw[[point_col, actual_col]].copy()
                df.columns = ['点位数', '实际工时']
                df = df.dropna()
                df = df[df['点位数'] > 0]
                df = df[df['实际工时'] > 0]
                
                if len(df) > 0:
                    # 保存数据
                    save_data(df, line_type)
                    st.success(f"✅ 数据已上传，共 {len(df)} 行")
                    st.rerun()
                else:
                    st.error("❌ 数据为空或包含无效值")
            else:
                st.error(f"❌ 未找到'单板点数'/'元件总数'或'实际工时/s'列，当前列名：{df_raw.columns.tolist()}")
