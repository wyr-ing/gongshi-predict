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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="工时预测系统 - SMT/DIP",
    page_icon="⚙️",
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
# 屏幕自适应工具函数
# ============================================================
def get_screen_size():
    try:
        screen_width = st.session_state.get('screen_width', 1200)
    except:
        screen_width = 1200
    
    if screen_width < 768:
        fig_width = 6
        fig_height = 4.5
        font_size = 8
        title_size = 10
        legend_size = 7
        marker_size = 30
        tick_size = 7
    elif screen_width < 1024:
        fig_width = 8
        fig_height = 5.5
        font_size = 9
        title_size = 12
        legend_size = 8
        marker_size = 40
        tick_size = 8
    elif screen_width < 1366:
        fig_width = 10
        fig_height = 6
        font_size = 10
        title_size = 13
        legend_size = 9
        marker_size = 45
        tick_size = 9
    else:
        fig_width = 12
        fig_height = 6.5
        font_size = 11
        title_size = 14
        legend_size = 9.5
        marker_size = 55
        tick_size = 10
    
    return {
        'fig_width': fig_width,
        'fig_height': fig_height,
        'font_size': font_size,
        'title_size': title_size,
        'legend_size': legend_size,
        'marker_size': marker_size,
        'tick_size': tick_size
    }

# ============================================================
# 列名映射
# ============================================================
def get_column_mapping(df):
    columns = df.columns.tolist()
    
    point_col = None
    for col in columns:
        if '单板点数' in col or '点位数' in col or '点数' in col:
            point_col = col
            break
    
    actual_col = None
    for col in columns:
        if '实际工时/s' in col or '实际工时' in col:
            actual_col = col
            break
    
    theory_col = None
    for col in columns:
        if '理论工时/s' in col or '理论工时' in col:
            theory_col = col
            break
    
    return point_col, actual_col, theory_col

# ============================================================
# 预测历史保存/加载
# ============================================================
def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except:
        pass

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

# ============================================================
# 数据保存/加载（支持SMT和DIP）
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
            point_col, actual_col, theory_col = get_column_mapping(df)
            if point_col is not None and actual_col is not None:
                df_clean = df[[point_col, actual_col]].copy()
                df_clean.columns = ['点位数', '实际工时']
                return df_clean
        except:
            pass
    return None

# ============================================================
# 训练预测模型
# ============================================================
def train_prediction_model(df):
    if len(df) < 2:
        return None, None, None, None, None, None, None
    
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    residuals = y - y_pred
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    return model, poly, r2, mae, mape, residuals, y_pred

# ============================================================
# 检测异常数据（基于残差标准差，可调阈值）
# ============================================================
def detect_outliers_by_std(df, model, poly, threshold_std=3.0):
    """
    基于残差标准差检测异常数据
    threshold_std: 标准差倍数，值越大越宽松
    """
    if len(df) < 2:
        return df, pd.DataFrame(), [], {'total_count': len(df), 'outlier_count': 0, 'clean_count': len(df), 'outlier_ratio': 0, 'threshold_std': threshold_std, 'threshold_value': 0, 'std_residual': 0, 'mean_residual': 0}
    
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    X_poly = poly.transform(X)
    y_pred = model.predict(X_poly)
    
    residuals = y - y_pred
    std_residual = np.std(residuals, ddof=1) if len(residuals) > 1 else 0
    mean_residual = np.mean(residuals)
    
    # 计算阈值
    threshold = threshold_std * std_residual if std_residual > 0 else float('inf')
    
    # 检测异常（残差绝对值超出阈值）
    outlier_mask = np.abs(residuals) > threshold if threshold != float('inf') else np.zeros(len(df), dtype=bool)
    
    outlier_indices = df.index[outlier_mask].tolist()
    outlier_data = df.loc[outlier_mask].copy()
    if len(outlier_data) > 0:
        outlier_data['残差'] = residuals[outlier_mask]
        outlier_data['预测值'] = y_pred[outlier_mask]
        outlier_data['残差百分比'] = (residuals[outlier_mask] / y[outlier_mask] * 100)
    
    # 正常数据
    clean_df = df.loc[~outlier_mask].copy()
    
    stats = {
        'total_count': len(df),
        'outlier_count': len(outlier_data),
        'clean_count': len(clean_df),
        'outlier_ratio': len(outlier_data) / len(df) * 100 if len(df) > 0 else 0,
        'threshold_std': threshold_std,
        'threshold_value': threshold if threshold != float('inf') else 0,
        'std_residual': std_residual,
        'mean_residual': mean_residual
    }
    
    return clean_df, outlier_data, outlier_indices, stats

# ============================================================
# 理论工时计算
# ============================================================
def calculate_theory_time(point_count, a=0.0362, b=0.5):
    return a * point_count + b

# ============================================================
# 对比图（自适应版）
# ============================================================
def plot_chart(df, model, poly, mape, point_count=None, predicted_time=None, outlier_df=None, line_type="SMT"):
    
    screen = get_screen_size()
    
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    if len(X) == 0:
        fig, ax = plt.subplots(figsize=(screen['fig_width'], screen['fig_height']), dpi=100)
        ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=20)
        return fig
    
    x_min_plot = max(0, X.min() - 50)
    x_max_plot = X.max() + 50
    X_smooth = np.linspace(x_min_plot, x_max_plot, 300).reshape(-1, 1)
    
    X_smooth_poly = poly.transform(X_smooth)
    y_pred_smooth = model.predict(X_smooth_poly)
    y_theory = calculate_theory_time(X_smooth.flatten())

    fig, ax = plt.subplots(figsize=(screen['fig_width'], screen['fig_height']), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)

    # 正常数据点
    ax.scatter(X, y, color='#1f77b4', s=screen['marker_size'], alpha=0.7, 
               label='Normal Data', zorder=3)
    
    # 异常数据点
    if outlier_df is not None and len(outlier_df) > 0:
        ax.scatter(outlier_df['点位数'], outlier_df['实际工时'], 
                   color='red', s=screen['marker_size'] * 1.8, alpha=0.8,
                   marker='x', linewidth=2.5,
                   label=f'Outliers ({len(outlier_df)} removed)', zorder=5)
    
    # 预测曲线
    ax.plot(X_smooth, y_pred_smooth, color='#d62728', linewidth=2.5, 
            label='Prediction Curve', zorder=2)
    
    # 理论直线
    ax.plot(X_smooth, y_theory, color='#2ca02c', linewidth=2, linestyle='--', 
            label='Theory Line', zorder=2)
    
    # 误差带
    mape_val = mape if mape is not None else 17.0
    y_upper = y_pred_smooth * (1 + mape_val / 100)
    y_lower = y_pred_smooth * (1 - mape_val / 100)
    ax.fill_between(X_smooth.flatten(), y_lower, y_upper, 
                    color='#d62728', alpha=0.10, 
                    label=f'±{mape_val:.1f}% Error Band')

    # 预测点标记
    if point_count is not None and predicted_time is not None:
        ax.scatter([point_count], [predicted_time], color='#ff6b6b', 
                   s=screen['marker_size'] * 3.5,
                   edgecolors='white', linewidth=2, zorder=6, 
                   label=f'Prediction: {point_count} pts → {predicted_time:.1f}s')
        ax.axvline(x=point_count, color='#ff6b6b', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.axhline(y=predicted_time, color='#ff6b6b', linestyle=':', alpha=0.6, linewidth=1.2)

    ax.legend(loc='upper left', fontsize=screen['legend_size'], 
              framealpha=0.92, edgecolor='#ccc')
    
    ax.set_xlabel('Point Count', fontsize=screen['font_size'], fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=screen['font_size'], fontweight='bold')
    
    # 标题 - 显示产线类型
    title = f'📊 {line_type} Manhour Prediction Chart'
    if outlier_df is not None and len(outlier_df) > 0:
        title += f' (Auto-cleaned: {len(outlier_df)} outliers removed)'
    ax.set_title(title, fontsize=screen['title_size'], fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.25, linestyle='--')
    
    # 坐标轴范围
    x_max = X.max() * 1.15
    y_max = max(y.max(), y_theory.max(), y_pred_smooth.max()) * 1.2
    
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, max(y_max, 10))
    
    # 智能刻度设置
    if x_max <= 100:
        x_step = 10
    elif x_max <= 200:
        x_step = 20
    elif x_max <= 500:
        x_step = 50
    elif x_max <= 1000:
        x_step = 100
    else:
        x_step = 200
    
    x_ticks = np.arange(0, x_max + x_step, x_step)
    ax.set_xticks(x_ticks)
    
    if y_max <= 50:
        y_step = 10
    elif y_max <= 100:
        y_step = 20
    elif y_max <= 500:
        y_step = 50
    elif y_max <= 1000:
        y_step = 100
    else:
        y_step = 200
    
    y_max_rounded = int(np.ceil(y_max / y_step)) * y_step
    y_ticks = np.arange(0, y_max_rounded + y_step, y_step)
    ax.set_yticks(y_ticks)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=screen['tick_size'])

    plt.tight_layout()
    return fig

# ============================================================
# 预测函数
# ============================================================
def predict_time(point_count, line_type):
    state_key = f"model_trained_{line_type}"
    if st.session_state.get(state_key, False) and st.session_state.get(f"model_{line_type}") is not None:
        X_input = np.array([[point_count]])
        X_input_poly = st.session_state.get(f"poly_{line_type}").transform(X_input)
        predicted = st.session_state.get(f"model_{line_type}").predict(X_input_poly)[0]
        theory = calculate_theory_time(point_count)
        deviation_pct = (predicted - theory) / theory * 100
        return {
            "predicted": predicted,
            "theory": theory,
            "deviation_pct": deviation_pct,
            "mape": st.session_state.get(f"mape_{line_type}", 17.0),
            "r2": st.session_state.get(f"r2_{line_type}"),
            "mae": st.session_state.get(f"mae_{line_type}")
        }
    return None

# ============================================================
# AI对话
# ============================================================
def chat_with_ai(user_message, prediction_result=None, line_type="SMT"):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    if prediction_result:
        p = prediction_result['predicted']
        dev = prediction_result['deviation_pct']
        theory = prediction_result['theory']
        point_count = prediction_result['point_count']
        mape_val = prediction_result['mape']

        if abs(dev) <= mape_val:
            status = "在正常范围内"
        else:
            status = "超出正常范围"

        system_prompt = f"""你是{line_type}产线工时预测数据分析专家，有丰富的产线经验。

用户输入点位数 {point_count}，模型预测工时 {p:.2f} 秒。
理论标准工时 {theory:.2f} 秒，偏差 {dev:+.1f}%，{status}（正常误差范围 ±{mape_val:.1f}%）。

请严格按以下格式输出：
1.模型预测工时 {p:.2f} 秒。
2.理论标准工时 {theory:.2f} 秒，
3.偏差 {dev:+.1f}%，{status}。
4.然后简要分析原因。"""

        user_message = f"用户输入点位数{point_count}，请分析预测结果。"
    else:
        system_prompt = f"你是{line_type}产线工时预测数据分析专家。请提示用户输入具体点位数，以便进行预测分析。"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages[-10:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000
    }

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"API错误：{response.status_code}"
    except Exception as e:
        return f"连接失败：{str(e)}"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ============================================================
# 初始化会话状态
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "prediction_history" not in st.session_state:
    loaded_history = load_history()
    st.session_state.prediction_history = loaded_history if loaded_history else []

# 为SMT和DIP分别初始化状态
for line_type in ["SMT", "DIP"]:
    if f"model_trained_{line_type}" not in st.session_state:
        st.session_state[f"model_trained_{line_type}"] = False
        st.session_state[f"model_{line_type}"] = None
        st.session_state[f"poly_{line_type}"] = None
        st.session_state[f"mape_{line_type}"] = None
        st.session_state[f"df_{line_type}"] = None
        st.session_state[f"r2_{line_type}"] = None
        st.session_state[f"mae_{line_type}"] = None
        st.session_state[f"residuals_{line_type}"] = None
        st.session_state[f"outlier_df_{line_type}"] = None
        st.session_state[f"clean_stats_{line_type}"] = None
        st.session_state[f"raw_df_{line_type}"] = None
        st.session_state[f"outlier_threshold_{line_type}"] = 3.0

if "upload_authorized" not in st.session_state:
    st.session_state.upload_authorized = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_prediction_result" not in st.session_state:
    st.session_state.last_prediction_result = None
if "current_line_type" not in st.session_state:
    st.session_state.current_line_type = "SMT"

if "screen_width" not in st.session_state:
    st.session_state.screen_width = 1200

# ============================================================
# 加载数据函数
# ============================================================
def load_data_for_line(line_type):
    saved_df = load_saved_data(line_type)
    if saved_df is not None and len(saved_df) > 0:
        st.session_state[f"raw_df_{line_type}"] = saved_df.copy()
        
        # 先用所有数据训练一次，获取残差
        temp_model, temp_poly, temp_r2, temp_mae, temp_mape, temp_residuals, temp_y_pred = train_prediction_model(saved_df)
        
        if temp_model is not None:
            # 检测异常（使用当前阈值）
            threshold = st.session_state.get(f"outlier_threshold_{line_type}", 3.0)
            clean_df, outlier_df, outlier_indices, stats = detect_outliers_by_std(
                saved_df, temp_model, temp_poly, threshold
            )
            
            st.session_state[f"outlier_df_{line_type}"] = outlier_df
            st.session_state[f"clean_stats_{line_type}"] = stats
            
            # 用清理后的数据训练最终模型
            df_to_use = clean_df if len(clean_df) > 0 else saved_df
            model, poly, r2, mae, mape, residuals, y_pred = train_prediction_model(df_to_use)
            
            if model is not None:
                st.session_state[f"model_trained_{line_type}"] = True
                st.session_state[f"model_{line_type}"] = model
                st.session_state[f"poly_{line_type}"] = poly
                st.session_state[f"r2_{line_type}"] = r2
                st.session_state[f"mae_{line_type}"] = mae
                st.session_state[f"mape_{line_type}"] = mape
                st.session_state[f"df_{line_type}"] = df_to_use
                st.session_state[f"residuals_{line_type}"] = residuals
                return True
    return False

# 自动加载SMT和DIP数据
for line_type in ["SMT", "DIP"]:
    if not st.session_state.get(f"model_trained_{line_type}", False):
        load_data_for_line(line_type)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    # 产线选择
    st.markdown("### 🏭 产线选择")
    line_type = st.radio(
        "选择产线",
        ["SMT", "DIP"],
        index=0 if st.session_state.current_line_type == "SMT" else 1,
        horizontal=True,
        help="选择要查看和预测的产线"
    )
    
    if line_type != st.session_state.current_line_type:
        st.session_state.current_line_type = line_type
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ⚙️ 数据管理")
    is_trained = st.session_state.get(f"model_trained_{line_type}", False)
    df = st.session_state.get(f"df_{line_type}")
    
    if is_trained and df is not None:
        st.success(f"✅ 当前数据：{len(df)} 行 ({line_type})")
        stats = st.session_state.get(f"clean_stats_{line_type}")
        if stats is not None and stats['outlier_count'] > 0:
            st.info(f"🧹 已剔除 {stats['outlier_count']} 个异常数据")
    else:
        st.warning(f"⚠️ 暂无{line_type}数据")

    st.markdown("---")
    
    # ============================================================
    # 异常剔除阈值设置（核心功能）
    # ============================================================
    st.markdown("#### 🎯 异常剔除阈值")
    st.caption("数值越大，剔除越少（越宽松）")
    
    current_threshold = st.session_state.get(f"outlier_threshold_{line_type}", 3.0)
    
    # 滑块：0~5，步长0.1，默认3.0
    threshold_value = st.slider(
        "标准差倍数",
        min_value=0.0,
        max_value=5.0,
        value=current_threshold,
        step=0.1,
        key=f"threshold_slider_{line_type}",
        help="0=不剔除，数值越大剔除越少"
    )
    
    # 显示当前设置的效果
    if threshold_value == 0:
        st.info("🔓 当前：不剔除任何数据")
    elif threshold_value <= 2.0:
        st.warning("⚠️ 当前：严格模式（剔除较多）")
    elif threshold_value <= 3.5:
        st.info("✅ 当前：标准模式（推荐）")
    else:
        st.success("✅ 当前：宽松模式（剔除较少）")
    
    # 如果阈值改变，重新训练
    if threshold_value != current_threshold:
        st.session_state[f"outlier_threshold_{line_type}"] = threshold_value
        st.session_state[f"model_trained_{line_type}"] = False
        st.rerun()
    
    # 显示异常统计
    stats = st.session_state.get(f"clean_stats_{line_type}")
    if stats is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("异常数量", stats['outlier_count'])
        with col2:
            st.metric("占比", f"{stats['outlier_ratio']:.1f}%")
        
        st.caption(f"当前阈值：±{stats['threshold_value']:.2f} 秒")
        st.caption(f"残差标准差：±{stats['std_residual']:.2f} 秒")
    
    # 查看异常数据详情
    outlier_df = st.session_state.get(f"outlier_df_{line_type}")
    if outlier_df is not None and len(outlier_df) > 0:
        with st.expander(f"📋 查看异常数据 ({len(outlier_df)}个)"):
            display_df = outlier_df[['点位数', '实际工时', '预测值', '残差', '残差百分比']].copy()
            display_df['残差百分比'] = display_df['残差百分比'].round(2)
            st.dataframe(display_df, use_container_width=True)
    
    # 重置按钮
    if st.button("🔄 重置为默认 (3.0倍标准差)", use_container_width=True):
        st.session_state[f"outlier_threshold_{line_type}"] = 3.0
        st.session_state[f"model_trained_{line_type}"] = False
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🔒 管理员验证")
    admin_pwd = st.text_input("上传密码", type="password", key="admin_pwd")
    if st.button("验证并上传", use_container_width=True):
        if hash_password(admin_pwd) == hash_password("admin123"):
            st.session_state.upload_authorized = True
            st.success("验证成功，请上传数据")
        else:
            st.session_state.upload_authorized = False
            st.error("密码错误")

    if st.session_state.upload_authorized:
        st.markdown("---")
        st.markdown(f"#### 📤 上传{line_type}数据")
        st.caption("支持包含多列的Excel文件，自动识别'单板点数'和'实际工时/s'列")
        uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file:
            df_raw = pd.read_excel(uploaded_file)
            
            point_col, actual_col, theory_col = get_column_mapping(df_raw)
            
            if point_col is not None and actual_col is not None:
                df = df_raw[[point_col, actual_col]].copy()
                df.columns = ['点位数', '实际工时']
                df = df.dropna()
                
                st.session_state[f"model_trained_{line_type}"] = False
                save_data(df, line_type)
                
                st.success(f"✅ {line_type}数据已上传，共 {len(df)} 行")
                st.info(f"识别到列：'{point_col}' → 点位数，'{actual_col}' → 实际工时")
                st.balloons()
                st.rerun()
            else:
                st.error(f"❌ 未找到'单板点数'或'实际工时/s'列，当前列名：{df_raw.columns.tolist()}")

    st.markdown("---")
    with st.expander("📋 示例数据格式"):
        st.markdown("""
        | 线别 | 单板点数 | 实际工时/s | 理论工时/s |
        |------|---------|-----------|-----------|
        | L1 | 71 | 10.22 | 9.67 |
        | L2 | 68 | 57.60 | 68.40 |
        """)
        st.caption("系统会自动识别'单板点数'和'实际工时/s'列")

    # 显示数据更新时间
    data_file = get_data_file(line_type)
    if os.path.exists(data_file):
        mod_time = os.path.getmtime(data_file)
        update_time = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        st.markdown("---")
        st.caption(f"📅 {line_type}数据更新：{update_time}")

# ============================================================
# 标题
# ============================================================
st.markdown(f"<h1 style='text-align: center;'>⚙️ {st.session_state.current_line_type} 工时预测系统</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

# ============================================================
# 左右两栏
# ============================================================
left_col, right_col = st.columns(2, gap="large")

# ============================================================
# 左侧：模型评估 + 对比图
# ============================================================
with left_col:
    is_trained = st.session_state.get(f"model_trained_{line_type}", False)
    df = st.session_state.get(f"df_{line_type}")
    
    if is_trained and df is not None:
        with st.container():
            st.markdown("### 📊 模型评估")
            
            # 显示清理信息
            stats = st.session_state.get(f"clean_stats_{line_type}")
            if stats is not None:
                if stats['outlier_count'] > 0:
                    st.info(f"🧹 已剔除 {stats['outlier_count']} 个异常数据，使用 {len(df)} 条干净数据训练")
                else:
                    st.success("✅ 数据质量良好，无异常数据")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                r2_val = st.session_state.get(f"r2_{line_type}")
                st.metric("R²", f"{r2_val:.3f}" if r2_val is not None else "--")
            with col2:
                mae_val = st.session_state.get(f"mae_{line_type}")
                st.metric("MAE", f"{mae_val:.2f}" if mae_val is not None else "--", help="平均绝对误差（秒）")
            with col3:
                mape_val = st.session_state.get(f"mape_{line_type}")
                st.metric("MAPE", f"{mape_val:.1f}%" if mape_val is not None else "--", help="平均绝对百分比误差")

        with st.container():
            st.markdown("### 📈 对比图")
            
            plot_placeholder = st.empty()
            
            outlier_df = st.session_state.get(f"outlier_df_{line_type}")
            model = st.session_state.get(f"model_{line_type}")
            poly = st.session_state.get(f"poly_{line_type}")
            mape = st.session_state.get(f"mape_{line_type}")
            
            if model is not None and poly is not None:
                if st.session_state.last_prediction is not None:
                    fig = plot_chart(
                        df,
                        model,
                        poly,
                        mape,
                        point_count=st.session_state.last_prediction.get("point_count"),
                        predicted_time=st.session_state.last_prediction.get("predicted"),
                        outlier_df=outlier_df,
                        line_type=line_type
                    )
                    plot_placeholder.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    fig = plot_chart(
                        df, 
                        model, 
                        poly, 
                        mape,
                        outlier_df=outlier_df,
                        line_type=line_type
                    )
                    plot_placeholder.pyplot(fig, use_container_width=True)
                    plt.close(fig)
    else:
        st.info(f"👈 请在左侧上传{line_type}数据")

# ============================================================
# 右侧：AI智能体对话
# ============================================================
with right_col:
    st.markdown(f"### 🎯 {line_type}工时预测小助手")
    st.caption("输入点位数，AI估算工时 | 基于实际数据训练的预测模型")

    chat_container = st.container(height=280)

    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            elif msg["role"] == "assistant":
                st.chat_message("assistant").write(msg["content"])

    # 预测结果卡片
    if st.session_state.last_prediction_result is not None:
        last = st.session_state.last_prediction_result
        p = last["predicted"]
        dev = last["deviation_pct"]
        theory = last["theory"]
        point_count = last["point_count"]
        mape_val = last["mape"]

        if abs(dev) <= mape_val:
            status_color = "#2ecc71"
            status_text = "✅ 可信"
        else:
            status_color = "#e74c3c"
            status_text = "⚠️ 超出正常范围"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%); 
                    padding: 0.8rem 1rem; 
                    border-radius: 10px; 
                    border-left: 4px solid #4a6cf7;
                    margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.8rem; color: #888;">上次预测 ({line_type})</span>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #1f77b4;">
                        {point_count}点 → {p:.2f}s
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #888;">理论标准</span>
                    <div style="font-size: 1rem; font-weight: 600;">{theory:.2f}s</div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #888;">偏差</span>
                    <div style="font-size: 1rem; font-weight: 600; color: {'#2ecc71' if abs(dev) <= mape_val else '#e74c3c'};">
                        {dev:+.1f}%
                    </div>
                </div>
                <div>
                    <span style="background: {status_color}; color: white; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.7rem; font-weight: 600;">
                        {status_text}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    user_input = st.chat_input("输入点位数（如 1000）或提问...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        numbers = re.findall(r'\d+', user_input)
        prediction_result = None

        if numbers and st.session_state.get(f"model_trained_{line_type}", False):
            point_count = int(numbers[0])
            pred_data = predict_time(point_count, line_type)
            if pred_data:
                prediction_result = {
                    "point_count": point_count,
                    "predicted": pred_data["predicted"],
                    "theory": pred_data["theory"],
                    "deviation_pct": pred_data["deviation_pct"],
                    "mape": pred_data["mape"],
                    "r2": pred_data["r2"],
                    "mae": pred_data["mae"]
                }
                st.session_state.prediction_history.append({
                    "line_type": line_type,
                    "point_count": point_count,
                    "predicted": pred_data["predicted"],
                    "deviation_pct": pred_data["deviation_pct"]
                })
                save_history(st.session_state.prediction_history)

                st.session_state.last_prediction = {
                    "point_count": point_count,
                    "predicted": pred_data["predicted"]
                }
                st.session_state.last_prediction_result = {
                    "point_count": point_count,
                    "predicted": pred_data["predicted"],
                    "theory": pred_data["theory"],
                    "deviation_pct": pred_data["deviation_pct"],
                    "mape": pred_data["mape"]
                }

        with st.spinner("智能体分析中..."):
            response = chat_with_ai(user_input, prediction_result, line_type)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with btn_col2:
        with st.expander("📊 预测历史"):
            if st.session_state.prediction_history:
                for h in st.session_state.prediction_history[-20:]:
                    line = h.get('line_type', 'SMT')
                    st.write(f"- [{line}] {h['point_count']}点: {h['predicted']:.1f}s (偏差{h['deviation_pct']:+.1f}%)")
            else:
                st.write("暂无预测记录")
