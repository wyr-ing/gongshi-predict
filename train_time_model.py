import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    page_title="Manhour Prediction System",
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
        fig_width = 8
        fig_height = 6
        font_size = 8
        title_size = 10
        legend_size = 7
        marker_size = 30
        tick_size = 7
    elif screen_width < 1024:
        fig_width = 10
        fig_height = 7
        font_size = 9
        title_size = 12
        legend_size = 8
        marker_size = 40
        tick_size = 8
    elif screen_width < 1366:
        fig_width = 11
        fig_height = 7.5
        font_size = 10
        title_size = 13
        legend_size = 9
        marker_size = 45
        tick_size = 9
    else:
        fig_width = 12
        fig_height = 8
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
# 列名映射（增强版 - 支持"元件总数"）
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
    
    theory_col = None
    theory_keywords = ['理论工时/s', '理论工时', '标准工时', '理论时间']
    for keyword in theory_keywords:
        for col in columns:
            if keyword in col:
                theory_col = col
                break
        if theory_col is not None:
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
                df_clean = df_clean.dropna()
                df_clean = df_clean[df_clean['点位数'] > 0]
                return df_clean
        except Exception as e:
            print(f"加载{line_type}数据失败: {e}")
            return None
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
# 理论工时计算
# ============================================================
def calculate_theory_time(point_count, a=0.5, b=2.0):
    return a * point_count + b

# ============================================================
# 3D对比图（核心改动）
# ============================================================
def plot_chart_3d(df, model, poly, mape, point_count=None, predicted_time=None, line_type="SMT"):
    
    screen = get_screen_size()
    
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    if len(X) == 0:
        fig = plt.figure(figsize=(screen['fig_width'], screen['fig_height']))
        ax = fig.add_subplot(111, projection='3d')
        ax.text(0.5, 0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=20)
        return fig
    
    # 生成平滑曲线数据
    x_min_plot = max(0, X.min() - 50)
    x_max_plot = X.max() + 50
    X_smooth = np.linspace(x_min_plot, x_max_plot, 300).reshape(-1, 1)
    X_smooth_poly = poly.transform(X_smooth)
    y_pred_smooth = model.predict(X_smooth_poly)
    y_theory = calculate_theory_time(X_smooth.flatten())
    
    # 创建3D图
    fig = plt.figure(figsize=(screen['fig_width'], screen['fig_height']), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 为每个数据点添加一个小的Z轴偏移（残差），形成3D效果
    # 计算残差
    X_poly = poly.transform(X)
    y_pred_all = model.predict(X_poly)
    residuals = y.flatten() - y_pred_all
    
    # 归一化残差用于颜色映射
    norm_residuals = (residuals - residuals.min()) / (residuals.max() - residuals.min() + 1e-10)
    
    # 3D散点图 - 实际数据
    scatter = ax.scatter(
        X.flatten(), 
        y.flatten(), 
        residuals * 0.5,  # Z轴：残差的缩放版本，增加立体感
        c=norm_residuals,
        cmap='coolwarm',
        s=screen['marker_size'],
        alpha=0.8,
        label='Actual Data',
        zorder=3
    )
    
    # 颜色条
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Residual Magnitude', fontsize=screen['font_size'])
    
    # 预测曲线投影到3D空间（在Z=0平面）
    ax.plot(
        X_smooth.flatten(), 
        y_pred_smooth, 
        np.zeros_like(X_smooth.flatten()),
        color='#d62728', 
        linewidth=2.5, 
        label='Prediction Curve',
        zorder=5
    )
    
    # 理论直线投影到3D空间（在Z=0平面）
    ax.plot(
        X_smooth.flatten(), 
        y_theory, 
        np.zeros_like(X_smooth.flatten()),
        color='#2ca02c', 
        linewidth=2, 
        linestyle='--', 
        label='Theory Line',
        zorder=4
    )
    
    # 预测点标记（在Z=0平面）
    if point_count is not None and predicted_time is not None:
        ax.scatter(
            [point_count], 
            [predicted_time], 
            [0],
            color='#ff6b6b', 
            s=screen['marker_size'] * 3.5,
            edgecolors='white', 
            linewidth=2, 
            zorder=6,
            label=f'Prediction: {point_count} pts → {predicted_time:.1f}s'
        )
        # 投影线到坐标轴
        ax.plot([point_count, point_count], [0, predicted_time], [0, 0], 
                color='#ff6b6b', linestyle=':', alpha=0.5, linewidth=1)
        ax.plot([0, point_count], [predicted_time, predicted_time], [0, 0], 
                color='#ff6b6b', linestyle=':', alpha=0.5, linewidth=1)
    
    # 设置标签（英文）
    ax.set_xlabel('Point Count', fontsize=screen['font_size'], fontweight='bold', labelpad=10)
    ax.set_ylabel('Time (seconds)', fontsize=screen['font_size'], fontweight='bold', labelpad=10)
    ax.set_zlabel('Residual', fontsize=screen['font_size'], fontweight='bold', labelpad=10)
    
    # 标题
    title = f'📊 {line_type} 3D Manhour Prediction Chart'
    ax.set_title(title, fontsize=screen['title_size'], fontweight='bold', pad=20)
    
    # 图例
    ax.legend(loc='upper left', fontsize=screen['legend_size'], framealpha=0.92)
    
    # 设置视角
    ax.view_init(elev=25, azim=-60)
    
    # 设置坐标轴范围
    x_max = X.max() * 1.15
    y_max = max(y.max(), y_theory.max(), y_pred_smooth.max()) * 1.2
    z_max = max(abs(residuals.min()), abs(residuals.max())) * 1.5
    
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, max(y_max, 10))
    ax.set_zlim(-z_max, z_max)
    
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
    
    # 设置z轴刻度
    z_ticks = np.linspace(-z_max, z_max, 5)
    ax.set_zticks(z_ticks)
    
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
            status = "within normal range"
        else:
            status = "outside normal range"

        system_prompt = f"""You are a {line_type} production line manhour prediction data analysis expert with extensive production line experience.

User input point count {point_count}, model predicted manhour {p:.2f} seconds.
Theoretical standard manhour {theory:.2f} seconds, deviation {dev:+.1f}%, {status} (normal error range ±{mape_val:.1f}%).

Please strictly follow this format:
1. Model predicted manhour {p:.2f} seconds.
2. Theoretical standard manhour {theory:.2f} seconds,
3. Deviation {dev:+.1f}%, {status}.
4. Then briefly analyze the reasons."""

        user_message = f"User input point count {point_count}, please analyze the prediction result."
    else:
        system_prompt = f"You are a {line_type} production line manhour prediction data analysis expert. Please prompt the user to input a specific point count for prediction analysis."

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
        return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Connection failed: {str(e)}"

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
        st.session_state[f"raw_df_{line_type}"] = None

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
        
        model, poly, r2, mae, mape, residuals, y_pred = train_prediction_model(saved_df)
        
        if model is not None:
            st.session_state[f"model_trained_{line_type}"] = True
            st.session_state[f"model_{line_type}"] = model
            st.session_state[f"poly_{line_type}"] = poly
            st.session_state[f"r2_{line_type}"] = r2
            st.session_state[f"mae_{line_type}"] = mae
            st.session_state[f"mape_{line_type}"] = mape
            st.session_state[f"df_{line_type}"] = saved_df
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
    st.markdown("### 🏭 Line Selection")
    line_type = st.radio(
        "Select Line",
        ["SMT", "DIP"],
        index=0 if st.session_state.current_line_type == "SMT" else 1,
        horizontal=True
    )
    
    if line_type != st.session_state.current_line_type:
        st.session_state.current_line_type = line_type
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Data Management")
    is_trained = st.session_state.get(f"model_trained_{line_type}", False)
    df = st.session_state.get(f"df_{line_type}")
    
    if is_trained and df is not None:
        st.success(f"✅ Current Data: {len(df)} rows ({line_type})")
    else:
        st.warning(f"⚠️ No {line_type} data available")

    st.markdown("---")
    
    st.markdown("#### 🔒 Admin Verification")
    admin_pwd = st.text_input("Upload Password", type="password", key="admin_pwd")
    if st.button("Verify & Upload", use_container_width=True):
        if hash_password(admin_pwd) == hash_password("admin123"):
            st.session_state.upload_authorized = True
            st.success("Verification successful, please upload data")
        else:
            st.session_state.upload_authorized = False
            st.error("Incorrect password")

    if st.session_state.upload_authorized:
        st.markdown("---")
        st.markdown(f"#### 📤 Upload {line_type} Data")
        st.caption("Supports Excel files with multiple columns. Auto-detects '单板点数', '元件总数', and '实际工时/s'")
        uploaded_file = st.file_uploader("Select Excel file", type=["xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file:
            df_raw = pd.read_excel(uploaded_file)
            
            point_col, actual_col, theory_col = get_column_mapping(df_raw)
            
            if point_col is not None and actual_col is not None:
                df = df_raw[[point_col, actual_col]].copy()
                df.columns = ['点位数', '实际工时']
                df = df.dropna()
                df = df[df['点位数'] > 0]
                
                if len(df) > 0:
                    st.session_state[f"model_trained_{line_type}"] = False
                    save_data(df, line_type)
                    
                    st.success(f"✅ {line_type} data uploaded, {len(df)} rows")
                    st.info(f"Detected: '{point_col}' → Point Count, '{actual_col}' → Actual Time")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Data is empty or all point counts are <= 0")
            else:
                st.error(f"❌ Could not find '单板点数'/'元件总数' or '实际工时/s' column. Current columns: {df_raw.columns.tolist()}")

    st.markdown("---")
    with st.expander("📋 Sample Data Format"):
        st.markdown("""
        | Line | Point Count | Actual Time/s | Theory Time/s |
        |------|---------|-----------|-----------|
        | L1 | 71 | 10.22 | 9.67 |
        | L2 | 68 | 57.60 | 68.40 |
        """)
        st.caption("System auto-detects '单板点数', '元件总数' and '实际工时/s' columns")

    data_file = get_data_file(line_type)
    if os.path.exists(data_file):
        mod_time = os.path.getmtime(data_file)
        update_time = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        st.markdown("---")
        st.caption(f"📅 {line_type} Data Updated: {update_time}")

# ============================================================
# 标题
# ============================================================
st.markdown(f"<h1 style='text-align: center;'>⚙️ {st.session_state.current_line_type} Manhour Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

# ============================================================
# 左右两栏
# ============================================================
left_col, right_col = st.columns(2, gap="large")

# ============================================================
# 左侧：模型评估 + 3D对比图
# ============================================================
with left_col:
    is_trained = st.session_state.get(f"model_trained_{line_type}", False)
    df = st.session_state.get(f"df_{line_type}")
    
    if is_trained and df is not None:
        with st.container():
            st.markdown("### 📊 Model Evaluation")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                r2_val = st.session_state.get(f"r2_{line_type}")
                st.metric("R²", f"{r2_val:.3f}" if r2_val is not None else "--")
            with col2:
                mae_val = st.session_state.get(f"mae_{line_type}")
                st.metric("MAE", f"{mae_val:.2f}" if mae_val is not None else "--", help="Mean Absolute Error (seconds)")
            with col3:
                mape_val = st.session_state.get(f"mape_{line_type}")
                st.metric("MAPE", f"{mape_val:.1f}%" if mape_val is not None else "--", help="Mean Absolute Percentage Error")

        with st.container():
            st.markdown("### 📈 3D Comparison Chart")
            
            plot_placeholder = st.empty()
            
            model = st.session_state.get(f"model_{line_type}")
            poly = st.session_state.get(f"poly_{line_type}")
            mape = st.session_state.get(f"mape_{line_type}")
            
            if model is not None and poly is not None:
                if st.session_state.last_prediction is not None:
                    fig = plot_chart_3d(
                        df,
                        model,
                        poly,
                        mape,
                        point_count=st.session_state.last_prediction.get("point_count"),
                        predicted_time=st.session_state.last_prediction.get("predicted"),
                        line_type=line_type
                    )
                    plot_placeholder.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    fig = plot_chart_3d(
                        df, 
                        model, 
                        poly, 
                        mape,
                        line_type=line_type
                    )
                    plot_placeholder.pyplot(fig, use_container_width=True)
                    plt.close(fig)
    else:
        st.info(f"👈 Please upload {line_type} data on the left")

# ============================================================
# 右侧：AI智能体对话
# ============================================================
with right_col:
    st.markdown(f"### 🎯 {line_type} Manhour Prediction Assistant")
    st.caption("Enter point count, AI estimates manhour | Based on actual data-trained prediction model")

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
            status_text = "✅ Reliable"
        else:
            status_color = "#e74c3c"
            status_text = "⚠️ Outside Range"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%); 
                    padding: 0.8rem 1rem; 
                    border-radius: 10px; 
                    border-left: 4px solid #4a6cf7;
                    margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.8rem; color: #888;">Last Prediction ({line_type})</span>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #1f77b4;">
                        {point_count} pts → {p:.2f}s
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #888;">Theory</span>
                    <div style="font-size: 1rem; font-weight: 600;">{theory:.2f}s</div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #888;">Deviation</span>
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

    user_input = st.chat_input("Enter point count (e.g., 1000) or ask a question...")

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

        with st.spinner("AI analyzing..."):
            response = chat_with_ai(user_input, prediction_result, line_type)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with btn_col2:
        with st.expander("📊 Prediction History"):
            if st.session_state.prediction_history:
                for h in st.session_state.prediction_history[-20:]:
                    line = h.get('line_type', 'SMT')
                    st.write(f"- [{line}] {h['point_count']} pts: {h['predicted']:.1f}s (dev {h['deviation_pct']:+.1f}%)")
            else:
                st.write("No prediction records")
