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
    page_title="Manhour Prediction System",
    page_icon="⚙️",
    layout="wide"
)

# ============================================================
# 配置
# ============================================================
DATA_FILE = "saved_data.xlsx"
API_KEY = "sk-fvxkdwbhjcokafftooavzvedrlmmrffotehplsnfnjupogqb"
BASE_URL = "https://api.siliconflow.cn/v1"
HISTORY_FILE = "prediction_history.json"

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
# 数据保存/加载
# ============================================================
def save_data(df):
    df.to_excel(DATA_FILE, index=False)

def load_saved_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_excel(DATA_FILE)
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
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    return model, poly, r2, mae, mape

# ============================================================
# 理论工时计算
# ============================================================
def calculate_theory_time(point_count, a=0.0362, b=0.5):
    return a * point_count + b

# ============================================================
# 对比图（英文版 - 无乱码）
# ============================================================
def plot_chart(df, model, poly, mape, point_count=None, predicted_time=None):
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    x_min_plot = max(0, X.min() - 50)
    x_max_plot = X.max() + 50
    X_smooth = np.linspace(x_min_plot, x_max_plot, 300).reshape(-1, 1)
    
    X_smooth_poly = poly.transform(X_smooth)
    y_pred_smooth = model.predict(X_smooth_poly)
    y_theory = calculate_theory_time(X_smooth.flatten())

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)

    # Actual data points
    ax.scatter(X, y, color='#1f77b4', s=55, alpha=0.7, 
               label='Actual Data', zorder=3)
    
    # Prediction curve
    ax.plot(X_smooth, y_pred_smooth, color='#d62728', linewidth=3, 
            label='Prediction Curve', zorder=2)
    
    # Theory line
    ax.plot(X_smooth, y_theory, color='#2ca02c', linewidth=2.2, linestyle='--', 
            label='Theory Line', zorder=2)
    
    # Error band
    mape_val = mape if mape is not None else 17.0
    y_upper = y_pred_smooth * (1 + mape_val / 100)
    y_lower = y_pred_smooth * (1 - mape_val / 100)
    ax.fill_between(X_smooth.flatten(), y_lower, y_upper, 
                    color='#d62728', alpha=0.12, 
                    label=f'±{mape_val:.1f}% Error Band')

    # Prediction point
    if point_count is not None and predicted_time is not None:
        ax.scatter([point_count], [predicted_time], color='#ff6b6b', s=250,
                   edgecolors='white', linewidth=2.5, zorder=6, 
                   label=f'Prediction: {point_count} pts → {predicted_time:.1f}s')
        ax.axvline(x=point_count, color='#ff6b6b', linestyle=':', alpha=0.6, linewidth=1.5)
        ax.axhline(y=predicted_time, color='#ff6b6b', linestyle=':', alpha=0.6, linewidth=1.5)

    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.92, edgecolor='#ccc')
    
    ax.set_xlabel('Point Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('📊 Manhour Prediction Chart', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, linestyle='--')
    
    # Axis range
    x_max = X.max() * 1.15
    y_max = max(y.max(), y_theory.max(), y_pred_smooth.max()) * 1.2
    
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    # Smart tick settings
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
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    return fig

# ============================================================
# 预测函数
# ============================================================
def predict_time(point_count):
    if st.session_state.model_trained and st.session_state.model is not None:
        X_input = np.array([[point_count]])
        X_input_poly = st.session_state.poly.transform(X_input)
        predicted = st.session_state.model.predict(X_input_poly)[0]
        theory = calculate_theory_time(point_count)
        deviation_pct = (predicted - theory) / theory * 100
        return {
            "predicted": predicted,
            "theory": theory,
            "deviation_pct": deviation_pct,
            "mape": st.session_state.mape if st.session_state.mape is not None else 17.0,
            "r2": st.session_state.r2,
            "mae": st.session_state.mae
        }
    return None

# ============================================================
# AI对话
# ============================================================
def chat_with_ai(user_message, prediction_result=None):
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

        system_prompt = f"""You are an SMT/DIP production line manhour prediction data analysis expert with extensive production line experience.

The user entered {point_count} points, and the model predicts {p:.2f} seconds.
The theoretical standard time is {theory:.2f} seconds, with a deviation of {dev:+.1f}%, which is {status} (normal error range ±{mape_val:.1f}%).

Please strictly follow this format:
1. Model predicted time: {p:.2f} seconds.
2. Theoretical standard time: {theory:.2f} seconds.
3. Deviation: {dev:+.1f}%, {status}.
4. Briefly analyze the reasons."""

        user_message = f"The user entered {point_count} points, please analyze the prediction result."
    else:
        system_prompt = "You are an SMT/DIP production line manhour prediction data analysis expert. Please prompt the user to enter a specific point count for prediction analysis."

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

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.poly = None
    st.session_state.mape = None
    st.session_state.df = None
    st.session_state.r2 = None
    st.session_state.mae = None
    
if "upload_authorized" not in st.session_state:
    st.session_state.upload_authorized = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_prediction_result" not in st.session_state:
    st.session_state.last_prediction_result = None

# ============================================================
# 自动加载并训练模型
# ============================================================
if not st.session_state.model_trained:
    saved_df = load_saved_data()
    if saved_df is not None and len(saved_df) > 0:
        model, poly, r2, mae, mape = train_prediction_model(saved_df)
        st.session_state.model_trained = True
        st.session_state.model = model
        st.session_state.poly = poly
        st.session_state.r2 = r2
        st.session_state.mae = mae
        st.session_state.mape = mape
        st.session_state.df = saved_df

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Data Management")
    if st.session_state.model_trained and st.session_state.df is not None:
        st.success(f"✅ Current data: {len(st.session_state.df)} rows")
    else:
        st.warning("⚠️ No data available")

    st.markdown("---")
    st.markdown("#### 🔒 Admin Verification")
    admin_pwd = st.text_input("Upload Password", type="password", key="admin_pwd")
    if st.button("Verify & Upload", use_container_width=True):
        if hash_password(admin_pwd) == hash_password("admin123"):
            st.session_state.upload_authorized = True
            st.success("Verification successful, please upload data")
        else:
            st.session_state.upload_authorized = False
            st.error("Wrong password")

    if st.session_state.upload_authorized:
        st.markdown("---")
        st.markdown("#### 📤 Upload Data")
        st.caption("Supports multi-column Excel files, auto-detects '单板点数' and '实际工时/s' columns")
        uploaded_file = st.file_uploader("Select Excel file", type=["xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file:
            df_raw = pd.read_excel(uploaded_file)
            
            point_col, actual_col, theory_col = get_column_mapping(df_raw)
            
            if point_col is not None and actual_col is not None:
                df = df_raw[[point_col, actual_col]].copy()
                df.columns = ['点位数', '实际工时']
                df = df.dropna()
                
                model, poly, r2, mae, mape = train_prediction_model(df)
                st.session_state.model_trained = True
                st.session_state.model = model
                st.session_state.poly = poly
                st.session_state.r2 = r2
                st.session_state.mae = mae
                st.session_state.mape = mape
                st.session_state.df = df
                save_data(df)
                st.success(f"✅ Data saved, {len(df)} rows")
                st.info(f"Detected: '{point_col}' → Point Count, '{actual_col}' → Actual Time")
                st.balloons()
                st.rerun()
            else:
                st.error(f"❌ Cannot find '单板点数' or '实际工时/s' columns. Current columns: {df_raw.columns.tolist()}")

    st.markdown("---")
    with st.expander("📋 Sample Data Format"):
        st.markdown("""
        | Line | Points | Actual Time/s | Theory Time/s |
        |------|--------|---------------|---------------|
        | L1 | 71 | 10.22 | 9.67 |
        | L2 | 68 | 57.60 | 68.40 |
        """)
        st.caption("System auto-detects '单板点数' and '实际工时/s' columns")

    if os.path.exists(DATA_FILE):
        mod_time = os.path.getmtime(DATA_FILE)
        update_time = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        st.markdown("---")
        st.caption(f"📅 Data updated: {update_time}")

# ============================================================
# 标题
# ============================================================
st.markdown("<h1 style='text-align: center;'>⚙️ Manhour Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

# ============================================================
# 左右两栏
# ============================================================
left_col, right_col = st.columns(2, gap="large")

# ============================================================
# 左侧：模型评估 + 对比图
# ============================================================
with left_col:
    if st.session_state.model_trained and st.session_state.df is not None:
        with st.container():
            st.markdown("### 📊 Model Evaluation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{st.session_state.r2:.3f}" if st.session_state.r2 is not None else "--")
            with col2:
                mae_value = f"{st.session_state.mae:.2f}" if st.session_state.mae is not None else "--"
                st.metric("MAE", mae_value, help="Mean Absolute Error (seconds)")
            with col3:
                mape_value = f"{st.session_state.mape:.1f}%" if st.session_state.mape is not None else "--"
                st.metric("MAPE", mape_value, help="Mean Absolute Percentage Error")

        with st.container():
            st.markdown("### 📈 Comparison Chart")
            
            plot_placeholder = st.empty()
            
            if st.session_state.last_prediction is not None:
                fig = plot_chart(
                    st.session_state.df,
                    st.session_state.model,
                    st.session_state.poly,
                    st.session_state.mape,
                    point_count=st.session_state.last_prediction["point_count"],
                    predicted_time=st.session_state.last_prediction["predicted"]
                )
                plot_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                fig = plot_chart(
                    st.session_state.df, 
                    st.session_state.model, 
                    st.session_state.poly, 
                    st.session_state.mape
                )
                plot_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
    else:
        st.info("👈 Please upload data in the sidebar")

# ============================================================
# 右侧：AI智能体对话
# ============================================================
with right_col:
    st.markdown("### 🎯 Manhour Prediction Assistant")
    st.caption("Enter point count, AI estimates manhour | Based on historical data")

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
            status_text = "⚠️ Out of Range"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%); 
                    padding: 0.8rem 1rem; 
                    border-radius: 10px; 
                    border-left: 4px solid #4a6cf7;
                    margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.8rem; color: #888;">Last Prediction</span>
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

        if numbers and st.session_state.model_trained:
            point_count = int(numbers[0])
            pred_data = predict_time(point_count)
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

        with st.spinner("AI is analyzing..."):
            response = chat_with_ai(user_input, prediction_result)

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
                    st.write(f"- {h['point_count']} pts: {h['predicted']:.1f}s (deviation {h['deviation_pct']:+.1f}%)")
            else:
                st.write("No prediction records")
