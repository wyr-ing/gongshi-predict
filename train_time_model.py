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
    page_title="工时预测系统",
    page_icon="⚙️",
    layout="wide"
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
            point_col, actual_col, _ = get_column_mapping(df)
            if point_col is not None and actual_col is not None:
                df_clean = df[[point_col, actual_col]].copy()
                df_clean.columns = ['点位数', '实际工时']
                return df_clean
        except:
            pass
    return None

# ============================================================
# 训练模型
# ============================================================
def train_model(df):
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
    residuals = y - y_pred
    return model, poly, r2, mae, mape, residuals

# ============================================================
# 对比图
# ============================================================
def plot_chart(df, model, poly, mape, point_count=None, predicted_time=None):
    X = df[['点位数']].values
    y = df['实际工时'].values
    X_smooth = np.linspace(X.min() - 50, X.max() + 50, 300).reshape(-1, 1)
    X_smooth_poly = poly.transform(X_smooth)
    y_smooth = model.predict(X_smooth_poly)
    theory_a, theory_b = 0.0362, 0.5
    y_theory = theory_a * X_smooth.flatten() + theory_b

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)

    # 散点
    ax.scatter(X, y, color='#1f77b4', s=55, alpha=0.6, label='实际工时数据', zorder=3)
    
    # 拟合曲线
    ax.plot(X_smooth, y_smooth, color='#d62728', linewidth=2.8, label='实际拟合曲线', zorder=2)
    
    # 理论直线
    ax.plot(X_smooth, y_theory, color='#2ca02c', linewidth=2.2, linestyle='--', label='理论直线', zorder=2)
    
    # 误差带
    mape_val = mape if mape is not None else 17.0
    y_upper = y_smooth * (1 + mape_val / 100)
    y_lower = y_smooth * (1 - mape_val / 100)
    ax.fill_between(X_smooth.flatten(), y_lower, y_upper, color='#d62728', alpha=0.15, label=f'±{mape_val:.1f}% 误差带')

    # 预测点标记
    if point_count is not None and predicted_time is not None:
        ax.scatter([point_count], [predicted_time], color='red', s=200,
                   edgecolors='white', linewidth=2, zorder=6, 
                   label=f'当前预测: {point_count}点 → {predicted_time:.1f}s')
        ax.axvline(x=point_count, color='red', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.axhline(y=predicted_time, color='red', linestyle=':', alpha=0.6, linewidth=1.2)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='gray')
    
    ax.set_xlabel('点位数（个）', fontsize=12)
    ax.set_ylabel('工时（秒）', fontsize=12)
    ax.set_title('工时预测对比图', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 坐标轴范围
    x_min = 0
    x_max = X.max() * 1.1
    ax.set_xlim(x_min, x_max)
    
    y_max = max(y.max(), y_theory.max(), y_smooth.max()) * 1.2
    ax.set_ylim(0, y_max)
    
    # 刻度设置
    x_ticks = np.arange(0, x_max + 50, 50)
    ax.set_xticks(x_ticks)
    
    y_max_rounded = int(np.ceil(y_max / 50)) * 50
    y_ticks = np.arange(0, y_max_rounded + 50, 50)
    ax.set_yticks(y_ticks)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    return fig

# ============================================================
# 预测函数
# ============================================================
def predict_by_model(point_count):
    if st.session_state.model_trained and st.session_state.model is not None:
        X_input = np.array([[point_count]])
        X_input_poly = st.session_state.poly.transform(X_input)
        predicted = st.session_state.model.predict(X_input_poly)[0]
        theory = 0.0362 * point_count + 0.5
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
            status = "在正常范围内"
        else:
            status = "超出正常范围"

        system_prompt = f"""你是SMT/DIP产线工时预测数据分析专家，有丰富的产线经验。

用户输入了点位数 {point_count}，模型预测工时 {p:.2f} 秒。
理论标准工时 {theory:.2f} 秒，偏差 {dev:+.1f}%，{status}（正常误差范围 ±{mape_val:.1f}%）。

请严格按以下格式输出：
1.模型预测工时 {p:.2f} 秒。
2.理论标准工时 {theory:.2f} 秒，
3.偏差 {dev:+.1f}%，{status}。
4.然后简要分析原因。"""

        user_message = f"用户输入点位数{point_count}，请分析预测结果。"
    else:
        system_prompt = "你是SMT/DIP产线工时预测数据分析专家。请提示用户输入具体点位数，以便进行预测分析。"

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

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.poly = None
    st.session_state.mape = None
    st.session_state.df = None
    st.session_state.r2 = None
    st.session_state.mae = None
    st.session_state.residuals = None
    
if "upload_authorized" not in st.session_state:
    st.session_state.upload_authorized = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_prediction_result" not in st.session_state:
    st.session_state.last_prediction_result = None

# ============================================================
# 自动加载并训练数据
# ============================================================
if not st.session_state.model_trained:
    saved_df = load_saved_data()
    if saved_df is not None and len(saved_df) > 0:
        model, poly, r2, mae, mape, residuals = train_model(saved_df)
        st.session_state.model_trained = True
        st.session_state.model = model
        st.session_state.poly = poly
        st.session_state.r2 = r2
        st.session_state.mae = mae
        st.session_state.mape = mape
        st.session_state.df = saved_df
        st.session_state.residuals = residuals

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ 数据管理")
    if st.session_state.model_trained and st.session_state.df is not None:
        st.success(f"✅ 当前数据：{len(st.session_state.df)} 行")
    else:
        st.warning("⚠️ 暂无数据")

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
        st.markdown("#### 📤 上传数据")
        st.caption("支持包含多列的Excel文件，自动识别'单板点数'和'实际工时/s'列")
        uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file:
            df_raw = pd.read_excel(uploaded_file)
            
            point_col, actual_col, _ = get_column_mapping(df_raw)
            
            if point_col is not None and actual_col is not None:
                df = df_raw[[point_col, actual_col]].copy()
                df.columns = ['点位数', '实际工时']
                df = df.dropna()
                
                model, poly, r2, mae, mape, residuals = train_model(df)
                st.session_state.model_trained = True
                st.session_state.model = model
                st.session_state.poly = poly
                st.session_state.r2 = r2
                st.session_state.mae = mae
                st.session_state.mape = mape
                st.session_state.df = df
                st.session_state.residuals = residuals
                save_data(df)
                st.success(f"✅ 数据已保存，共 {len(df)} 行")
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

    if os.path.exists(DATA_FILE):
        mod_time = os.path.getmtime(DATA_FILE)
        update_time = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
        st.markdown("---")
        st.caption(f"📅 数据更新：{update_time}")

# ============================================================
# 标题
# ============================================================
st.markdown("<h1 style='text-align: center;'>⚙️ 工时预测系统</h1>", unsafe_allow_html=True)
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
            st.markdown("### 📊 模型评估")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{st.session_state.r2:.3f}" if st.session_state.r2 is not None else "--")
            with col2:
                mae_value = f"{st.session_state.mae:.2f}" if st.session_state.mae is not None else "--"
                st.metric("MAE", mae_value, help="平均绝对误差（秒）")
            with col3:
                mape_value = f"{st.session_state.mape:.1f}%" if st.session_state.mape is not None else "--"
                st.metric("MAPE", mape_value, help="平均绝对百分比误差")

        with st.container():
            st.markdown("### 📈 对比图")
            
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
                fig = plot_chart(st.session_state.df, st.session_state.model, st.session_state.poly, st.session_state.mape)
                plot_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
    else:
        st.info("👈 请在左侧菜单上传数据")

# ============================================================
# 右侧：AI智能体对话
# ============================================================
with right_col:
    st.markdown("### 🎯 工时预测小助手")
    st.caption("输入点位数，AI估算工时 | 基于历史数据预测新订单")

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
                    <span style="font-size: 0.8rem; color: #888;">上次预测</span>
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

        if numbers and st.session_state.model_trained:
            point_count = int(numbers[0])
            pred_data = predict_by_model(point_count)
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

        with st.spinner("智能体分析中..."):
            response = chat_with_ai(user_input, prediction_result)

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
                    st.write(f"- {h['point_count']}点: {h['predicted']:.1f}秒 (偏差{h['deviation_pct']:+.1f}%)")
            else:
                st.write("暂无预测记录")
    subprocess.run(["streamlit", "run", __file__, "--server.port", "8501"])
