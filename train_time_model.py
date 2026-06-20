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
    page_title="SMT工时预测系统",
    page_icon="⚙️",
    layout="wide"
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置
# ============================================================
DATA_FILE_SMT = "smt_data.xlsx"
CHAT_HISTORY_FILE = "chat_history.json"
API_KEY = "sk-fvxkdwbhjcokafftooavzvedrlmmrffotehplsnfnjupogqb"
BASE_URL = "https://api.siliconflow.cn/v1"

# ============================================================
# 屏幕自适应
# ============================================================
def get_screen_size():
    try:
        screen_width = st.session_state.get('screen_width', 1200)
    except:
        screen_width = 1200
    
    if screen_width < 768:
        return {'fig_width': 6, 'fig_height': 4.5, 'font_size': 8, 'title_size': 10, 
                'legend_size': 7, 'marker_size': 30, 'tick_size': 7}
    elif screen_width < 1024:
        return {'fig_width': 8, 'fig_height': 5.5, 'font_size': 9, 'title_size': 12, 
                'legend_size': 8, 'marker_size': 40, 'tick_size': 8}
    elif screen_width < 1366:
        return {'fig_width': 10, 'fig_height': 6, 'font_size': 10, 'title_size': 13, 
                'legend_size': 9, 'marker_size': 45, 'tick_size': 9}
    else:
        return {'fig_width': 12, 'fig_height': 6.5, 'font_size': 11, 'title_size': 14, 
                'legend_size': 9.5, 'marker_size': 55, 'tick_size': 10}

# ============================================================
# 聊天记录持久化
# ============================================================
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_chat_history(messages):
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except:
        pass

def clear_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
    return []

# ============================================================
# 数据加载/保存
# ============================================================
def load_smt_data():
    if os.path.exists(DATA_FILE_SMT):
        try:
            df = pd.read_excel(DATA_FILE_SMT)
            required_cols = ['单板点数', '标准工时']
            for col in required_cols:
                if col not in df.columns:
                    return None
            return df
        except:
            return None
    return None

def save_smt_data(df):
    df.to_excel(DATA_FILE_SMT, index=False)

# ============================================================
# 训练模型（单板点数 → 标准工时）
# ============================================================
def train_models(df):
    models = {}
    
    df_clean = df.dropna(subset=['单板点数', '标准工时'])
    df_clean = df_clean[(df_clean['单板点数'] > 0) & (df_clean['标准工时'] > 0)]
    
    if len(df_clean) >= 3:
        X = df_clean[['单板点数']].values
        y = df_clean['标准工时'].values
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        models['smt'] = {
            'model': model,
            'poly': poly,
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100,
            'data': df_clean,
            'sample_count': len(df_clean),
            'x_min': df_clean['单板点数'].min(),
            'x_max': df_clean['单板点数'].max(),
            'y_min': df_clean['标准工时'].min(),
            'y_max': df_clean['标准工时'].max()
        }
    
    return models

# ============================================================
# 预测函数
# ============================================================
def predict_time(points, models=None):
    if models is None or 'smt' not in models:
        return None
    
    X = np.array([[points]])
    X_poly = models['smt']['poly'].transform(X)
    pred = models['smt']['model'].predict(X_poly)[0]
    
    return {
        'points': points,
        'time': pred,
        'r2': models['smt']['r2'],
        'mape': models['smt']['mape'],
        'mae': models['smt']['mae']
    }

# ============================================================
# 绘图
# ============================================================
def plot_chart(models, points=None, predicted=None, line_type="SMT"):
    
    screen = get_screen_size()
    
    fig, ax = plt.subplots(figsize=(screen['fig_width'], screen['fig_height']), dpi=100)
    
    if 'smt' in models:
        model_info = models['smt']
        df = model_info['data']
        x = df['单板点数']
        y = df['标准工时']
        
        ax.scatter(x, y, color='#1f77b4', s=screen['marker_size'], alpha=0.6, 
                  label='Data Points')
        
        if len(x) > 1:
            x_smooth = np.linspace(max(0, x.min() - 10), x.max() + 10, 100).reshape(-1, 1)
            x_poly = model_info['poly'].transform(x_smooth)
            y_smooth = model_info['model'].predict(x_poly)
            ax.plot(x_smooth, y_smooth, color='#d62728', linewidth=2, 
                   label=f'Fit Curve (R²={model_info["r2"]:.3f})')
    
    if predicted is not None and points is not None:
        ax.scatter([points], [predicted['time']], color='#ff6b6b', 
                  s=screen['marker_size'] * 3, edgecolors='white', linewidth=2, zorder=6,
                  label=f'Prediction: {points} pts → {predicted["time"]:.1f}s')
    
    ax.set_xlabel('Points (pts)', fontsize=screen['font_size'], fontweight='bold')
    ax.set_ylabel('Time (s)', fontsize=screen['font_size'], fontweight='bold')
    ax.set_title(f'📊 {line_type} Manhour Prediction Chart', 
                 fontsize=screen['title_size'], fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=screen['legend_size'])
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    return fig

# ============================================================
# AI对话 - 直接从对比图（模型数据）获取信息，不依赖预测模块
# ============================================================
def chat_with_ai(user_message, models=None, line_type="SMT"):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # 检查是否有模型数据
    if models is not None and 'smt' in models:
        model_info = models['smt']
        
        # 提取图表数据（对比图的核心信息）
        df = model_info['data']
        x_data = df['单板点数'].tolist()
        y_data = df['标准工时'].tolist()
        
        # 计算数据分布统计
        data_summary = f"""
【对比图（散点图与拟合曲线）数据摘要】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 数据概况
  • 数据点数量：{model_info['sample_count']} 个
  • 点数范围：{model_info['x_min']} ~ {model_info['x_max']} 点
  • 工时范围：{model_info['y_min']:.1f} ~ {model_info['y_max']:.1f} 秒

📈 拟合曲线（二次多项式回归）
  • R²（决定系数）：{model_info['r2']:.3f}
  • MAPE（平均绝对百分比误差）：{model_info['mape']:.1f}%
  • MAE（平均绝对误差）：{model_info['mae']:.2f} 秒

📌 数据分布特征
  • 点数平均值：{np.mean(x_data):.1f} 点
  • 工时平均值：{np.mean(y_data):.1f} 秒
  • 点数中位数：{np.median(x_data):.1f} 点
  • 工时中位数：{np.median(y_data):.1f} 秒

📐 拟合曲线公式（二次多项式）
  • 根据二次多项式回归拟合，曲线平滑上升/下降
  • 反映了点数与工时之间的整体变化趋势
        """
        
        system_prompt = f"""你是SMT工时预测数据分析助手。

【核心规则 - 必须100%遵守】：
1. 你只能基于上面的【对比图数据摘要】中的数据进行分析
2. 这些数据直接来自"散点图与拟合曲线"图表
3. 不要说任何不在这个数据摘要中的内容
4. 禁止使用："行业基准"、"行业标准"、"通常"、"一般"、"应该"等词汇
5. 禁止给出"优化建议"、"改进建议"等超出数据范围的内容
6. 所有分析必须基于图表数据，从数据本身出发

{data_summary}

请根据用户的问题进行分析。如果用户输入了具体点数（如 100），你可以：
1. 参考图表中附近的数据点
2. 根据拟合曲线趋势给出估算

请按以下格式输出（严格遵循）：

【图表数据解读】
（基于数据摘要，说明数据的整体分布特征）

【针对用户问题的分析】
（根据用户输入的点数或问题，结合图表数据给出具体分析）

【数据可信度评估】
（基于R²、MAPE、MAE三个指标说明模型拟合程度）

【总结】
（基于图表数据的总结性结论）"""
        
    else:
        system_prompt = f"""你是SMT工时预测数据分析助手。
当前还没有上传数据，请提示用户先在左侧上传Excel数据文件（包含：单板点数、标准工时两列）。"""

    messages = [{"role": "system", "content": system_prompt}]
    chat_history = st.session_state.messages[-20:] if st.session_state.messages else []
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1500
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
    st.session_state.messages = load_chat_history()

for key in ['models', 'df_raw', 'last_prediction', 'last_prediction_result']:
    if key not in st.session_state:
        st.session_state[key] = None

if "upload_authorized" not in st.session_state:
    st.session_state.upload_authorized = False

if "screen_width" not in st.session_state:
    st.session_state.screen_width = 1200

# ============================================================
# 加载数据
# ============================================================
smt_df = load_smt_data()
if smt_df is not None and len(smt_df) > 0:
    st.session_state.df_raw = smt_df
    st.session_state.models = train_models(smt_df)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ 数据管理")
    
    if st.session_state.models is not None and 'smt' in st.session_state.models:
        sample_count = st.session_state.models['smt']['sample_count']
        st.success(f"✅ 数据行数: {sample_count}")
    else:
        st.warning("⚠️ 暂无数据，请上传")

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
        st.markdown(f"#### 📤 上传SMT数据")
        st.caption("Excel需包含：单板点数、标准工时")
        uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file:
            df_raw = pd.read_excel(uploaded_file)
            required = ['单板点数', '标准工时']
            missing = [c for c in required if c not in df_raw.columns]
            if not missing:
                df_raw = df_raw.dropna(subset=['单板点数', '标准工时'])
                save_smt_data(df_raw)
                st.session_state.models = train_models(df_raw)
                st.session_state.df_raw = df_raw
                st.success(f"✅ 已上传，共 {len(df_raw)} 行")
                st.rerun()
            else:
                st.error(f"❌ 缺少列：{missing}")

    st.markdown("---")
    with st.expander("📋 数据格式要求"):
        st.markdown("""
        | 列名 | 说明 |
        |------|------|
        | 单板点数 | 整数 |
        | 标准工时 | 浮点数（秒） |
        """)

# ============================================================
# 主界面
# ============================================================
st.markdown("<h1 style='text-align: center;'>⚙️ SMT 工时预测系统</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

# ============================================================
# 第一行：左右两栏（模型评估+对比图 | 工时预测）
# ============================================================
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    if st.session_state.models is not None and 'smt' in st.session_state.models:
        models = st.session_state.models
        model_info = models['smt']
        
        with st.container():
            st.markdown("### 📊 模型评估")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R²", f"{model_info['r2']:.3f}")
            with col2:
                st.metric("MAPE", f"{model_info['mape']:.1f}%")
            with col3:
                st.metric("MAE", f"{model_info['mae']:.2f}s")
        
        with st.container():
            st.markdown("### 📈 散点图与拟合曲线")
            plot_placeholder = st.empty()
            
            if st.session_state.last_prediction_result is not None:
                last = st.session_state.last_prediction_result
                fig = plot_chart(
                    models,
                    points=last.get('points'),
                    predicted=last
                )
                plot_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                fig = plot_chart(models)
                plot_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
    else:
        st.info("👈 请上传数据")

with right_col:
    with st.container():
        st.markdown("### 🎯 SMT工时预测")
        
        points = st.number_input("单板点数", min_value=0, value=100, step=10, key="points_input")
        
        if st.button("🚀 预测", use_container_width=True, key="predict_btn"):
            if st.session_state.models is not None and 'smt' in st.session_state.models:
                result = predict_time(points, st.session_state.models)
                if result:
                    st.session_state.last_prediction_result = result
                    st.session_state.last_prediction = {
                        'points': points,
                        'time': result['time']
                    }
                    st.rerun()
            else:
                st.error("请先上传数据")
        
        if st.session_state.last_prediction_result is not None:
            result = st.session_state.last_prediction_result
            lower_bound = result['time'] * (1 - result['mape'] / 100)
            upper_bound = result['time'] * (1 + result['mape'] / 100)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); 
                        padding: 1rem 1.5rem; border-radius: 12px; border-left: 5px solid #2e7d32; margin-top: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="color: #555; font-size: 0.85rem;">📌 预测结果</span>
                        <div style="font-size: 2rem; font-weight: 700; color: #2e7d32;">
                            {result['time']:.1f}s
                        </div>
                        <span style="color: #666; font-size: 0.8rem;">标准工时</span>
                        <div style="font-size: 0.75rem; color: #888; margin-top: 2px;">
                            范围: {lower_bound:.1f} ~ {upper_bound:.1f} s
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #666; font-size: 0.8rem;">点位数</span>
                        <div style="font-size: 1.2rem; font-weight: 600;">{result['points']}</div>
                        <span style="color: #888; font-size: 0.7rem;">R²={result.get('r2', 0):.3f}</span>
                    </div>
                </div>
                <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px dashed #a5d6a7; display: flex; gap: 1rem; flex-wrap: wrap;">
                    <span style="color: #555; font-size: 0.75rem;">📊 MAPE: {result.get('mape', 0):.1f}%</span>
                    <span style="color: #555; font-size: 0.75rem;">📏 MAE: {result.get('mae', 0):.2f}s</span>
                    <span style="color: #555; font-size: 0.75rem;">📈 拟合度: {result.get('r2', 0):.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# 第二行：AI分析 - 直接从对比图获取数据
# ============================================================
st.markdown("---")

with st.container():
    st.markdown("<h3 style='text-align: center;'>💬 AI 分 析</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 0.85rem;'>基于对比图（散点图与拟合曲线）的数据分析</p>", unsafe_allow_html=True)
    
    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            elif msg["role"] == "assistant":
                st.chat_message("assistant").write(msg["content"])
    
    col_input, col_btn = st.columns([6, 1])
    
    with col_input:
        user_input = st.chat_input("输入点数（如 100）或提问...", key="ai_chat_input")
    
    with col_btn:
        if st.button("🗑️ 清空对话", use_container_width=True, key="clear_chat_btn"):
            st.session_state.messages = clear_chat_history()
            st.rerun()
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_chat_history(st.session_state.messages)
        
        # 直接从模型数据获取，不依赖预测模块
        with st.spinner("分析中..."):
            response = chat_with_ai(user_input, st.session_state.models)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_chat_history(st.session_state.messages)
        
        st.rerun()
