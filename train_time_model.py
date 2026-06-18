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
API_KEY = "sk-fvxkdwbhjcokafftooavzvedrlmmrffotehplsnfnjupogqb"
BASE_URL = "https://api.siliconflow.cn/v1"
HISTORY_FILE = "prediction_history.json"

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
# 数据加载/保存
# ============================================================
def load_smt_data():
    if os.path.exists(DATA_FILE_SMT):
        try:
            df = pd.read_excel(DATA_FILE_SMT)
            required_cols = ['物料编码', '单面/双面', 'TOP单板点数', 'BOT单板点数', 'TOP标准工时', 'BOT标准工时']
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
# 训练模型（分别训练单面和双面）
# ============================================================
def train_models(df):
    models = {}
    
    # 单面板模型：TOP点数 → TOP标准工时
    single_df = df[df['单面/双面'] == '单面'].copy()
    if len(single_df) >= 3:
        X = single_df[['TOP单板点数']].values
        y = single_df['TOP标准工时'].values
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        models['single'] = {
            'model': model,
            'poly': poly,
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100,
            'data': single_df,
            'sample_count': len(single_df)
        }
    
    # 双面板模型：总点数 → 总工时（TOP+BOT）
    double_df = df[df['单面/双面'] == '双面'].copy()
    if len(double_df) >= 3:
        double_df['总点数'] = double_df['TOP单板点数'] + double_df['BOT单板点数']
        double_df['总工时'] = double_df['TOP标准工时'] + double_df['BOT标准工时']
        
        X = double_df[['总点数']].values
        y = double_df['总工时'].values
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        models['double'] = {
            'model': model,
            'poly': poly,
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100,
            'data': double_df,
            'sample_count': len(double_df)
        }
    
    return models

# ============================================================
# 预测函数
# ============================================================
def predict_time(board_type, top_points, bot_points=0, models=None):
    """预测工时"""
    if models is None:
        return None
    
    if board_type == '单面':
        if 'single' not in models:
            return None
        X = np.array([[top_points]])
        X_poly = models['single']['poly'].transform(X)
        pred = models['single']['model'].predict(X_poly)[0]
        return {
            'top_time': pred,
            'bot_time': 0,
            'total_time': pred,
            'model_type': '单面',
            'r2': models['single']['r2'],
            'mape': models['single']['mape']
        }
    
    elif board_type == '双面':
        if 'double' not in models:
            return None
        total_points = top_points + bot_points
        X = np.array([[total_points]])
        X_poly = models['double']['poly'].transform(X)
        pred = models['double']['model'].predict(X_poly)[0]
        
        # 按点数比例分配TOP和BOT工时
        if total_points > 0:
            top_ratio = top_points / total_points
            bot_ratio = bot_points / total_points
            top_time = pred * top_ratio
            bot_time = pred * bot_ratio
        else:
            top_time = 0
            bot_time = 0
        
        return {
            'top_time': top_time,
            'bot_time': bot_time,
            'total_time': pred,
            'model_type': '双面',
            'r2': models['double']['r2'],
            'mape': models['double']['mape']
        }
    
    return None

# ============================================================
# 绘图（图例和标题使用英文）
# ============================================================
def plot_chart(models, board_type=None, top_points=None, bot_points=None, 
               predicted=None, line_type="SMT"):
    
    screen = get_screen_size()
    
    fig, ax = plt.subplots(figsize=(screen['fig_width'], screen['fig_height']), dpi=100)
    
    colors = {'单面': '#1f77b4', '双面': '#d62728'}
    
    for model_name, model_info in models.items():
        if model_name == 'single':
            df = model_info['data']
            x = df['TOP单板点数']
            y = df['TOP标准工时']
            ax.scatter(x, y, color=colors['单面'], s=screen['marker_size'], alpha=0.6, 
                      label='Single Side Data')
            
            if len(x) > 1:
                x_smooth = np.linspace(x.min() - 10, x.max() + 10, 100).reshape(-1, 1)
                x_poly = model_info['poly'].transform(x_smooth)
                y_smooth = model_info['model'].predict(x_poly)
                ax.plot(x_smooth, y_smooth, color=colors['单面'], linewidth=2, 
                       label=f'Single Fit (R²={model_info["r2"]:.3f})')
        
        elif model_name == 'double':
            df = model_info['data']
            x = df['总点数']
            y = df['总工时']
            ax.scatter(x, y, color=colors['双面'], s=screen['marker_size'], alpha=0.6, 
                      label='Double Side Data')
            
            if len(x) > 1:
                x_smooth = np.linspace(x.min() - 20, x.max() + 20, 100).reshape(-1, 1)
                x_poly = model_info['poly'].transform(x_smooth)
                y_smooth = model_info['model'].predict(x_poly)
                ax.plot(x_smooth, y_smooth, color=colors['双面'], linewidth=2, 
                       label=f'Double Fit (R²={model_info["r2"]:.3f})')
    
    if predicted is not None and top_points is not None:
        if board_type == '单面':
            ax.scatter([top_points], [predicted['total_time']], color='#ff6b6b', 
                      s=screen['marker_size'] * 3, edgecolors='white', linewidth=2, zorder=6,
                      label=f'Prediction: {top_points}pts → {predicted["total_time"]:.1f}s')
        elif board_type == '双面' and bot_points is not None:
            total = top_points + bot_points
            ax.scatter([total], [predicted['total_time']], color='#ff6b6b', 
                      s=screen['marker_size'] * 3, edgecolors='white', linewidth=2, zorder=6,
                      label=f'Prediction: {top_points}+{bot_points}pts → {predicted["total_time"]:.1f}s')
    
    ax.set_xlabel('点数', fontsize=screen['font_size'], fontweight='bold')
    ax.set_ylabel('工时 (秒)', fontsize=screen['font_size'], fontweight='bold')
    ax.set_title(f'📊 {line_type} Manhour Prediction Chart', 
                 fontsize=screen['title_size'], fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=screen['legend_size'])
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    return fig

# ============================================================
# AI对话
# ============================================================
def chat_with_ai(user_message, prediction_result=None, line_type="SMT"):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    if prediction_result:
        p = prediction_result
        board_type = p['model_type']
        
        if board_type == '单面':
            msg = f"""用户预测单面板：
- TOP面点数: {p['top_points']}
- 预测TOP标准工时: {p['top_time']:.2f}秒
- 模型R²: {p['r2']:.3f}
- 模型MAPE: {p['mape']:.1f}%"""
        else:
            msg = f"""用户预测双面板：
- TOP面点数: {p['top_points']}
- BOT面点数: {p['bot_points']}
- 总点数: {p['top_points'] + p['bot_points']}
- 预测TOP标准工时: {p['top_time']:.2f}秒
- 预测BOT标准工时: {p['bot_time']:.2f}秒
- 预测总工时: {p['total_time']:.2f}秒
- 模型R²: {p['r2']:.3f}
- 模型MAPE: {p['mape']:.1f}%"""
        
        system_prompt = f"""你是{line_type}产线工时预测数据分析专家。

{msg}

请分析这个预测结果是否合理，给出专业建议。"""

        user_message = "请分析预测结果"
    else:
        system_prompt = f"你是{line_type}产线工时预测数据分析专家。请提示用户输入板类型和点数。"

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
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                st.session_state.prediction_history = json.load(f)
        except:
            st.session_state.prediction_history = []
    else:
        st.session_state.prediction_history = []

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
    
    if st.session_state.models is not None:
        models = st.session_state.models
        single_count = models['single']['sample_count'] if 'single' in models else 0
        double_count = models['double']['sample_count'] if 'double' in models else 0
        st.success(f"✅ 单面: {single_count} 行 | 双面: {double_count} 行")
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
        st.caption("Excel需包含：物料编码、单面/双面、TOP/BOT点数、TOP/BOT标准工时")
        uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx", "xls"], label_visibility="collapsed")
        if uploaded_file:
            df_raw = pd.read_excel(uploaded_file)
            required = ['物料编码', '单面/双面', 'TOP单板点数', 'BOT单板点数', 'TOP标准工时', 'BOT标准工时']
            missing = [c for c in required if c not in df_raw.columns]
            if not missing:
                df_raw = df_raw.dropna(subset=['物料编码', '单面/双面'])
                df_raw['TOP单板点数'] = df_raw['TOP单板点数'].fillna(0)
                df_raw['BOT单板点数'] = df_raw['BOT单板点数'].fillna(0)
                df_raw['TOP标准工时'] = df_raw['TOP标准工时'].fillna(0)
                df_raw['BOT标准工时'] = df_raw['BOT标准工时'].fillna(0)
                
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
        | 物料编码 | 唯一标识 |
        | 单面/双面 | 单面 或 双面 |
        | TOP单板点数 | TOP面点数 |
        | BOT单板点数 | BOT面点数 |
        | TOP标准工时 | TOP面工时 |
        | BOT标准工时 | BOT面工时 |
        """)

# ============================================================
# 主界面
# ============================================================
st.markdown("<h1 style='text-align: center;'>⚙️ SMT 工时预测系统</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

# ============================================================
# 左右两栏
# ============================================================
left_col, right_col = st.columns(2, gap="large")

# ============================================================
# 左侧：模型评估 + 图表
# ============================================================
with left_col:
    if st.session_state.models is not None:
        models = st.session_state.models
        
        with st.container():
            st.markdown("### 📊 模型评估")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'single' in models:
                    st.metric("单面 R²", f"{models['single']['r2']:.3f}", 
                              help=f"MAPE: {models['single']['mape']:.1f}%")
                else:
                    st.metric("单面 R²", "无数据")
            
            with col2:
                if 'double' in models:
                    st.metric("双面 R²", f"{models['double']['r2']:.3f}",
                              help=f"MAPE: {models['double']['mape']:.1f}%")
                else:
                    st.metric("双面 R²", "无数据")
        
        with st.container():
            st.markdown("### 📈 对比图")
            plot_placeholder = st.empty()
            
            if st.session_state.last_prediction_result is not None:
                last = st.session_state.last_prediction_result
                fig = plot_chart(
                    models,
                    board_type=last['model_type'],
                    top_points=last.get('top_points'),
                    bot_points=last.get('bot_points', 0),
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

# ============================================================
# 右侧：预测 + AI对话
# ============================================================
with right_col:
    st.markdown("### 🎯 SMT工时预测")
    
    board_type = st.radio(
        "选择板类型",
        ["单面", "双面"],
        horizontal=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_points = st.number_input("TOP面点数", min_value=0, value=100, step=10)
    with col2:
        if board_type == "双面":
            bot_points = st.number_input("BOT面点数", min_value=0, value=50, step=10)
        else:
            bot_points = 0
            st.markdown("**BOT面点数**")
            st.caption("单面板无需输入")
    
    if st.button("🚀 预测", use_container_width=True):
        if st.session_state.models is not None:
            result = predict_time(board_type, top_points, bot_points, st.session_state.models)
            if result:
                result['top_points'] = top_points
                result['bot_points'] = bot_points
                st.session_state.last_prediction_result = result
                st.session_state.last_prediction = {
                    'top_points': top_points,
                    'bot_points': bot_points,
                    'board_type': board_type,
                    'total_time': result['total_time']
                }
                st.rerun()
        else:
            st.error("请先上传数据")
    
    if st.session_state.last_prediction_result is not None:
        result = st.session_state.last_prediction_result
        
        if result['model_type'] == '单面':
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #2e7d32;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <span style="color: #555;">📌 单面板</span>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #2e7d32;">
                            {result['total_time']:.1f}s
                        </div>
                        <span style="color: #666; font-size: 0.9rem;">TOP面工时</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #666;">点位数</span>
                        <div style="font-size: 1.2rem; font-weight: 600;">{top_points}</div>
                        <span style="color: #888; font-size: 0.8rem;">R²={result.get('r2', 0):.3f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); 
                        padding: 1rem; border-radius: 10px; border-left: 4px solid #1565c0;">
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div>
                        <span style="color: #555;">📌 双面板</span>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #1565c0;">
                            {result['total_time']:.1f}s
                        </div>
                        <span style="color: #666; font-size: 0.9rem;">总工时</span>
                    </div>
                    <div>
                        <span style="color: #666;">TOP面</span>
                        <div style="font-size: 1.1rem;">{result['top_time']:.1f}s</div>
                        <span style="color: #888; font-size: 0.8rem;">{top_points}点</span>
                    </div>
                    <div>
                        <span style="color: #666;">BOT面</span>
                        <div style="font-size: 1.1rem;">{result['bot_time']:.1f}s</div>
                        <span style="color: #888; font-size: 0.8rem;">{bot_points}点</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #888; font-size: 0.8rem;">R²={result.get('r2', 0):.3f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💬 AI分析")
    
    chat_container = st.container(height=200)
    with chat_container:
        for msg in st.session_state.messages[-10:]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            elif msg["role"] == "assistant":
                st.chat_message("assistant").write(msg["content"])
    
    user_input = st.chat_input("提问...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        numbers = re.findall(r'\d+', user_input)
        
        with st.spinner("分析中..."):
            if st.session_state.last_prediction_result is not None and numbers:
                response = chat_with_ai(user_input, st.session_state.last_prediction_result)
            else:
                response = chat_with_ai(user_input, None)
        
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
                    st.write(f"- {h}")
            else:
                st.write("暂无预测记录")
