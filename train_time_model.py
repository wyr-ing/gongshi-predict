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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="SMT工时预测系统 - AI智能体",
    page_icon="🤖",
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
# 数据分布工具
# ============================================================

def get_data_density_zones(df, column='单板点数'):
    """获取数据密度区域"""
    data = df[column].dropna()
    
    zones = {
        'min': float(data.min()),
        'q1': float(data.quantile(0.25)),
        'median': float(data.quantile(0.5)),
        'q3': float(data.quantile(0.75)),
        'max': float(data.max()),
        'count': len(data),
        'unique_values': len(data.unique()),
        'density_areas': {}
    }
    
    bins = [data.min(), data.quantile(0.25), data.quantile(0.5), data.quantile(0.75), data.max()]
    labels = ['低密度区(0-25%)', '中低密度区(25-50%)', '中高密度区(50-75%)', '高密度区(75-100%)']
    
    for i in range(len(bins)-1):
        count = len(data[(data >= bins[i]) & (data <= bins[i+1])])
        zones['density_areas'][labels[i]] = {
            'range': f'{bins[i]:.0f}~{bins[i+1]:.0f}',
            'count': count,
            'percentage': count / len(data) * 100 if len(data) > 0 else 0
        }
    
    return zones

def create_density_chart(models):
    """创建数据分布密度图"""
    screen = get_screen_size()
    
    fig, ax = plt.subplots(figsize=(screen['fig_width'], screen['fig_height'] * 0.6), dpi=100)
    
    if 'smt' in models:
        model_info = models['smt']
        df = model_info['data']
        x = df['单板点数']
        
        n, bins, patches = ax.hist(x, bins=20, color='#1f77b4', alpha=0.7, edgecolor='white', linewidth=1)
        
        q1 = x.quantile(0.25)
        q2 = x.quantile(0.5)
        q3 = x.quantile(0.75)
        
        ax.axvline(q1, color='orange', linestyle='--', linewidth=2, label=f'Q1 (25%): {q1:.0f}')
        ax.axvline(q2, color='green', linestyle='--', linewidth=2, label=f'Q2 (50%): {q2:.0f}')
        ax.axvline(q3, color='red', linestyle='--', linewidth=2, label=f'Q3 (75%): {q3:.0f}')
        
        ax.set_xlabel('Points (pts)', fontsize=screen['font_size'], fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=screen['font_size'], fontweight='bold')
        ax.set_title('📊 Data Distribution (Points)', fontsize=screen['title_size'], fontweight='bold')
        ax.legend(loc='upper right', fontsize=screen['legend_size'])
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    return fig

def auto_analyze_data(df):
    """自动分析数据"""
    if df is None or len(df) == 0:
        return {"error": "无数据"}
    
    analysis = {}
    analysis['row_count'] = len(df)
    analysis['columns'] = df.columns.tolist()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    analysis['numeric_cols'] = numeric_cols
    
    for col in numeric_cols:
        if col in ['单板点数', '标准工时']:
            analysis[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q1': float(df[col].quantile(0.25)),
                'q3': float(df[col].quantile(0.75)),
                'missing': int(df[col].isna().sum())
            }
    
    if '单板点数' in df.columns:
        analysis['density_zones'] = get_data_density_zones(df, '单板点数')
    
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        analysis['correlation'] = corr.to_dict()
    
    return analysis

def generate_data_summary(df):
    """生成数据摘要"""
    if df is None or len(df) == 0:
        return "无数据"
    
    analysis = auto_analyze_data(df)
    
    summary = f"""
📊 **数据概览**
- 数据行数：{analysis.get('row_count', 0)} 行
- 列名：{', '.join(analysis.get('columns', []))}

📈 **单板点数统计**
- 平均值：{analysis.get('单板点数', {}).get('mean', 0):.2f} 点
- 中位数：{analysis.get('单板点数', {}).get('median', 0):.2f} 点
- 标准差：{analysis.get('单板点数', {}).get('std', 0):.2f}
- 范围：{analysis.get('单板点数', {}).get('min', 0):.0f} ~ {analysis.get('单板点数', {}).get('max', 0):.0f} 点

📈 **标准工时统计**
- 平均值：{analysis.get('标准工时', {}).get('mean', 0):.2f} 秒
- 中位数：{analysis.get('标准工时', {}).get('median', 0):.2f} 秒
- 标准差：{analysis.get('标准工时', {}).get('std', 0):.2f}
- 范围：{analysis.get('标准工时', {}).get('min', 0):.2f} ~ {analysis.get('标准工时', {}).get('max', 0):.2f} 秒
"""
    
    if analysis.get('correlation'):
        corr = analysis['correlation']
        if '单板点数' in corr and '标准工时' in corr:
            summary += f"\n📊 **相关性**\n- 单板点数与标准工时的相关系数：{corr['单板点数'].get('标准工时', 0):.3f}"
    
    return summary

# ============================================================
# 训练模型
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
            'x_min': float(df_clean['单板点数'].min()),
            'x_max': float(df_clean['单板点数'].max()),
            'y_min': float(df_clean['标准工时'].min()),
            'y_max': float(df_clean['标准工时'].max()),
            'x_mean': float(df_clean['单板点数'].mean()),
            'y_mean': float(df_clean['标准工时'].mean()),
            'x_std': float(df_clean['单板点数'].std()),
            'y_std': float(df_clean['标准工时'].std()),
            'df_clean': df_clean
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
        'time': float(pred),
        'r2': float(models['smt']['r2']),
        'mape': float(models['smt']['mape']),
        'mae': float(models['smt']['mae'])
    }

# ============================================================
# 绘图函数
# ============================================================
def create_chart(models, points=None, predicted_time=None, line_type="SMT"):
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
    
    if points is not None and predicted_time is not None:
        ax.scatter([points], [predicted_time], color='#ff6b6b', 
                  s=screen['marker_size'] * 3, edgecolors='white', linewidth=2, zorder=6,
                  label=f'Prediction: {points} pts → {predicted_time:.1f}s')
    
    ax.set_xlabel('Points (pts)', fontsize=screen['font_size'], fontweight='bold')
    ax.set_ylabel('Time (s)', fontsize=screen['font_size'], fontweight='bold')
    ax.set_title(f'📊 {line_type} Manhour Prediction Chart', 
                 fontsize=screen['title_size'], fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=screen['legend_size'])
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    return fig

# ============================================================
# 🤖 智能体核心
# ============================================================

def agent_chat(user_message, models=None, prediction_result=None, df_raw=None, line_type="SMT"):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    numbers = re.findall(r'\d+', user_message)
    target_points = int(numbers[0]) if numbers else None
    
    if target_points is not None and target_points > 0 and models is not None and 'smt' in models:
        pred_result = predict_time(target_points, models)
        if pred_result:
            pred_time = pred_result['time']
            lower_bound = pred_time * (1 - pred_result['mape'] / 100)
            upper_bound = pred_time * (1 + pred_result['mape'] / 100)
        else:
            pred_time = None
            lower_bound = None
            upper_bound = None
            pred_result = None
    else:
        pred_time = None
        lower_bound = None
        upper_bound = None
        pred_result = None

    knowledge_base = []
    
    if df_raw is not None and len(df_raw) > 0:
        knowledge_base.append("【📁 原始数据】")
        knowledge_base.append(f"- 数据行数：{len(df_raw)} 行")
        knowledge_base.append("")
    
    if df_raw is not None and len(df_raw) > 0:
        data_summary = generate_data_summary(df_raw)
        knowledge_base.append("【📊 数据统计摘要】")
        knowledge_base.append(data_summary)
        knowledge_base.append("")
    
    if models is not None and 'smt' in models:
        model_info = models['smt']
        knowledge_base.append("【📈 模型信息】")
        knowledge_base.append(f"- 模型类型：二次多项式回归")
        knowledge_base.append(f"- 有效数据点：{model_info['sample_count']} 个")
        knowledge_base.append(f"- R²：{model_info['r2']:.4f}")
        knowledge_base.append(f"- MAPE：{model_info['mape']:.2f}%")
        knowledge_base.append(f"- MAE：{model_info['mae']:.2f} 秒")
        knowledge_base.append("")
    
    if pred_result is not None:
        knowledge_base.append("【🎯 当前预测结果】")
        knowledge_base.append(f"- 输入点数：{pred_result['points']} 点")
        knowledge_base.append(f"- 预测工时：{pred_result['time']:.2f} 秒")
        knowledge_base.append(f"- 预测范围：{lower_bound:.2f} ~ {upper_bound:.2f} 秒")
        knowledge_base.append("")
    
    full_knowledge = "\n".join(knowledge_base)
    
    if pred_result is not None:
        system_prompt = f"""你是一个专业的SMT工时预测分析智能体。

【📚 你掌握的知识】
{full_knowledge}

【🎯 输出格式】

---

**📊 预测结果**

> 输入 {target_points} 点 → 预测 **{pred_time:.2f}** 秒（范围 {lower_bound:.2f} ~ {upper_bound:.2f} 秒）

---

**📈 数据解读**

（基于数据统计摘要进行分析）

---

**📌 总结**

（总结性结论）

---"""
    else:
        system_prompt = f"""你是一个专业的SMT工时预测分析智能体。

【📚 你掌握的知识】
{full_knowledge}

请提示用户输入具体点数（如 100），以便基于拟合曲线给出预测。"""
    
    messages = [{"role": "system", "content": system_prompt}]
    chat_history = st.session_state.messages[-20:] if st.session_state.messages else []
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})
    
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": messages,
        "temperature": 0.2,
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
    st.session_state.messages = load_chat_history()

for key in ['models', 'df_raw', 'last_prediction', 'last_prediction_result', 'chart_fig']:
    if key not in st.session_state:
        st.session_state[key] = None

# 控制数据分布图显示的状态
if "show_density_chart" not in st.session_state:
    st.session_state.show_density_chart = False

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
    if st.session_state.models is not None and 'smt' in st.session_state.models:
        st.session_state.chart_fig = create_chart(st.session_state.models)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### 🤖 智能体管理")
    
    if st.session_state.models is not None and 'smt' in st.session_state.models:
        sample_count = st.session_state.models['smt']['sample_count']
        st.success(f"✅ 数据行数: {sample_count}")
    else:
        st.warning("⚠️ 暂无数据，请上传")
    
    st.markdown("---")
    st.markdown("#### 🔧 智能体工具")
    
    # 查看数据分布 - 使用session_state控制显示
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("📊 查看数据分布", use_container_width=True):
            st.session_state.show_density_chart = not st.session_state.show_density_chart
            st.rerun()
    
    with col2:
        if st.session_state.show_density_chart:
            if st.button("✖️ 收起", use_container_width=True):
                st.session_state.show_density_chart = False
                st.rerun()
    
    # 显示数据分布图
    if st.session_state.show_density_chart:
        if st.session_state.models is not None and 'smt' in st.session_state.models:
            fig = create_density_chart(st.session_state.models)
            st.pyplot(fig)
            plt.close(fig)
            
            # 显示密度区域统计
            density = get_data_density_zones(st.session_state.df_raw, '单板点数')
            st.write("**📊 数据分布区域：**")
            for area, info in density.get('density_areas', {}).items():
                st.write(f"- {area}：{info['range']} 点，{info['count']} 个 ({info['percentage']:.1f}%)")
        else:
            st.info("请先上传数据")
    
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
                st.session_state.last_prediction_result = None
                if st.session_state.models is not None and 'smt' in st.session_state.models:
                    st.session_state.chart_fig = create_chart(st.session_state.models)
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
st.markdown("<h1 style='text-align: center;'>🤖 SMT 工时预测系统 · AI智能体</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)

# ============================================================
# 第一行：左右两栏
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
                fig = create_chart(
                    models,
                    points=last.get('points'),
                    predicted_time=last.get('time')
                )
                plot_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                if st.session_state.chart_fig is not None:
                    plot_placeholder.pyplot(st.session_state.chart_fig, use_container_width=True)
                else:
                    fig = create_chart(models)
                    st.session_state.chart_fig = fig
                    plot_placeholder.pyplot(fig, use_container_width=True)
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
# 第二行：AI智能体分析
# ============================================================
st.markdown("---")

with st.container():
    st.markdown("<h3 style='text-align: center;'>🤖 AI 智能体分析</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 0.85rem;'>基于数据统计和拟合曲线进行分析</p>", unsafe_allow_html=True)
    
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    with status_col1:
        if st.session_state.df_raw is not None:
            st.success("📁 数据已加载")
        else:
            st.warning("📁 无数据")
    
    with status_col2:
        if st.session_state.models is not None and 'smt' in st.session_state.models:
            st.success("📈 模型就绪")
        else:
            st.warning("📈 未训练")
    
    with status_col3:
        if st.session_state.last_prediction_result is not None:
            st.success("🎯 已预测")
        else:
            st.warning("🎯 待预测")
    
    with status_col4:
        st.info("🧠 智能体在线")
    
    st.markdown("---")
    
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            elif msg["role"] == "assistant":
                st.chat_message("assistant").markdown(msg["content"])
    
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
        
        with st.spinner("🧠 智能体思考中..."):
            response = agent_chat(
                user_input, 
                models=st.session_state.models,
                prediction_result=st.session_state.last_prediction_result,
                df_raw=st.session_state.df_raw
            )
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_chat_history(st.session_state.messages)
        
        st.rerun()
