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
# ============================================================
# 🤖 智能体工具函数
# ============================================================

def detect_outliers(df, column='单板点数', method='iqr', threshold=1.5):
    """检测异常值"""
    data = df[column].dropna()
    outliers = []
    
    if method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = data[z_scores > threshold]
    
    return {
        'count': len(outliers),
        'values': outliers.tolist() if len(outliers) > 0 else [],
        'indices': outliers.index.tolist() if len(outliers) > 0 else [],
        'percentage': len(outliers) / len(data) * 100 if len(data) > 0 else 0
    }

def train_multiple_models(df):
    """训练多个模型并返回最佳"""
    df_clean = df.dropna(subset=['单板点数', '标准工时'])
    df_clean = df_clean[(df_clean['单板点数'] > 0) & (df_clean['标准工时'] > 0)]
    
    if len(df_clean) < 5:
        return None
    
    X = df_clean[['单板点数']].values
    y = df_clean['标准工时'].values
    
    models = {}
    results = []
    
    # 1. 多项式回归（2次）
    try:
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        cv_score = np.mean(cross_val_score(model, X_poly, y, cv=min(5, len(df_clean))))
        models['多项式回归(2次)'] = {
            'model': model,
            'poly': poly,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'cv_score': cv_score,
            'type': 'polynomial'
        }
        results.append(('多项式回归(2次)', r2, mae, mape))
    except:
        pass
    
    # 2. 多项式回归（3次）
    try:
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        cv_score = np.mean(cross_val_score(model, X_poly, y, cv=min(5, len(df_clean))))
        models['多项式回归(3次)'] = {
            'model': model,
            'poly': poly,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'cv_score': cv_score,
            'type': 'polynomial'
        }
        results.append(('多项式回归(3次)', r2, mae, mape))
    except:
        pass
    
    # 3. 随机森林
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        cv_score = np.mean(cross_val_score(model, X, y, cv=min(5, len(df_clean))))
        models['随机森林'] = {
            'model': model,
            'poly': None,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'cv_score': cv_score,
            'type': 'random_forest'
        }
        results.append(('随机森林', r2, mae, mape))
    except:
        pass
    
    # 4. Ridge回归
    try:
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        cv_score = np.mean(cross_val_score(model, X_scaled, y, cv=min(5, len(df_clean))))
        models['Ridge回归'] = {
            'model': model,
            'poly': poly,
            'scaler': scaler,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'cv_score': cv_score,
            'type': 'ridge'
        }
        results.append(('Ridge回归', r2, mae, mape))
    except:
        pass
    
    # 选择最佳模型（按R²排序）
    if results:
        best = max(results, key=lambda x: x[1])
        return {
            'best_model_name': best[0],
            'best_model': models[best[0]],
            'all_models': models,
            'comparison': results,
            'data': df_clean
        }
    
    return None

def auto_analyze_data(df):
    """自动分析数据质量"""
    if df is None or len(df) == 0:
        return {"error": "无数据"}
    
    analysis = {}
    
    # 基本统计
    analysis['row_count'] = len(df)
    analysis['columns'] = df.columns.tolist()
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    analysis['numeric_cols'] = numeric_cols
    
    for col in numeric_cols:
        if col in ['单板点数', '标准工时']:
            analysis[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75),
                'missing': df[col].isna().sum()
            }
    
    # 异常值检测
    if '单板点数' in df.columns:
        analysis['outliers_points'] = detect_outliers(df, '单板点数')
    
    if '标准工时' in df.columns:
        analysis['outliers_time'] = detect_outliers(df, '标准工时')
    
    # 相关性
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        analysis['correlation'] = corr.to_dict()
    
    return analysis

def generate_data_summary(df):
    """生成数据摘要（用于AI上下文）"""
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
- Q1：{analysis.get('单板点数', {}).get('q1', 0):.0f} 点
- Q3：{analysis.get('单板点数', {}).get('q3', 0):.0f} 点

📈 **标准工时统计**
- 平均值：{analysis.get('标准工时', {}).get('mean', 0):.2f} 秒
- 中位数：{analysis.get('标准工时', {}).get('median', 0):.2f} 秒
- 标准差：{analysis.get('标准工时', {}).get('std', 0):.2f}
- 范围：{analysis.get('标准工时', {}).get('min', 0):.2f} ~ {analysis.get('标准工时', {}).get('max', 0):.2f} 秒

🔍 **异常值检测**
- 单板点数异常值：{analysis.get('outliers_points', {}).get('count', 0)} 个 ({analysis.get('outliers_points', {}).get('percentage', 0):.1f}%)
- 标准工时异常值：{analysis.get('outliers_time', {}).get('count', 0)} 个 ({analysis.get('outliers_time', {}).get('percentage', 0):.1f}%)
"""
    
    if analysis.get('correlation'):
        corr = analysis['correlation']
        if '单板点数' in corr and '标准工时' in corr:
            summary += f"\n📊 **相关性**\n- 单板点数与标准工时的相关系数：{corr['单板点数'].get('标准工时', 0):.3f}"
    
    return summary

# ============================================================
# 训练模型（保留原功能）
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
            'y_max': df_clean['标准工时'].max(),
            'x_mean': df_clean['单板点数'].mean(),
            'y_mean': df_clean['标准工时'].mean(),
            'x_std': df_clean['单板点数'].std(),
            'y_std': df_clean['标准工时'].std()
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
# ============================================================
# 🤖 智能体核心 - 带工具调用的AI对话
# ============================================================

def agent_chat(user_message, models=None, prediction_result=None, df_raw=None, line_type="SMT"):
    """智能体对话 - 拥有工具调用能力"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    # ========== 构建智能体知识库 ==========
    knowledge_base = []
    
    # 1. 原始数据信息
    if df_raw is not None and len(df_raw) > 0:
        knowledge_base.append("【📁 原始数据】")
        knowledge_base.append(f"- 数据行数：{len(df_raw)} 行")
        knowledge_base.append(f"- 列名：{', '.join(df_raw.columns.tolist())}")
        knowledge_base.append(f"- 前5行数据预览：")
        preview = df_raw.head(5).to_string()
        knowledge_base.append(f"```\n{preview}\n```")
        knowledge_base.append("")
    
    # 2. 自动数据分析结果
    if df_raw is not None and len(df_raw) > 0:
        data_summary = generate_data_summary(df_raw)
        knowledge_base.append("【📊 自动数据分析结果】")
        knowledge_base.append(data_summary)
        knowledge_base.append("")
    
    # 3. 模型信息
    if models is not None and 'smt' in models:
        model_info = models['smt']
        knowledge_base.append("【📈 模型信息】")
        knowledge_base.append(f"- 模型类型：二次多项式回归")
        knowledge_base.append(f"- 有效数据点：{model_info['sample_count']} 个")
        knowledge_base.append(f"- R²：{model_info['r2']:.4f}")
        knowledge_base.append(f"- MAPE：{model_info['mape']:.2f}%")
        knowledge_base.append(f"- MAE：{model_info['mae']:.2f} 秒")
        knowledge_base.append("")
    
    # 4. 预测结果
    if prediction_result is not None:
        p = prediction_result
        lower_bound = p['time'] * (1 - p['mape'] / 100)
        upper_bound = p['time'] * (1 + p['mape'] / 100)
        knowledge_base.append("【🎯 当前预测结果】")
        knowledge_base.append(f"- 输入点数：{p['points']} 点")
        knowledge_base.append(f"- 预测工时：{p['time']:.2f} 秒")
        knowledge_base.append(f"- 预测范围：{lower_bound:.2f} ~ {upper_bound:.2f} 秒")
        knowledge_base.append("")
    
    # 5. 可用工具列表
    knowledge_base.append("【🔧 可用工具】")
    knowledge_base.append("1. `detect_outliers()` - 检测数据中的异常值")
    knowledge_base.append("2. `auto_analyze_data()` - 自动分析数据质量")
    knowledge_base.append("3. `train_multiple_models()` - 训练多个模型并选择最佳")
    knowledge_base.append("4. `predict_time()` - 根据点数预测工时")
    knowledge_base.append("5. `generate_data_summary()` - 生成数据摘要报告")
    knowledge_base.append("")
    
    full_knowledge = "\n".join(knowledge_base)
    
    # ========== 构建系统提示 ==========
    system_prompt = f"""你是一个专业的SMT工时预测分析智能体（Agent）。

【🤖 你的身份】
你是一个拥有完整权限的智能体，可以访问所有数据和工具。
你的目标是帮助用户分析SMT工时数据，提供准确的预测和建议。

【📚 你掌握的全部知识】
{full_knowledge}

【🧠 你的能力】
1. **数据理解**：理解原始数据结构、分布特征、统计信息
2. **异常检测**：自动识别数据中的异常值和离群点
3. **模型评估**：评估拟合曲线的质量，判断预测可信度
4. **趋势分析**：分析点数与工时的变化趋势
5. **预测分析**：基于拟合曲线给出精准预测和范围
6. **工具调用**：需要时可以调用分析工具进行深入分析

【📋 分析规则】
1. 所有分析必须基于知识库中的数据
2. 不要使用"行业基准"等没有数据依据的词汇
3. 分析要逻辑清晰，先给结论再给依据
4. 如果用户输入点数，先给出预测结果，再进行分析
5. 如果发现数据异常，主动提醒用户

【🎯 输出格式】
使用Markdown格式，结构清晰。

如果用户输入了具体点数，请按以下格式输出：

---

**📊 预测结果**

> 输入 X 点 → 预测 **X.XX** 秒（范围 X.XX ~ X.XX 秒）

---

**📈 数据解读**

（分析数据分布、拟合曲线质量、预测点位置）

---

**🔍 深度分析**（可选）

（如果发现异常值、数据质量问题时，主动指出）

---

**📌 总结**

（总结性结论，包括预测可信度评估）

---

请开始分析用户的输入。"""
    
    # ========== 构建对话消息 ==========
    messages = [{"role": "system", "content": system_prompt}]
    chat_history = st.session_state.messages[-20:] if st.session_state.messages else []
    for msg in chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})
    
    # ========== 调用AI ==========
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2500
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

for key in ['models', 'df_raw', 'last_prediction', 'last_prediction_result', 'chart_fig', 'agent_analysis']:
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
    
    if st.button("🔍 检测异常值", use_container_width=True):
        if st.session_state.df_raw is not None:
            outliers = detect_outliers(st.session_state.df_raw, '单板点数')
            if outliers['count'] > 0:
                st.warning(f"发现 {outliers['count']} 个异常值 ({outliers['percentage']:.1f}%)")
                st.write(f"异常值：{outliers['values'][:10]}")
            else:
                st.success("✅ 未发现异常值")
        else:
            st.info("请先上传数据")
    
    if st.button("📊 数据质量报告", use_container_width=True):
        if st.session_state.df_raw is not None:
            analysis = auto_analyze_data(st.session_state.df_raw)
            st.json(analysis)
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
    st.markdown("<p style='text-align: center; color: #888; font-size: 0.85rem;'>智能体拥有完整权限：数据感知 · 模型分析 · 异常检测 · 趋势预测 · 工具调用</p>", unsafe_allow_html=True)
    
    # 显示智能体状态
    status_col1, status_col2, status_col3, status_col4 = st
