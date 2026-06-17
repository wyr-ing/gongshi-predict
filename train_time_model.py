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
    """
    基于实际工时数据训练预测模型
    返回：模型、多项式特征转换器、评估指标
    """
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    # 使用二次多项式拟合
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    # 评估指标
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    # 计算残差用于异常检测
    residuals = y - y_pred
    
    return model, poly, r2, mae, mape, residuals

# ============================================================
# 理论工时计算
# ============================================================
def calculate_theory_time(point_count, a=0.0362, b=0.5):
    """计算理论工时：y = a * x + b"""
    return a * point_count + b

# ============================================================
# 异常值检测（使用 IQR 方法）
# ============================================================
def detect_outliers(df, model, poly, threshold=1.5):
    """
    检测异常值（基于预测残差）
    """
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    # 计算预测值
    X_poly = poly.transform(X)
    y_pred = model.predict(X_poly)
    
    # 计算残差（实际值 - 预测值）
    residuals = y - y_pred
    
    # 使用 IQR 方法检测异常
    Q1 = np.percentile(residuals, 25)
    Q3 = np.percentile(residuals, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
    
    outlier_indices = df.index[outlier_mask].tolist()
    outlier_data = df.loc[outlier_mask].copy()
    outlier_data['残差'] = residuals[outlier_mask]
    outlier_data['预测值'] = y_pred[outlier_mask]
    outlier_data['残差百分比'] = (residuals[outlier_mask] / y[outlier_mask] * 100)
    
    return outlier_indices, outlier_data, lower_bound, upper_bound

# ============================================================
# 对比图（核心图表）
# ============================================================
def plot_chart(df, model, poly, mape, point_count=None, predicted_time=None, outliers_df=None,
               x_max=None, y_max=None, x_tick_step=None, y_tick_step=None):
    """
    绘制工时预测对比图
    - 实际工时数据：散点（蓝色）
    - 预测拟合曲线：实线（红色）- 基于实际数据训练的模型
    - 理论直线：虚线（绿色）- 理论公式
    - 误差带：半透明区域
    - 预测点：红色星标
    """
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    # 生成平滑曲线用于显示拟合和理论线
    x_min_plot = max(0, X.min() - 50)
    x_max_plot = X.max() + 50
    X_smooth = np.linspace(x_min_plot, x_max_plot, 300).reshape(-1, 1)
    
    # 预测拟合曲线（基于实际数据训练的模型）
    X_smooth_poly = poly.transform(X_smooth)
    y_pred_smooth = model.predict(X_smooth_poly)
    
    # 理论直线（基于理论公式）
    y_theory = calculate_theory_time(X_smooth.flatten())

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)

    # 1. 实际工时数据散点（蓝色）
    ax.scatter(X, y, color='#1f77b4', s=55, alpha=0.7, 
               label='实际工时数据（测量值）', zorder=3)
    
    # 2. 异常值高亮（如果有）
    if outliers_df is not None and len(outliers_df) > 0:
        ax.scatter(outliers_df['点位数'], outliers_df['实际工时'], 
                   color='red', s=120, alpha=0.8, 
                   marker='x', linewidth=2,
                   label=f'⚠️ 异常数据 ({len(outliers_df)}个)', zorder=5)
    
    # 3. 预测拟合曲线（红色实线）- 这是基于实际数据训练出来的
    ax.plot(X_smooth, y_pred_smooth, color='#d62728', linewidth=3, 
            label='预测拟合曲线（模型预测）', zorder=2)
    
    # 4. 理论直线（绿色虚线）
    ax.plot(X_smooth, y_theory, color='#2ca02c', linewidth=2.2, linestyle='--', 
            label='理论直线（标准公式）', zorder=2)
    
    # 5. 误差带（基于MAPE）
    mape_val = mape if mape is not None else 17.0
    y_upper = y_pred_smooth * (1 + mape_val / 100)
    y_lower = y_pred_smooth * (1 - mape_val / 100)
    ax.fill_between(X_smooth.flatten(), y_lower, y_upper, 
                    color='#d62728', alpha=0.12, 
                    label=f'±{mape_val:.1f}% 预测误差带')

    # 6. 当前预测点标记（如果有）
    if point_count is not None and predicted_time is not None:
        ax.scatter([point_count], [predicted_time], color='#ff6b6b', s=250,
                   edgecolors='white', linewidth=2.5, zorder=6, 
                   label=f'🔮 当前预测: {point_count}点 → {predicted_time:.1f}s')
        ax.axvline(x=point_count, color='#ff6b6b', linestyle=':', alpha=0.6, linewidth=1.5)
        ax.axhline(y=predicted_time, color='#ff6b6b', linestyle=':', alpha=0.6, linewidth=1.5)

    # 图例
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.92, edgecolor='#ccc')
    
    # 坐标轴标签
    ax.set_xlabel('点位数（个）', fontsize=12, fontweight='bold')
    ax.set_ylabel('工时（秒）', fontsize=12, fontweight='bold')
    ax.set_title('📊 工时预测对比图\n（实际数据 → 预测模型 → 理论标准）', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, linestyle='--')
    
    # ===== 坐标轴范围设置 =====
    x_min = 0
    if x_max is not None and x_max > 0:
        x_max_actual = x_max
    else:
        x_max_actual = X.max() * 1.15
    
    ax.set_xlim(x_min, x_max_actual)
    
    if y_max is not None and y_max > 0:
        y_max_actual = y_max
    else:
        y_max_actual = max(y.max(), y_theory.max(), y_pred_smooth.max()) * 1.2
    
    ax.set_ylim(0, y_max_actual)
    
    # ===== 刻度间距设置 =====
    if x_tick_step is not None and x_tick_step > 0:
        x_tick_step_actual = x_tick_step
    else:
        data_range = x_max_actual - x_min
        if data_range <= 100:
            x_tick_step_actual = 10
        elif data_range <= 500:
            x_tick_step_actual = 50
        elif data_range <= 1000:
            x_tick_step_actual = 100
        else:
            x_tick_step_actual = 200
    
    x_ticks = np.arange(0, x_max_actual + x_tick_step_actual, x_tick_step_actual)
    ax.set_xticks(x_ticks)
    
    if y_tick_step is not None and y_tick_step > 0:
        y_tick_step_actual = y_tick_step
    else:
        if y_max_actual <= 100:
            y_tick_step_actual = 10
        elif y_max_actual <= 500:
            y_tick_step_actual = 50
        elif y_max_actual <= 1000:
            y_tick_step_actual = 100
        else:
            y_tick_step_actual = 200
    
    y_max_rounded = int(np.ceil(y_max_actual / y_tick_step_actual)) * y_tick_step_actual
    y_ticks = np.arange(0, y_max_rounded + y_tick_step_actual, y_tick_step_actual)
    ax.set_yticks(y_ticks)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    return fig

# ============================================================
# 预测函数
# ============================================================
def predict_time(point_count):
    """使用训练好的模型预测工时"""
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

if "outliers_removed" not in st.session_state:
    st.session_state.outliers_removed = False

# ============================================================
# 图表设置默认值
# ============================================================
if "chart_x_max" not in st.session_state:
    st.session_state.chart_x_max = None
if "chart_y_max" not in st.session_state:
    st.session_state.chart_y_max = None
if "chart_x_tick" not in st.session_state:
    st.session_state.chart_x_tick = None
if "chart_y_tick" not in st.session_state:
    st.session_state.chart_y_tick = None
if "use_custom_axis" not in st.session_state:
    st.session_state.use_custom_axis = False

# ============================================================
# 自动加载并训练模型
# ============================================================
if not st.session_state.model_trained:
    saved_df = load_saved_data()
    if saved_df is not None and len(saved_df) > 0:
        model, poly, r2, mae, mape, residuals = train_prediction_model(saved_df)
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
        if st.session_state.outliers_removed:
            st.info("🔧 已删除异常数据")
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
            
            point_col, actual_col, theory_col = get_column_mapping(df_raw)
            
            if point_col is not None and actual_col is not None:
                df = df_raw[[point_col, actual_col]].copy()
                df.columns = ['点位数', '实际工时']
                df = df.dropna()
                
                model, poly, r2, mae, mape, residuals = train_prediction_model(df)
                st.session_state.model_trained = True
                st.session_state.model = model
                st.session_state.poly = poly
                st.session_state.r2 = r2
                st.session_state.mae = mae
                st.session_state.mape = mape
                st.session_state.df = df
                st.session_state.residuals = residuals
                st.session_state.outliers_removed = False
                save_data(df)
                st.success(f"✅ 数据已保存，共 {len(df)} 行")
                st.info(f"识别到列：'{point_col}' → 点位数，'{actual_col}' → 实际工时")
                st.balloons()
                st.rerun()
            else:
                st.error(f"❌ 未找到'单板点数'或'实际工时/s'列，当前列名：{df_raw.columns.tolist()}")
    
    # ============================================================
    # 异常数据管理
    # ============================================================
    st.markdown("---")
    st.markdown("#### 🗑️ 异常数据管理")
    
    if st.session_state.model_trained and st.session_state.df is not None:
        if st.button("🔍 检测异常数据", use_container_width=True):
            with st.spinner("正在检测异常数据..."):
                outlier_indices, outlier_data, lower_bound, upper_bound = detect_outliers(
                    st.session_state.df, 
                    st.session_state.model, 
                    st.session_state.poly,
                    threshold=1.5
                )
                
                if len(outlier_data) > 0:
                    st.session_state._outlier_indices = outlier_indices
                    st.session_state._outlier_data = outlier_data
                    st.session_state._outlier_count = len(outlier_data)
                    st.success(f"✅ 发现 {len(outlier_data)} 个异常数据点")
                else:
                    st.success("✅ 未发现异常数据点，数据质量良好！")
                    st.session_state._outlier_data = None
        
        if hasattr(st.session_state, '_outlier_data') and st.session_state._outlier_data is not None:
            outlier_df = st.session_state._outlier_data
            st.info(f"发现 {len(outlier_df)} 个异常数据点")
            
            with st.expander(f"📋 查看异常数据详情 ({len(outlier_df)}个)"):
                display_df = outlier_df[['点位数', '实际工时', '预测值', '残差', '残差百分比']].copy()
                display_df['残差百分比'] = display_df['残差百分比'].round(2)
                st.dataframe(display_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("异常数据数量", len(outlier_df))
            with col2:
                st.metric("占总数据比例", f"{len(outlier_df)/len(st.session_state.df)*100:.1f}%")
            
            if st.button("🗑️ 删除所有异常数据", type="primary", use_container_width=True):
                clean_df = st.session_state.df.drop(outlier_df.index).copy()
                model, poly, r2, mae, mape, residuals = train_prediction_model(clean_df)
                
                st.session_state.df = clean_df
                st.session_state.model = model
                st.session_state.poly = poly
                st.session_state.r2 = r2
                st.session_state.mae = mae
                st.session_state.mape = mape
                st.session_state.residuals = residuals
                st.session_state.outliers_removed = True
                st.session_state._outlier_data = None
                
                save_data(clean_df)
                
                st.success(f"✅ 已删除 {len(outlier_df)} 个异常数据，剩余 {len(clean_df)} 条数据")
                st.balloons()
                st.rerun()
        
        if st.session_state.outliers_removed:
            if st.button("🔄 恢复原始数据", use_container_width=True):
                saved_df = load_saved_data()
                if saved_df is not None and len(saved_df) > 0:
                    st.warning("⚠️ 恢复功能需要从备份文件恢复，请联系管理员")
        
        with st.expander("⚙️ 异常检测阈值设置"):
            threshold = st.slider(
                "IQR 倍数阈值",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="值越小，检测越严格"
            )
            if st.button("应用阈值设置"):
                outlier_indices, outlier_data, lower_bound, upper_bound = detect_outliers(
                    st.session_state.df, 
                    st.session_state.model, 
                    st.session_state.poly,
                    threshold=threshold
                )
                if len(outlier_data) > 0:
                    st.session_state._outlier_indices = outlier_indices
                    st.session_state._outlier_data = outlier_data
                    st.session_state._outlier_count = len(outlier_data)
                    st.success(f"✅ 使用阈值 {threshold} 检测到 {len(outlier_data)} 个异常数据点")
                else:
                    st.success(f"✅ 使用阈值 {threshold} 未发现异常数据点")
                    st.session_state._outlier_data = None

    # ============================================================
    # 图表坐标轴设置
    # ============================================================
    st.markdown("---")
    st.markdown("#### 📐 图表坐标轴设置")
    
    if st.session_state.model_trained and st.session_state.df is not None:
        use_custom = st.checkbox("启用自定义坐标轴", value=st.session_state.use_custom_axis)
        st.session_state.use_custom_axis = use_custom
        
        if use_custom:
            X = st.session_state.df[['点位数']].values
            y = st.session_state.df['实际工时'].values
            x_data_max = X.max()
            y_data_max = y.max()
            
            st.caption(f"📊 数据范围：X轴 0~{x_data_max:.0f}，Y轴 0~{y_data_max:.0f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**X轴设置（点位数）**")
                x_max_input = st.number_input(
                    "X轴最大值",
                    min_value=0,
                    value=int(x_data_max * 1.15) if st.session_state.chart_x_max is None else int(st.session_state.chart_x_max),
                    step=50,
                    key="x_max_input"
                )
                if x_max_input > 0:
                    st.session_state.chart_x_max = x_max_input
                else:
                    st.session_state.chart_x_max = None
                
                x_tick_options = [10, 20, 25, 50, 100, 200, 250, 500]
                x_tick_default = 50 if x_data_max > 100 else 10
                if st.session_state.chart_x_tick is not None:
                    x_tick_default = st.session_state.chart_x_tick
                x_tick_input = st.selectbox(
                    "X轴刻度步长",
                    options=x_tick_options,
                    index=x_tick_options.index(x_tick_default) if x_tick_default in x_tick_options else 0,
                    key="x_tick_select"
                )
                st.session_state.chart_x_tick = x_tick_input
            
            with col2:
                st.markdown("**Y轴设置（工时）**")
                y_max_input = st.number_input(
                    "Y轴最大值",
                    min_value=0,
                    value=int(y_data_max * 1.2) if st.session_state.chart_y_max is None else int(st.session_state.chart_y_max),
                    step=50,
                    key="y_max_input"
                )
                if y_max_input > 0:
                    st.session_state.chart_y_max = y_max_input
                else:
                    st.session_state.chart_y_max = None
                
                y_tick_options = [10, 20, 25, 50, 100, 200, 250, 500]
                y_tick_default = 50 if y_data_max > 100 else 10
                if st.session_state.chart_y_tick is not None:
                    y_tick_default = st.session_state.chart_y_tick
                y_tick_input = st.selectbox(
                    "Y轴刻度步长",
                    options=y_tick_options,
                    index=y_tick_options.index(y_tick_default) if y_tick_default in y_tick_options else 0,
                    key="y_tick_select"
                )
                st.session_state.chart_y_tick = y_tick_input
            
            if st.button("🔄 重置为自动", use_container_width=True):
                st.session_state.chart_x_max = None
                st.session_state.chart_y_max = None
                st.session_state.chart_x_tick = None
                st.session_state.chart_y_tick = None
                st.session_state.use_custom_axis = False
                st.rerun()
        else:
            st.caption("当前使用自动坐标轴设置")

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
            
            outliers_df = None
            if hasattr(st.session_state, '_outlier_data') and st.session_state._outlier_data is not None:
                outliers_df = st.session_state._outlier_data
            
            x_max = st.session_state.chart_x_max if st.session_state.use_custom_axis else None
            y_max = st.session_state.chart_y_max if st.session_state.use_custom_axis else None
            x_tick = st.session_state.chart_x_tick if st.session_state.use_custom_axis else None
            y_tick = st.session_state.chart_y_tick if st.session_state.use_custom_axis else None
            
            if st.session_state.last_prediction is not None:
                fig = plot_chart(
                    st.session_state.df,
                    st.session_state.model,
                    st.session_state.poly,
                    st.session_state.mape,
                    point_count=st.session_state.last_prediction["point_count"],
                    predicted_time=st.session_state.last_prediction["predicted"],
                    outliers_df=outliers_df,
                    x_max=x_max,
                    y_max=y_max,
                    x_tick_step=x_tick,
                    y_tick_step=y_tick
                )
                plot_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                fig = plot_chart(
                    st.session_state.df, 
                    st.session_state.model, 
                    st.session_state.poly, 
                    st.session_state.mape,
                    outliers_df=outliers_df,
                    x_max=x_max,
                    y_max=y_max,
                    x_tick_step=x_tick,
                    y_tick_step=y_tick
                )
                plot_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
    else:
        st.info("👈 请在左侧菜单上传数据")

# ============================================================
# 右侧：AI智能体对话
# ============================================================
with right_col:
    st.markdown("### 🎯 工时预测小助手")
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
