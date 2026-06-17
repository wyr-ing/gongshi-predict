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
from scipy import stats
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
# 屏幕自适应工具函数
# ============================================================
def get_screen_size():
    try:
        screen_width = st.session_state.get('screen_width', 1200)
        screen_height = st.session_state.get('screen_height', 800)
    except:
        screen_width = 1200
        screen_height = 800
    
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
# 训练预测模型（含置信区间计算）
# ============================================================
def train_prediction_model(df):
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    # 计算残差
    residuals = y - y_pred
    
    # 计算评估指标
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    # 计算置信区间参数
    n = len(residuals)
    std_residuals = np.std(residuals, ddof=1)
    mean_residual = np.mean(residuals)
    
    # 95% 置信区间
    confidence_level = 0.95
    t_value = stats.t.ppf((1 + confidence_level) / 2, n - 2)
    ci_margin = t_value * std_residuals * np.sqrt(1 + 1/n)  # 预测区间
    
    return model, poly, r2, mae, mape, residuals, ci_margin, mean_residual

# ============================================================
# 自动检测并剔除异常数据（基于置信区间）
# ============================================================
def auto_clean_data(df, confidence_level=0.95, outlier_threshold=2.5):
    """
    自动检测并剔除异常数据
    confidence_level: 置信度 (0.90, 0.95, 0.99)
    outlier_threshold: 异常值倍数阈值 (默认2.5倍标准差)
    """
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    # 训练初始模型
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    # 计算残差
    residuals = y - y_pred
    std_residual = np.std(residuals, ddof=1)
    mean_residual = np.mean(residuals)
    
    # 方法1：基于置信区间（残差超出 t 分布置信区间）
    n = len(residuals)
    t_value = stats.t.ppf((1 + confidence_level) / 2, n - 2)
    ci_margin = t_value * std_residual * np.sqrt(1 + 1/n)
    
    # 方法2：基于标准差倍数（更严格）
    std_threshold = outlier_threshold * std_residual
    
    # 综合判断：取两种方法中更严格的
    threshold = max(ci_margin, std_threshold)
    
    # 识别异常数据
    outlier_mask = np.abs(residuals) > threshold
    outlier_indices = df.index[outlier_mask].tolist()
    outlier_data = df.loc[outlier_mask].copy()
    outlier_data['残差'] = residuals[outlier_mask]
    outlier_data['预测值'] = y_pred[outlier_mask]
    outlier_data['残差百分比'] = (residuals[outlier_mask] / y[outlier_mask] * 100)
    outlier_data['是否异常'] = '异常'
    
    # 正常数据
    clean_df = df.loc[~outlier_mask].copy()
    
    # 统计信息
    clean_stats = {
        'original_count': len(df),
        'outlier_count': len(outlier_data),
        'cleaned_count': len(clean_df),
        'outlier_ratio': len(outlier_data) / len(df) * 100,
        'threshold': threshold,
        'confidence_level': confidence_level,
        'std_residual': std_residual,
        'ci_margin': ci_margin
    }
    
    # 如果异常数据比例超过30%，说明数据质量太差，使用更宽松的阈值
    if clean_stats['outlier_ratio'] > 30:
        # 使用2.5倍标准差重新检测
        new_threshold = 2.5 * std_residual
        new_outlier_mask = np.abs(residuals) > new_threshold
        new_outlier_indices = df.index[new_outlier_mask].tolist()
        new_outlier_data = df.loc[new_outlier_mask].copy()
        new_clean_df = df.loc[~new_outlier_mask].copy()
        
        # 更新结果
        outlier_indices = new_outlier_indices
        outlier_data = new_outlier_data
        clean_df = new_clean_df
        clean_stats['threshold'] = new_threshold
        clean_stats['outlier_count'] = len(new_outlier_data)
        clean_stats['cleaned_count'] = len(new_clean_df)
        clean_stats['outlier_ratio'] = len(new_outlier_data) / len(df) * 100
        clean_stats['note'] = '数据质量较差，使用宽松阈值'
    
    return clean_df, outlier_data, outlier_indices, clean_stats

# ============================================================
# 理论工时计算
# ============================================================
def calculate_theory_time(point_count, a=0.0362, b=0.5):
    return a * point_count + b

# ============================================================
# 对比图（自适应版 + 置信区间显示）
# ============================================================
def plot_chart(df, model, poly, mape, residuals=None, ci_margin=None, 
               point_count=None, predicted_time=None, outlier_df=None):
    
    screen = get_screen_size()
    
    X = df[['点位数']].values
    y = df['实际工时'].values
    
    x_min_plot = max(0, X.min() - 50)
    x_max_plot = X.max() + 50
    X_smooth = np.linspace(x_min_plot, x_max_plot, 300).reshape(-1, 1)
    
    X_smooth_poly = poly.transform(X_smooth)
    y_pred_smooth = model.predict(X_smooth_poly)
    y_theory = calculate_theory_time(X_smooth.flatten())

    fig, ax = plt.subplots(figsize=(screen['fig_width'], screen['fig_height']), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)

    # 1. 正常数据点
    ax.scatter(X, y, color='#1f77b4', s=screen['marker_size'], alpha=0.7, 
               label='Normal Data', zorder=3)
    
    # 2. 异常数据点（如果有）
    if outlier_df is not None and len(outlier_df) > 0:
        ax.scatter(outlier_df['点位数'], outlier_df['实际工时'], 
                   color='red', s=screen['marker_size'] * 1.8, alpha=0.8,
                   marker='x', linewidth=2.5,
                   label=f'Outliers ({len(outlier_df)} removed)', zorder=5)
    
    # 3. 预测曲线
    ax.plot(X_smooth, y_pred_smooth, color='#d62728', linewidth=2.5, 
            label='Prediction Curve', zorder=2)
    
    # 4. 理论直线
    ax.plot(X_smooth, y_theory, color='#2ca02c', linewidth=2, linestyle='--', 
            label='Theory Line', zorder=2)
    
    # 5. 置信区间（如果有残差数据）
    if residuals is not None and ci_margin is not None:
        # 计算置信区间上下界
        y_upper_ci = y_pred_smooth + ci_margin
        y_lower_ci = y_pred_smooth - ci_margin
        ax.fill_between(X_smooth.flatten(), y_lower_ci, y_upper_ci, 
                        color='#d62728', alpha=0.08, 
                        label=f'95% Confidence Interval', zorder=1)
    
    # 6. 误差带（基于MAPE）
    mape_val = mape if mape is not None else 17.0
    y_upper = y_pred_smooth * (1 + mape_val / 100)
    y_lower = y_pred_smooth * (1 - mape_val / 100)
    ax.fill_between(X_smooth.flatten(), y_lower, y_upper, 
                    color='#d62728', alpha=0.10, 
                    label=f'±{mape_val:.1f}% Error Band')

    # 7. 预测点标记
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
    
    # 标题显示数据清理信息
    title = '📊 Manhour Prediction Chart'
    if outlier_df is not None and len(outlier_df) > 0:
        title += f' (Auto-cleaned: {len(outlier_df)} outliers removed)'
    ax.set_title(title, fontsize=screen['title_size'], fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.25, linestyle='--')
    
    # 坐标轴范围
    x_max = X.max() * 1.15
    all_y = np.concatenate([y, y_theory, y_pred_smooth])
    if residuals is not None:
        all_y = np.concatenate([all_y, y_pred_smooth + ci_margin, y_pred_smooth - ci_margin])
    y_max = max(all_y) * 1.2
    
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
    st.session_state.ci_margin = None
    st.session_state.outlier_df = None
    st.session_state.clean_stats = None
    
if "upload_authorized" not in st.session_state:
    st.session_state.upload_authorized = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_prediction_result" not in st.session_state:
    st.session_state.last_prediction_result = None

if "auto_clean_enabled" not in st.session_state:
    st.session_state.auto_clean_enabled = True  # 默认开启自动清理

if "screen_width" not in st.session_state:
    st.session_state.screen_width = 1200
if "screen_height" not in st.session_state:
    st.session_state.screen_height = 800

# ============================================================
# 自动加载并训练模型（带自动清理）
# ============================================================
if not st.session_state.model_trained:
    saved_df = load_saved_data()
    if saved_df is not None and len(saved_df) > 0:
        df_to_use = saved_df.copy()
        
        # 如果启用自动清理
        if st.session_state.auto_clean_enabled:
            # 执行自动清理
            clean_df, outlier_df, outlier_indices, clean_stats = auto_clean_data(
                df_to_use, 
                confidence_level=0.95, 
                outlier_threshold=2.5
            )
            
            # 保存清理信息
            st.session_state.outlier_df = outlier_df
            st.session_state.clean_stats = clean_stats
            st.session_state.original_df = df_to_use
            
            # 使用清理后的数据训练模型
            df_to_use = clean_df
        
        # 训练模型
        model, poly, r2, mae, mape, residuals, ci_margin, mean_residual = train_prediction_model(df_to_use)
        
        st.session_state.model_trained = True
        st.session_state.model = model
        st.session_state.poly = poly
        st.session_state.r2 = r2
        st.session_state.mae = mae
        st.session_state.mape = mape
        st.session_state.df = df_to_use
        st.session_state.residuals = residuals
        st.session_state.ci_margin = ci_margin

# ============================================================
# 注入 JavaScript 获取屏幕尺寸
# ============================================================
st.markdown("""
<script>
window.addEventListener('resize', function() {
    const width = window.innerWidth;
    const height = window.innerHeight;
});
</script>
""", unsafe_allow_html=True)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ 数据管理")
    if st.session_state.model_trained and st.session_state.df is not None:
        st.success(f"✅ 当前数据：{len(st.session_state.df)} 行")
        
        # 显示清理信息
        if st.session_state.clean_stats is not None:
            stats = st.session_state.clean_stats
            if stats['outlier_count'] > 0:
                st.info(f"🧹 自动清理：移除了 {stats['outlier_count']} 个异常数据")
                st.caption(f"清理比例：{stats['outlier_ratio']:.1f}%")
            else:
                st.success("✅ 数据质量良好，无异常数据")
    else:
        st.warning("⚠️ 暂无数据")

    st.markdown("---")
    
    # ============================================================
    # 自动清理设置
    # ============================================================
    st.markdown("#### 🧹 数据自动清理")
    
    auto_clean = st.checkbox(
        "启用自动异常数据清理", 
        value=st.session_state.auto_clean_enabled,
        help="自动检测并剔除超出95%置信区间的异常数据"
    )
    
    if auto_clean != st.session_state.auto_clean_enabled:
        st.session_state.auto_clean_enabled = auto_clean
        st.session_state.model_trained = False
        st.rerun()
    
    if st.session_state.auto_clean_enabled and st.session_state.clean_stats is not None:
        stats = st.session_state.clean_stats
        st.caption(f"置信度：{int(stats['confidence_level']*100)}%")
        st.caption(f"阈值：±{stats['threshold']:.2f} 秒")
        
        # 显示清理详情按钮
        if stats['outlier_count'] > 0:
            with st.expander(f"📋 查看异常数据详情 ({stats['outlier_count']}个)"):
                if st.session_state.outlier_df is not None and len(st.session_state.outlier_df) > 0:
                    display_df = st.session_state.outlier_df[['点位数', '实际工时', '预测值', '残差', '残差百分比']].copy()
                    display_df['残差百分比'] = display_df['残差百分比'].round(2)
                    st.dataframe(display_df, use_container_width=True)
    
    # 手动清理按钮
    if st.session_state.model_trained and st.session_state.df is not None:
        if st.button("🔄 重新检测异常数据", use_container_width=True):
            st.session_state.model_trained = False
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
                
                # 重置训练状态
                st.session_state.model_trained = False
                
                # 保存原始数据
                df_to_save = df.copy()
                save_data(df_to_save)
                
                st.success(f"✅ 数据已上传，共 {len(df)} 行")
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
            
            # 显示清理统计
            if st.session_state.clean_stats is not None and st.session_state.clean_stats['outlier_count'] > 0:
                st.info(f"🧹 已自动剔除 {st.session_state.clean_stats['outlier_count']} 个异常数据，使用 {len(st.session_state.df)} 条干净数据训练")
            
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
            
            # 获取异常数据
            outlier_df = st.session_state.outlier_df if hasattr(st.session_state, 'outlier_df') else None
            
            if st.session_state.last_prediction is not None:
                fig = plot_chart(
                    st.session_state.df,
                    st.session_state.model,
                    st.session_state.poly,
                    st.session_state.mape,
                    residuals=st.session_state.residuals,
                    ci_margin=st.session_state.ci_margin,
                    point_count=st.session_state.last_prediction["point_count"],
                    predicted_time=st.session_state.last_prediction["predicted"],
                    outlier_df=outlier_df
                )
                plot_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                fig = plot_chart(
                    st.session_state.df, 
                    st.session_state.model, 
                    st.session_state.poly, 
                    st.session_state.mape,
                    residuals=st.session_state.residuals,
                    ci_margin=st.session_state.ci_margin,
                    outlier_df=outlier_df
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
