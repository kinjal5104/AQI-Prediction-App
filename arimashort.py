"""
India's AQI Dashboard - Enhanced with LIME + Proper ARIMA Forecasting + Full XAI Suite
ARIMA with walk-forward validation | LIME for local interpretability | SHAP for global
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="India AQI — ARIMA & XAI",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,300&display=swap');

    * { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Space Mono', monospace !important; color: #e8f4f8 !important; }

    .stApp { background: #060e1a !important; }
    .main p, .main span, .main div, .main li, .main label { color: #c5d8e8 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0a1628 !important; border-right: 1px solid #1a3045; }
    [data-testid="stSidebar"] * { color: #c5d8e8 !important; }

    /* Metrics */
    [data-testid="stMetric"] { background: #0d1e33 !important; border: 1px solid #1e3a55 !important; border-radius: 10px; padding: 1rem; }
    [data-testid="stMetricValue"] { color: #7ecfff !important; font-family: 'Space Mono', monospace !important; font-size: 1.8rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #6a9ab8 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background: #0d1e33; color: #6a9ab8 !important; border-radius: 6px 6px 0 0;
        padding: 0.6rem 1.2rem; font-weight: 500; border: 1px solid #1e3a55; border-bottom: none;
        font-family: 'Space Mono', monospace; font-size: 12px;
    }
    .stTabs [aria-selected="true"] { background: #112840 !important; color: #7ecfff !important; border-color: #2a5a80; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e5080 0%, #1a3d60 100%);
        color: #7ecfff; border: 1px solid #2a6090; border-radius: 6px;
        padding: 0.5rem 1.5rem; font-weight: 600; font-family: 'Space Mono', monospace;
        transition: all 0.2s ease; font-size: 13px;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #2a6090 0%, #1e5080 100%); transform: translateY(-1px); }

    /* Cards */
    .glass-card {
        background: rgba(13, 30, 51, 0.8); border: 1px solid rgba(42, 90, 128, 0.4);
        border-radius: 12px; padding: 1.5rem; margin: 0.75rem 0;
        backdrop-filter: blur(8px); box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .highlight-card {
        background: linear-gradient(135deg, #0d2640 0%, #112840 100%);
        border: 1px solid #2a5a80; border-radius: 12px; padding: 1.5rem; margin: 0.75rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #2a1a00 0%, #1a1200 100%);
        border: 1px solid #5a3a00; border-radius: 12px; padding: 1.5rem; margin: 0.75rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #001a0d 0%, #001208 100%);
        border: 1px solid #005a28; border-radius: 12px; padding: 1.5rem; margin: 0.75rem 0;
    }
    .lime-card {
        background: linear-gradient(135deg, #001a1a 0%, #001212 100%);
        border: 2px solid #00695c; border-radius: 12px; padding: 1.5rem; margin: 0.75rem 0;
        box-shadow: 0 4px 20px rgba(0, 105, 92, 0.2);
    }
    .arima-card {
        background: linear-gradient(135deg, #0d1a40 0%, #0d1633 100%);
        border: 2px solid #3a5a9a; border-radius: 12px; padding: 1.5rem; margin: 0.75rem 0;
        box-shadow: 0 4px 20px rgba(58, 90, 154, 0.2);
    }
    .shap-card {
        background: linear-gradient(135deg, #2a0020 0%, #1a0018 100%);
        border: 2px solid #8b1560; border-radius: 12px; padding: 1.5rem; margin: 0.75rem 0;
        box-shadow: 0 4px 20px rgba(139, 21, 96, 0.2);
    }

    /* Pollutant bars */
    .pollutant-row {
        background: #0d1e33; border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; border-left: 3px solid #2a5a80;
        transition: border-left-color 0.2s;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #060e1a; }
    ::-webkit-scrollbar-thumb { background: #2a5a80; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a7aaa; }

    /* Selectbox, inputs */
    .stSelectbox > div > div { background: #0d1e33 !important; border-color: #1e3a55 !important; color: #c5d8e8 !important; }
    .stDateInput > div > div { background: #0d1e33 !important; border-color: #1e3a55 !important; }
    .stSlider > div > div { color: #7ecfff !important; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #1e3a55 !important; border-radius: 8px; }

    /* Spinner */
    .stSpinner > div { border-top-color: #7ecfff !important; }

    /* Info/warning boxes */
    .stInfo { background: #0d1e33 !important; border-color: #2a5a80 !important; color: #c5d8e8 !important; }
    .stWarning { background: #1a1200 !important; border-color: #5a3a00 !important; }
    .stSuccess { background: #001a0d !important; border-color: #005a28 !important; }
    .stError { background: #1a0008 !important; border-color: #5a0020 !important; }
</style>
""", unsafe_allow_html=True)

# ==================== AQI CALCULATION ====================

def calculate_sub_index(concentration, breakpoints):
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= concentration <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
    return breakpoints[-1][3]

def calculate_aqi(pollutants):
    aqi_values = []
    if pollutants.get('PM2.5') is not None and not pd.isna(pollutants.get('PM2.5', np.nan)):
        bp = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)]
        aqi_values.append(calculate_sub_index(pollutants['PM2.5'], bp))
    if pollutants.get('PM10') is not None and not pd.isna(pollutants.get('PM10', np.nan)):
        bp = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,550,401,500)]
        aqi_values.append(calculate_sub_index(pollutants['PM10'], bp))
    return max(aqi_values) if aqi_values else 0

def get_aqi_category(aqi):
    if aqi <= 50:   return {'category':'Good','color':'#22c55e','emoji':'😊','description':'Air quality is satisfactory.','health_impact':'Minimal impact — enjoy outdoor activities!'}
    elif aqi <= 100: return {'category':'Satisfactory','color':'#84cc16','emoji':'🙂','description':'Acceptable for most; risk for some sensitive individuals.','health_impact':'Sensitive groups should limit prolonged exertion.'}
    elif aqi <= 200: return {'category':'Moderate','color':'#eab308','emoji':'😐','description':'Sensitive groups may experience health effects.','health_impact':'General public should reduce heavy outdoor exertion.'}
    elif aqi <= 300: return {'category':'Poor','color':'#f97316','emoji':'😷','description':'Everyone may begin experiencing health effects.','health_impact':'Avoid prolonged outdoor exertion. Sensitive groups stay indoors.'}
    elif aqi <= 400: return {'category':'Very Poor','color':'#ef4444','emoji':'😨','description':'Health alert — risk increased for everyone.','health_impact':'Significantly limit all outdoor activity.'}
    else:            return {'category':'Severe','color':'#dc2626','emoji':'☠️','description':'Emergency conditions — everyone affected.','health_impact':'Avoid ALL outdoor physical activity.'}

# ==================== DATA LOADING ====================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('expanded_2022_2025_data_100days.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='mixed').dt.normalize()
        pollutant_cols = ['PM2.5','PM10','CO','NO','NO2','NH3','O3','SO2']
        for col in pollutant_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        def calc_row_aqi(row):
            return calculate_aqi({'PM2.5': row['PM2.5'], 'PM10': row['PM10']})
        df['AQI'] = df.apply(calc_row_aqi, axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ==================== DAILY VIEW HELPERS ====================

def calculate_daily_aqi(df, station, date):
    date_normalized = pd.to_datetime(date).normalize()
    mask = (df['Station'] == station) & (df['Date'] == date_normalized)
    day_data = df[mask]
    if len(day_data) == 0:
        return None, None
    pollutants = {k: day_data[k].mean() for k in ['PM2.5','PM10','CO','NO2','O3','SO2','NH3','NO']}
    aqi = calculate_aqi(pollutants)
    return aqi, pollutants

def get_hourly_data(df, station, date):
    date_normalized = pd.to_datetime(date).normalize()
    mask = (df['Station'] == station) & (df['Date'] == date_normalized)
    day_data = df[mask].copy()
    if len(day_data) == 0:
        return None
    day_data['Hour'] = pd.to_datetime(day_data['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    hourly_aqi = []
    for hour in range(24):
        hd = day_data[day_data['Hour'] == hour]
        if len(hd) > 0:
            aqi = calculate_aqi({'PM2.5': hd['PM2.5'].mean(), 'PM10': hd['PM10'].mean()})
            hourly_aqi.append({'hour': hour, 'aqi': aqi})
    return pd.DataFrame(hourly_aqi) if hourly_aqi else None

def get_historical_data(df, station, days=30):
    station_data = df[df['Station'] == station].copy()
    if len(station_data) == 0:
        return None
    dates = sorted(station_data['Date'].unique())
    recent_dates = dates[-days:] if len(dates) > days else dates
    historical = []
    for date in recent_dates:
        aqi, _ = calculate_daily_aqi(df, station, date)
        if aqi is not None:
            historical.append({'date': date, 'aqi': aqi})
    return pd.DataFrame(historical) if historical else None

def get_top_cities(df, date, top_n=10):
    city_aqi = []
    for station in df['Station'].unique():
        aqi, _ = calculate_daily_aqi(df, station, date)
        if aqi is not None:
            city_aqi.append({'station': station, 'aqi': aqi})
    if not city_aqi:
        return None
    return pd.DataFrame(city_aqi).nlargest(top_n, 'aqi').reset_index(drop=True)

# ==================== CHART HELPERS ====================

PLOTLY_DARK = dict(
    template="plotly_dark",
    plot_bgcolor='#060e1a',
    paper_bgcolor='#060e1a',
    font=dict(color='#c5d8e8', family='DM Sans'),
    xaxis=dict(gridcolor='#0d1e33', showgrid=True),
    yaxis=dict(gridcolor='#0d1e33', showgrid=True),
)

def dark_fig(height=400, title=None):
    fig = go.Figure()
    layout = dict(**PLOTLY_DARK, height=height,
                  margin=dict(t=50 if title else 30, b=40, l=50, r=30))
    if title:
        layout['title'] = dict(text=title, font=dict(color='#7ecfff', size=16, family='Space Mono'), x=0)
    fig.update_layout(**layout)
    return fig

def create_aqi_card(aqi, station, aqi_info, date_str):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{aqi_info['color']}22 0%,{aqi_info['color']}11 100%);
         border:2px solid {aqi_info['color']}66;border-radius:16px;padding:2rem;text-align:center;margin:1rem 0;
         box-shadow:0 8px 32px {aqi_info['color']}22;">
        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:{aqi_info['color']};
             letter-spacing:3px;text-transform:uppercase;margin-bottom:0.5rem;">AIR QUALITY INDEX</div>
        <div style="font-size:3.5rem;margin:0.5rem 0;">{aqi_info['emoji']}</div>
        <div style="font-family:'Space Mono',monospace;font-size:4rem;color:{aqi_info['color']};
             font-weight:700;line-height:1;">{int(aqi)}</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#c5d8e8;
             margin:0.5rem 0;letter-spacing:2px;">{aqi_info['category'].upper()}</div>
        <div style="font-size:0.85rem;color:#6a9ab8;margin-top:0.5rem;">{station}</div>
        <div style="font-size:0.8rem;color:#4a7a98;">{date_str}</div>
    </div>""", unsafe_allow_html=True)

def create_pollutant_bars(pollutants):
    st.markdown("### 🔬 Pollutant Concentrations")
    info = {
        'PM2.5': {'max':250,'unit':'µg/m³','icon':'💨'},
        'PM10':  {'max':430,'unit':'µg/m³','icon':'🌫️'},
        'CO':    {'max':5,'unit':'mg/m³','icon':'⚠️'},
        'NO2':   {'max':100,'unit':'µg/m³','icon':'🏭'},
        'O3':    {'max':200,'unit':'µg/m³','icon':'☀️'},
        'SO2':   {'max':100,'unit':'µg/m³','icon':'🔥'},
        'NH3':   {'max':400,'unit':'µg/m³','icon':'💧'},
        'NO':    {'max':100,'unit':'µg/m³','icon':'🚗'},
    }
    for key, meta in info.items():
        val = pollutants.get(key)
        if val is not None and not pd.isna(val):
            pct = min((val / meta['max']) * 100, 100)
            color = '#22c55e' if pct < 30 else '#eab308' if pct < 60 else '#ef4444'
            st.markdown(f"""
            <div class="pollutant-row" style="border-left-color:{color};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="font-size:20px;">{meta['icon']}</span>
                        <span style="color:#c5d8e8;font-weight:500;min-width:70px;">{key}</span>
                    </div>
                    <div style="flex:1;margin:0 16px;">
                        <div style="background:#0a1628;height:8px;border-radius:4px;overflow:hidden;">
                            <div style="background:{color};width:{pct}%;height:100%;border-radius:4px;transition:width 0.5s;"></div>
                        </div>
                    </div>
                    <div style="color:#c5d8e8;font-family:'Space Mono',monospace;font-size:13px;
                         min-width:110px;text-align:right;">{val:.2f} {meta['unit']}</div>
                </div>
            </div>""", unsafe_allow_html=True)

def create_hourly_chart(hourly_data, station, date, key_suffix=""):
    if hourly_data is None or len(hourly_data) == 0:
        st.info("No hourly data available for this date.")
        return
    colors = [get_aqi_category(a)['color'] for a in hourly_data['aqi']]
    fig = dark_fig(400, f"Hourly AQI — {date.strftime('%b %d, %Y')}")
    fig.add_trace(go.Bar(x=hourly_data['hour'], y=hourly_data['aqi'],
                         marker=dict(color=colors, opacity=0.85),
                         hovertemplate='<b>%{x}:00</b> → AQI: %{y:.0f}<extra></extra>'))
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=2, title='Hour of Day')
    fig.update_yaxes(title='AQI')
    st.plotly_chart(fig, use_container_width=True, key=f"hourly_{station}_{date.strftime('%Y%m%d')}{key_suffix}")

def create_historical_chart(historical_data, station, key_suffix=""):
    if historical_data is None or len(historical_data) == 0:
        st.info("No historical data available.")
        return
    colors = [get_aqi_category(a)['color'] for a in historical_data['aqi']]
    fig = dark_fig(400, f"30-Day AQI Trend — {station}")
    fig.add_trace(go.Scatter(
        x=historical_data['date'], y=historical_data['aqi'],
        mode='lines+markers',
        line=dict(color='#3a7aaa', width=2),
        marker=dict(color=colors, size=7, line=dict(color='#060e1a', width=1)),
        hovertemplate='<b>%{x|%b %d}</b> → AQI: %{y:.0f}<extra></extra>',
        fill='tozeroy', fillcolor='rgba(58,122,170,0.08)'
    ))
    fig.update_yaxes(title='AQI')
    st.plotly_chart(fig, use_container_width=True, key=f"hist_{station}{key_suffix}")

def create_india_map(df, date, key_suffix=""):
    map_data = []
    for station in df['Station'].unique():
        aqi, _ = calculate_daily_aqi(df, station, date)
        if aqi is not None:
            si = df[df['Station'] == station].iloc[0]
            if not pd.isna(si.get('latitude', np.nan)) and not pd.isna(si.get('longitude', np.nan)):
                map_data.append({'station': station, 'lat': si['latitude'], 'lon': si['longitude'],
                                 'aqi': aqi, 'category': get_aqi_category(aqi)['category']})
    if not map_data:
        st.info("No map data available for this date.")
        return
    map_df = pd.DataFrame(map_data)
    fig = go.Figure(go.Scattermapbox(
        lat=map_df['lat'], lon=map_df['lon'], mode='markers',
        marker=dict(size=12, color=map_df['aqi'],
                    colorscale=[[0,'#22c55e'],[0.25,'#84cc16'],[0.5,'#eab308'],
                                [0.625,'#f97316'],[0.75,'#ef4444'],[1,'#dc2626']],
                    showscale=True,
                    colorbar=dict(title=dict(text='AQI', font=dict(color='#c5d8e8')),
                                  tickfont=dict(color='#c5d8e8'), bgcolor='#0d1e33',
                                  bordercolor='#2a5a80', borderwidth=1)),
        text=map_df['station'],
        customdata=map_df[['aqi','category']],
        hovertemplate='<b>%{text}</b><br>AQI: %{customdata[0]:.0f} (%{customdata[1]})<extra></extra>'
    ))
    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", zoom=4, center=dict(lat=20.5937, lon=78.9629)),
        height=480, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor='#060e1a', font=dict(color='#c5d8e8')
    )
    st.plotly_chart(fig, use_container_width=True, key=f"map_{date.strftime('%Y%m%d')}{key_suffix}")

def display_top_cities(top_cities):
    if top_cities is None or len(top_cities) == 0:
        st.info("No city data available.")
        return
    st.markdown("### 🏙️ Top 10 Cities by AQI")
    for i, row in top_cities.iterrows():
        aqi_info = get_aqi_category(row['aqi'])
        st.markdown(f"""
        <div class="glass-card" style="padding:0.75rem 1rem;margin:0.3rem 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="font-family:'Space Mono',monospace;font-size:13px;color:#4a7a98;
                         min-width:24px;">#{i+1}</div>
                    <span style="color:#c5d8e8;font-weight:500;">{row['station']}</span>
                </div>
                <div style="background:{aqi_info['color']}22;color:{aqi_info['color']};
                     border:1px solid {aqi_info['color']}66;padding:4px 12px;border-radius:6px;
                     font-family:'Space Mono',monospace;font-weight:700;">{int(row['aqi'])}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ==================== PROPER ARIMA FORECASTING ====================

def check_stationarity(series):
    """ADF test for stationarity"""
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return {'adf_statistic': result[0], 'p_value': result[1],
                'is_stationary': result[1] < 0.05, 'critical_values': result[4]}
    except:
        return {'adf_statistic': 0, 'p_value': 1, 'is_stationary': False, 'critical_values': {}}

def find_best_arima_order(series, max_p=3, max_d=2, max_q=3):
    """Grid search for best ARIMA order using AIC"""
    best_aic = np.inf
    best_order = (1, 1, 1)
    series_clean = series.dropna()
    if len(series_clean) < 30:
        return best_order, best_aic
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series_clean, order=(p, d, q))
                    fit = model.fit(method_kwargs={'warn_convergence': False})
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                except:
                    continue
    return best_order, best_aic

def walk_forward_arima_validation(series, order, n_test=14):
    """
    Walk-forward (rolling origin) validation for ARIMA — the PROPER way to evaluate time series models.
    At each step, fit ARIMA on all past data, predict 1 step ahead, then expand the window.
    """
    series_clean = series.dropna()
    if len(series_clean) < n_test + 20:
        n_test = max(7, len(series_clean) // 5)

    train_end = len(series_clean) - n_test
    actuals, predictions = [], []
    pred_dates = series_clean.index[train_end:]

    for i in range(n_test):
        train_slice = series_clean.iloc[:train_end + i]
        try:
            model = ARIMA(train_slice, order=order)
            fit = model.fit(method_kwargs={'warn_convergence': False})
            pred = fit.forecast(steps=1)[0]
            predictions.append(pred)
            actuals.append(series_clean.iloc[train_end + i])
        except Exception:
            if predictions:
                predictions.append(predictions[-1])
            else:
                predictions.append(series_clean.iloc[train_end + i - 1] if i > 0 else series_clean.mean())
            actuals.append(series_clean.iloc[train_end + i])

    actuals = np.array(actuals)
    predictions = np.array(predictions)
    mae  = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2   = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / np.maximum(actuals, 1))) * 100

    return {
        'actuals': actuals, 'predictions': predictions,
        'pred_dates': pred_dates[:len(actuals)],
        'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape
    }

def arima_multi_step_forecast(series, order, steps=14):
    """
    Proper multi-step ARIMA forecast with 80% and 95% confidence intervals.
    Returns forecast values, lower/upper bounds for both CI levels.
    """
    series_clean = series.dropna()
    try:
        model = ARIMA(series_clean, order=order)
        fit = model.fit(method_kwargs={'warn_convergence': False})
        forecast_obj = fit.get_forecast(steps=steps)
        forecast_mean = forecast_obj.predicted_mean
        ci_95 = forecast_obj.conf_int(alpha=0.05)
        ci_80 = forecast_obj.conf_int(alpha=0.20)
        last_date = series_clean.index[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(steps)]
        return {
            'dates': forecast_dates,
            'mean': forecast_mean.values,
            'lower_95': ci_95.iloc[:, 0].values,
            'upper_95': ci_95.iloc[:, 1].values,
            'lower_80': ci_80.iloc[:, 0].values,
            'upper_80': ci_80.iloc[:, 1].values,
            'model_fit': fit,
            'aic': fit.aic,
            'bic': fit.bic
        }
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        return None

def create_arima_forecast_chart(series, forecast_result, station, val_result=None):
    """Rich ARIMA forecast chart with validation overlay, dual CIs, and AQI zone bands"""
    if forecast_result is None:
        return

    # Show last 30 days of history
    history = series.dropna().tail(30)

    fig = dark_fig(520, f"🔮 ARIMA Forecast — {station}")

    # AQI zone bands
    zones = [
        (0, 50, '#22c55e', 'Good'),
        (50, 100, '#84cc16', 'Satisfactory'),
        (100, 200, '#eab308', 'Moderate'),
        (200, 300, '#f97316', 'Poor'),
        (300, 400, '#ef4444', 'Very Poor'),
        (400, 500, '#dc2626', 'Severe'),
    ]
    for lo, hi, color, label in zones:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.04, line_width=0)

    # 95% CI band
    x_ci = list(forecast_result['dates']) + list(forecast_result['dates'])[::-1]
    y_ci = list(np.maximum(forecast_result['upper_95'], 0)) + list(np.maximum(forecast_result['lower_95'], 0))[::-1]
    fig.add_trace(go.Scatter(x=x_ci, y=y_ci, fill='toself', fillcolor='rgba(126,207,255,0.07)',
                             line=dict(color='rgba(0,0,0,0)'), name='95% CI', hoverinfo='skip'))

    # 80% CI band
    x_ci80 = list(forecast_result['dates']) + list(forecast_result['dates'])[::-1]
    y_ci80 = list(np.maximum(forecast_result['upper_80'], 0)) + list(np.maximum(forecast_result['lower_80'], 0))[::-1]
    fig.add_trace(go.Scatter(x=x_ci80, y=y_ci80, fill='toself', fillcolor='rgba(126,207,255,0.12)',
                             line=dict(color='rgba(0,0,0,0)'), name='80% CI', hoverinfo='skip'))

    # Walk-forward validation (if available)
    if val_result is not None:
        fig.add_trace(go.Scatter(
            x=val_result['pred_dates'], y=val_result['predictions'],
            mode='lines+markers', name='Walk-Fwd Validation',
            line=dict(color='#a78bfa', width=2, dash='dot'),
            marker=dict(size=6, color='#a78bfa'),
            hovertemplate='<b>%{x|%b %d}</b><br>Predicted: %{y:.0f}<extra></extra>'
        ))

    # Historical
    hist_colors = [get_aqi_category(a)['color'] for a in history.values]
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values,
        mode='lines+markers', name='Historical AQI',
        line=dict(color='#7ecfff', width=2.5),
        marker=dict(color=hist_colors, size=6, line=dict(color='#060e1a', width=1)),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>AQI: %{y:.0f}<extra></extra>',
        fill='tozeroy', fillcolor='rgba(126,207,255,0.05)'
    ))

    # Forecast
    fc_colors = [get_aqi_category(a)['color'] for a in forecast_result['mean']]
    fig.add_trace(go.Scatter(
        x=forecast_result['dates'], y=np.maximum(forecast_result['mean'], 0),
        mode='lines+markers', name='ARIMA Forecast',
        line=dict(color='#fbbf24', width=3, dash='dash'),
        marker=dict(color=fc_colors, size=10, symbol='diamond', line=dict(color='#060e1a', width=1.5)),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:.0f}<extra></extra>'
    ))

    # Vertical separator
    last_hist_date = history.index[-1]
    fig.add_vline(x=last_hist_date.timestamp() * 1000, line_width=1.5, line_dash='dash', line_color='#4a7a98',
                  annotation_text='Forecast →', annotation_font_color='#7ecfff',
                  annotation_position='top right')

    fig.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(color='#c5d8e8', size=12), bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified', yaxis_title='AQI', xaxis_title='Date',
        yaxis=dict(range=[0, max(500, forecast_result['upper_95'].max() * 1.1)])
    )

    st.plotly_chart(fig, use_container_width=True, key=f"arima_fc_{station}")

# ==================== ARIMA XAI FUNCTIONS ====================

def explain_arima_model(model_fit, series, station):
    st.markdown("## 🧠 ARIMA Model — Explainability")
    st.markdown(f"### Interpreting temporal patterns for **{station}**")

    # ---- Parameters ----
    st.markdown("#### 📋 Model Parameters")
    params = model_fit.params
    pvalues = model_fit.pvalues
    col1, col2, col3 = st.columns(3)
    for idx, (col, prefix, label) in enumerate([(col1, 'ar.', 'AR Coefficients'), (col2, 'ma.', 'MA Coefficients'), (col3, None, 'Model Info')]):
        with col:
            st.markdown(f"**{label}**")
            if prefix:
                sub = [(k, v, pvalues[k]) for k, v in params.items() if k.startswith(prefix)]
                if sub:
                    for k, v, pv in sub:
                        color = '#22c55e' if pv < 0.05 else '#f97316'
                        sig = '✓' if pv < 0.05 else '✗'
                        st.markdown(f"<div style='color:{color};font-family:Space Mono,monospace;font-size:13px;'>"
                                    f"{sig} {k}: {v:.4f} (p={pv:.3f})</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:#4a7a98;font-size:13px;'>None in model</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"**AIC:** `{model_fit.aic:.1f}`")
                st.markdown(f"**BIC:** `{model_fit.bic:.1f}`")
                st.markdown(f"**HQIC:** `{model_fit.hqic:.1f}`")
                st.markdown(f"**σ²:** `{model_fit.params.get('sigma2', 0):.2f}`")

    st.markdown("---")

    # ---- Decomposition ----
    st.markdown("#### 📊 Time Series Decomposition")
    try:
        period = 7
        decomp = seasonal_decompose(series.dropna(), model='additive', period=period)
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
        plt.rcParams['font.family'] = 'monospace'
        for ax, data, title, color in zip(
            axes,
            [series, decomp.trend, decomp.seasonal, decomp.resid],
            ['Original', 'Trend', 'Seasonal (period=7)', 'Residuals'],
            ['#7ecfff', '#22c55e', '#fbbf24', '#f87171']
        ):
            valid_data = data.dropna()
            ax.plot(valid_data.index, valid_data.values, color=color, linewidth=1.5)
            ax.fill_between(valid_data.index, valid_data.values, alpha=0.1, color=color)
            ax.set_title(title, color='#c5d8e8', fontsize=11, pad=4)
            ax.set_facecolor('#060e1a')
            ax.tick_params(colors='#6a9ab8', labelsize=8)
            ax.spines[:].set_color('#0d1e33')
            ax.axhline(0, color='#1e3a55', linewidth=0.5, linestyle='--')
        fig.patch.set_facecolor('#060e1a')
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    except Exception as e:
        st.warning(f"Decomposition unavailable: {e}")

    st.markdown("---")

    # ---- ACF / PACF ----
    st.markdown("#### 📈 Autocorrelation Analysis")
    col1, col2 = st.columns(2)
    for col, plot_fn, title, color in [
        (col1, plot_acf, 'ACF — MA order hint', '#7ecfff'),
        (col2, plot_pacf, 'PACF — AR order hint', '#22c55e')
    ]:
        with col:
            fig, ax = plt.subplots(figsize=(7, 4))
            plot_fn(series.dropna(), ax=ax, lags=20, color=color, alpha=0.5)
            ax.set_title(title, color='#c5d8e8', fontsize=11)
            ax.set_facecolor('#060e1a')
            fig.patch.set_facecolor('#060e1a')
            ax.tick_params(colors='#6a9ab8')
            ax.spines[:].set_color('#0d1e33')
            for line in ax.lines:
                line.set_color(color)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.markdown("---")

    # ---- Residuals ----
    st.markdown("#### 📉 Residuals Diagnostics")
    residuals = model_fit.resid.dropna()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{residuals.mean():.3f}")
    col2.metric("Std Dev", f"{residuals.std():.3f}")
    col3.metric("Skewness", f"{residuals.skew():.3f}")
    col4.metric("Kurtosis", f"{residuals.kurtosis():.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(residuals.index, residuals.values, color='#f87171', linewidth=1)
    ax1.axhline(0, color='#4a7a98', linewidth=1, linestyle='--')
    ax1.fill_between(residuals.index, residuals.values, alpha=0.2, color='#f87171')
    ax1.set_title('Residuals Over Time', color='#c5d8e8', fontsize=11)
    ax1.set_facecolor('#060e1a')
    ax1.tick_params(colors='#6a9ab8', labelsize=8)
    ax1.spines[:].set_color('#0d1e33')

    ax2.hist(residuals, bins=30, color='#7ecfff', alpha=0.7, edgecolor='#0d1e33')
    ax2.axvline(residuals.mean(), color='#22c55e', linewidth=2, linestyle='--',
                label=f'Mean: {residuals.mean():.2f}')
    ax2.set_title('Residuals Distribution', color='#c5d8e8', fontsize=11)
    ax2.set_facecolor('#060e1a')
    ax2.tick_params(colors='#6a9ab8', labelsize=8)
    ax2.spines[:].set_color('#0d1e33')
    ax2.legend(facecolor='#0d1e33', edgecolor='#1e3a55', labelcolor='#c5d8e8', fontsize=9)

    fig.patch.set_facecolor('#060e1a')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Residual quality
    mean_ok = abs(residuals.mean()) < 5
    norm_ok = abs(residuals.skew()) < 1
    st.markdown(f"""
    <div class="arima-card">
        <h4 style="color:#7ecfff;margin-bottom:12px;">✅ Residuals Quality Check</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div style="color:{'#22c55e' if mean_ok else '#f97316'};">
                {'✓' if mean_ok else '⚠'} Zero mean: {'Pass' if mean_ok else 'Slight bias detected'}
            </div>
            <div style="color:{'#22c55e' if norm_ok else '#f97316'};">
                {'✓' if norm_ok else '⚠'} Near-normal: {'Pass' if norm_ok else 'Slight skew detected'}
            </div>
        </div>
        <p style="color:#6a9ab8;font-size:13px;margin-top:12px;">
        Well-behaved residuals (near-zero mean, normal distribution, no patterns) indicate ARIMA has captured the predictable structure.
        </p>
    </div>""", unsafe_allow_html=True)

# ==================== ML DATA PREPARATION ====================

def prepare_ml_features(df, station, lag_days=7):
    station_data = df[df['Station'] == station].copy().sort_values('Date')
    if len(station_data) < 80:
        return None, None, None, None, None

    for lag in range(1, lag_days + 1):
        station_data[f'AQI_lag_{lag}'] = station_data['AQI'].shift(lag)

    station_data['AQI_roll7']  = station_data['AQI'].rolling(7,  min_periods=1).mean()
    station_data['AQI_roll30'] = station_data['AQI'].rolling(30, min_periods=1).mean()
    station_data['AQI_roll7_std'] = station_data['AQI'].rolling(7, min_periods=1).std().fillna(0)
    station_data['DayOfWeek_sin'] = np.sin(2 * np.pi * station_data['DayOfWeek'] / 7)
    station_data['DayOfWeek_cos'] = np.cos(2 * np.pi * station_data['DayOfWeek'] / 7)
    station_data['Month_sin']     = np.sin(2 * np.pi * station_data['Month'] / 12)
    station_data['Month_cos']     = np.cos(2 * np.pi * station_data['Month'] / 12)

    base_feats  = ['PM2.5','PM10','CO','NO','NO2','NH3','O3','SO2']
    lag_feats   = [f'AQI_lag_{i}' for i in range(1, lag_days + 1)]
    roll_feats  = ['AQI_roll7','AQI_roll30','AQI_roll7_std']
    time_feats  = ['DayOfWeek_sin','DayOfWeek_cos','Month_sin','Month_cos']
    all_feats   = [f for f in base_feats + lag_feats + roll_feats + time_feats if f in station_data.columns]

    station_data = station_data.dropna(subset=all_feats + ['AQI'])
    if len(station_data) < 50:
        return None, None, None, None, None

    X, y = station_data[all_feats], station_data['AQI']
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, all_feats

def train_ml_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree':     DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1),
        'XGBoost':           xgb.XGBRegressor(n_estimators=80, max_depth=5, random_state=42,
                                               verbosity=0, eval_metric='rmse'),
    }
    trained = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained[name] = model
        except Exception as e:
            st.warning(f"Could not train {name}: {e}")
    return trained

# ==================== LIME INTEGRATION ====================

def run_lime_explanation(model, X_train, X_test, y_test, feature_names, model_name, n_instances=5):
    """
    Full LIME analysis:
    1. Build tabular explainer on training data
    2. Explain N test instances
    3. Aggregate feature contributions
    4. Show per-instance waterfall + global summary
    """
    if not LIME_AVAILABLE:
        st.warning("LIME not installed. Run: `pip install lime`")
        return

    st.markdown(f"""
    <div class="lime-card">
        <h4 style="color:#26d7c0;margin-bottom:8px;">🍋 LIME — Local Interpretable Model-Agnostic Explanations</h4>
        <p style="color:#80cbc4;font-size:13px;margin:0;">
        LIME works by <strong>perturbing a specific input</strong>, fitting a simple linear model locally around it,
        and reading off which features pushed the prediction up or down. Unlike SHAP (global), LIME explains
        <em>one prediction at a time</em> — ideal for understanding edge cases and anomalies.
        </p>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Building LIME explainer..."):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            mode='regression',
            discretize_continuous=True,
            random_state=42
        )

    # Pick top-5 most interesting samples (highest prediction error)
    preds = model.predict(X_test)
    errors = np.abs(preds - y_test.values)
    if len(errors) < n_instances:
        n_instances = len(errors)
    interesting_idx = np.argsort(errors)[-n_instances:][::-1]

    # ---- Instance selector ----
    st.markdown("#### 🎯 Select Instance to Explain")
    sample_labels = [f"Sample {i+1}: Actual={y_test.iloc[i]:.0f}, Pred={preds[i]:.0f}, Error={errors[i]:.0f}"
                     for i in range(min(20, len(X_test)))]
    selected_idx = st.selectbox("Choose a test instance:", range(len(sample_labels)),
                                 format_func=lambda x: sample_labels[x],
                                 key=f"lime_inst_{model_name}")

    with st.spinner("Running LIME perturbation (~500 samples)..."):
        exp = explainer.explain_instance(
            data_row=X_test[selected_idx],
            predict_fn=model.predict,
            num_features=min(15, len(feature_names)),
            num_samples=500
        )

    lime_list = exp.as_list()  # [(condition_str, weight), ...]
    lime_df = pd.DataFrame(lime_list, columns=['Condition', 'Weight'])
    lime_df = lime_df.sort_values('Weight', key=abs, ascending=False)

    actual_val = y_test.iloc[selected_idx]
    pred_val   = preds[selected_idx]
    error_val  = actual_val - pred_val

    # ---- Prediction summary ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual AQI",    f"{actual_val:.1f}")
    col2.metric("LIME Pred AQI", f"{pred_val:.1f}")
    col3.metric("Error",         f"{error_val:.1f}", delta=f"{error_val:.1f}")

    # ---- Waterfall chart ----
    st.markdown("#### 📊 LIME Feature Contributions (Waterfall)")
    colors = ['#22c55e' if w > 0 else '#ef4444' for w in lime_df['Weight']]

    fig = dark_fig(420, f"LIME Waterfall — {model_name} | Instance #{selected_idx+1}")
    fig.add_trace(go.Bar(
        x=lime_df['Weight'], y=lime_df['Condition'],
        orientation='h',
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>'
    ))
    fig.add_vline(x=0, line_width=1.5, line_color='#4a7a98')
    fig.update_yaxes(autorange='reversed')
    fig.update_xaxes(title='Contribution to Prediction')
    st.plotly_chart(fig, use_container_width=True, key=f"lime_wf_{model_name}_{selected_idx}")

    # ---- Interpretation ----
    top_pos = lime_df[lime_df['Weight'] > 0].head(2)
    top_neg = lime_df[lime_df['Weight'] < 0].head(2)

    pos_list = ''.join([f"<li><b>{r['Condition']}</b> → +{r['Weight']:.3f}</li>" for _, r in top_pos.iterrows()])
    neg_list = ''.join([f"<li><b>{r['Condition']}</b> → {r['Weight']:.3f}</li>" for _, r in top_neg.iterrows()])

    st.markdown(f"""
    <div class="lime-card">
        <h4 style="color:#26d7c0;">📖 LIME Interpretation for Instance #{selected_idx+1}</h4>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px;">
            <div>
                <p style="color:#22c55e;font-weight:600;margin-bottom:6px;">🔺 AQI Increasing Factors:</p>
                <ul style="color:#80cbc4;font-size:13px;">{pos_list if pos_list else '<li>None significant</li>'}</ul>
            </div>
            <div>
                <p style="color:#ef4444;font-weight:600;margin-bottom:6px;">🔻 AQI Decreasing Factors:</p>
                <ul style="color:#80cbc4;font-size:13px;">{neg_list if neg_list else '<li>None significant</li>'}</ul>
            </div>
        </div>
        <p style="color:#4a9a90;font-size:12px;margin-top:12px;">
        ⚡ LIME perturbed this instance 500 times to build a local linear approximation.
        Positive weights push AQI higher; negative weights pull it lower.
        </p>
    </div>""", unsafe_allow_html=True)

    # ---- Aggregate LIME over multiple instances ----
    st.markdown("#### 🌐 Aggregate LIME Importance (Multiple Instances)")
    st.markdown("<div style='color:#6a9ab8;font-size:13px;margin-bottom:8px;'>"
                "Average |LIME weight| across high-error instances to get a global picture.</div>",
                unsafe_allow_html=True)

    all_weights = {fn: [] for fn in feature_names}
    progress = st.progress(0, text="Explaining instances...")
    for j, idx in enumerate(interesting_idx):
        try:
            e = explainer.explain_instance(X_test[idx], model.predict,
                                           num_features=len(feature_names), num_samples=300)
            for cond, w in e.as_list():
                # Match condition back to feature name (LIME uses bins like "0.50 < PM2.5 <= 1.20")
                for fn in feature_names:
                    if fn in cond:
                        all_weights[fn].append(abs(w))
                        break
        except:
            pass
        progress.progress((j + 1) / len(interesting_idx),
                          text=f"Explained {j+1}/{len(interesting_idx)} instances...")
    progress.empty()

    agg_df = pd.DataFrame({
        'feature': list(all_weights.keys()),
        'mean_abs_weight': [np.mean(v) if v else 0 for v in all_weights.values()]
    }).sort_values('mean_abs_weight', ascending=False).head(15)

    fig2 = dark_fig(380, f"LIME Global Feature Importance — {model_name}")
    fig2.add_trace(go.Bar(
        x=agg_df['mean_abs_weight'], y=agg_df['feature'],
        orientation='h',
        marker=dict(color=agg_df['mean_abs_weight'],
                    colorscale=[[0,'#0d4a44'],[0.5,'#26a69a'],[1,'#00e5d4']],
                    line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>Mean |Weight|: %{x:.4f}<extra></extra>'
    ))
    fig2.update_yaxes(autorange='reversed')
    fig2.update_xaxes(title='Mean Absolute LIME Weight')
    st.plotly_chart(fig2, use_container_width=True, key=f"lime_agg_{model_name}")

    return lime_df

# ==================== SHAP INTEGRATION ====================

def run_shap_analysis(model, X_train, X_test, feature_names, model_name, model_type='tree'):
    if not SHAP_AVAILABLE:
        st.warning("SHAP not installed. Run: `pip install shap`")
        return

    st.markdown(f"""
    <div class="shap-card">
        <h4 style="color:#e879c0;margin-bottom:8px;">⚡ SHAP — SHapley Additive exPlanations</h4>
        <p style="color:#f3b8e4;font-size:13px;margin:0;">
        SHAP uses <strong>cooperative game theory</strong> to assign each feature a fair share of the prediction.
        Unlike LIME (local linear), SHAP is <em>globally consistent</em> — the same feature always gets the
        same attribution regardless of context.
        </p>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Computing SHAP values..."):
        try:
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test[:100])
                expected_value = explainer.expected_value
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer.shap_values(X_test[:100])
                expected_value = explainer.expected_value
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 50))
                shap_values = explainer.shap_values(X_test[:50])
                expected_value = explainer.expected_value
        except Exception as e:
            st.error(f"SHAP failed: {e}")
            return

    # Global importance
    shap_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### 📊 SHAP Feature Importance (|mean SHAP|)")
        fig = dark_fig(400)
        fig.add_trace(go.Bar(
            x=shap_imp['importance'].head(15),
            y=shap_imp['feature'].head(15),
            orientation='h',
            marker=dict(color=shap_imp['importance'].head(15),
                        colorscale=[[0,'#4a0030'],[0.5,'#8b1560'],[1,'#e879c0']],
                        line=dict(width=0)),
            hovertemplate='<b>%{y}</b><br>|Mean SHAP|: %{x:.4f}<extra></extra>'
        ))
        fig.update_yaxes(autorange='reversed')
        fig.update_xaxes(title='Mean |SHAP Value|')
        st.plotly_chart(fig, use_container_width=True, key=f"shap_imp_{model_name}")

    with col2:
        st.markdown("#### 🏆 Top Features by SHAP")
        for _, row in shap_imp.head(8).iterrows():
            pct = (row['importance'] / shap_imp['importance'].sum()) * 100
            st.markdown(f"""
            <div style="background:#1a0018;border-radius:6px;padding:8px 12px;margin:4px 0;
                 border-left:3px solid #8b1560;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#c5d8e8;font-size:13px;">{row['feature']}</span>
                    <span style="color:#e879c0;font-family:Space Mono,monospace;font-size:12px;">
                        {pct:.1f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # SHAP summary beeswarm plot
    st.markdown("#### 🐝 SHAP Beeswarm Plot")
    try:
        fig2, ax = plt.subplots(figsize=(11, 6))
        shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names,
                          show=False, max_display=12, plot_type='dot')
        ax = plt.gca()
        ax.set_facecolor('#060e1a')
        fig2 = plt.gcf()
        fig2.patch.set_facecolor('#060e1a')
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('#c5d8e8')
        ax.xaxis.label.set_color('#c5d8e8')
        ax.yaxis.label.set_color('#c5d8e8')
        plt.title(f'SHAP Beeswarm — {model_name}', color='#c5d8e8', fontsize=12, pad=8)
        st.pyplot(fig2, use_container_width=True)
        plt.close()
    except Exception as e:
        st.info(f"Beeswarm plot unavailable: {e}")

    st.markdown("""
    <div class="shap-card" style="margin-top:12px;">
        <h5 style="color:#e879c0;">📖 How to Read SHAP:</h5>
        <ul style="color:#f3b8e4;font-size:13px;">
            <li><strong>Positive SHAP</strong> → feature pushes prediction <em>higher</em></li>
            <li><strong>Negative SHAP</strong> → feature pushes prediction <em>lower</em></li>
            <li><strong>Color (beeswarm)</strong> → Red = high value, Blue = low value</li>
            <li><strong>X-axis width</strong> → how strongly that instance was affected</li>
        </ul>
    </div>""", unsafe_allow_html=True)

# ==================== FULL ML XAI DASHBOARD ====================

def create_ml_xai_dashboard(model, X_train, X_test, y_train, y_test, feature_names, model_name):
    model_type = 'tree' if model_name in ('Random Forest','Decision Tree','XGBoost') else 'linear'

    xai_tabs = st.tabs(["📊 Feature Importance", "🍋 LIME", "⚡ SHAP", "🎯 Predictions"])

    with xai_tabs[0]:
        st.markdown("### 📊 Permutation Feature Importance")
        with st.spinner("Calculating permutation importance..."):
            perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({'feature': feature_names,
                               'importance_mean': perm.importances_mean,
                               'importance_std':  perm.importances_std}).sort_values('importance_mean', ascending=False)

        fig = dark_fig(460, f"Permutation Importance — {model_name}")
        fig.add_trace(go.Bar(
            x=imp_df['importance_mean'].head(15), y=imp_df['feature'].head(15),
            orientation='h',
            error_x=dict(type='data', array=imp_df['importance_std'].head(15), visible=True,
                         color='rgba(126,207,255,0.5)'),
            marker=dict(color=imp_df['importance_mean'].head(15),
                        colorscale=[[0,'#0d2640'],[0.5,'#1e5080'],[1,'#7ecfff']],
                        line=dict(width=0)),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f} ±%{error_x.array:.4f}<extra></extra>'
        ))
        fig.update_yaxes(autorange='reversed')
        fig.update_xaxes(title='Permutation Importance')
        st.plotly_chart(fig, use_container_width=True, key=f"perm_imp_{model_name}")

        if hasattr(model, 'feature_importances_'):
            st.markdown("#### 🌳 Built-in Feature Importance (Gini / Gain)")
            bi_df = pd.DataFrame({'feature': feature_names,
                                  'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(12)
            fig2 = dark_fig(380)
            fig2.add_trace(go.Bar(x=bi_df['importance'], y=bi_df['feature'], orientation='h',
                                  marker=dict(color=bi_df['importance'],
                                              colorscale=[[0,'#1a0040'],[0.5,'#5c35b8'],[1,'#a78bfa']],
                                              line=dict(width=0)),
                                  hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'))
            fig2.update_yaxes(autorange='reversed')
            st.plotly_chart(fig2, use_container_width=True, key=f"gini_imp_{model_name}")

    with xai_tabs[1]:
        run_lime_explanation(model, X_train, X_test, y_test, feature_names, model_name)

    with xai_tabs[2]:
        run_shap_analysis(model, X_train, X_test, feature_names, model_name, model_type)

    with xai_tabs[3]:
        st.markdown("### 🎯 Individual Prediction Analysis")
        preds = model.predict(X_test)
        residuals = y_test.values - preds

        col1, col2 = st.columns(2)
        with col1:
            fig = dark_fig(360, "Actual vs Predicted")
            fig.add_trace(go.Scatter(x=y_test.values, y=preds, mode='markers',
                                     marker=dict(color='#7ecfff', size=5, opacity=0.6),
                                     hovertemplate='Actual: %{x:.0f}<br>Pred: %{y:.0f}<extra></extra>'))
            mn, mx = y_test.min(), y_test.max()
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                     line=dict(color='#ef4444', dash='dash', width=1.5),
                                     name='Perfect fit'))
            fig.update_xaxes(title='Actual AQI')
            fig.update_yaxes(title='Predicted AQI')
            st.plotly_chart(fig, use_container_width=True, key=f"scatter_pred_{model_name}")

        with col2:
            fig2 = dark_fig(360, "Residuals Distribution")
            fig2.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                        marker=dict(color='#a78bfa', opacity=0.7, line=dict(width=0)),
                                        hovertemplate='Residual: %{x:.0f}<br>Count: %{y}<extra></extra>'))
            fig2.add_vline(x=0, line_color='#ef4444', line_dash='dash', line_width=1.5)
            fig2.update_xaxes(title='Residual (Actual - Predicted)')
            fig2.update_yaxes(title='Count')
            st.plotly_chart(fig2, use_container_width=True, key=f"resid_hist_{model_name}")

# ==================== MAIN APPLICATION ====================

def main():
    # Header
    st.markdown("# 🌫️ INDIA AQI DASHBOARD")
    st.markdown("### ARIMA Time-Series Forecasting + LIME · SHAP · XAI")
    st.markdown("---")

    with st.spinner("Loading data..."):
        df = load_data()
    if df is None:
        st.error("Failed to load data. Ensure `expanded_2022_2025_data_100days.csv` is present.")
        return

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("## ⚙️ Controls")

        available_years = sorted(df['Year'].unique())
        selected_year = st.selectbox("📅 Year", available_years, index=len(available_years)-1)

        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        available_months = sorted(df[df['Year'] == selected_year]['Month'].unique())
        selected_month = st.selectbox("📆 Month", available_months,
                                       format_func=lambda x: month_names[x-1],
                                       index=len(available_months)-1)

        stations = sorted(df['Station'].unique())
        selected_station = st.selectbox("📍 Station", stations)

        available_dates = sorted(df[(df['Station'] == selected_station) &
                                    (df['Year'] == selected_year) &
                                    (df['Month'] == selected_month)]['Date'].unique())
        if not available_dates:
            st.error("No data for selected filters.")
            return

        selected_date = st.date_input(
            "🗓️ Date",
            value=available_dates[-1].date(),
            min_value=available_dates[0].date(),
            max_value=available_dates[-1].date()
        )

        st.markdown("---")
        st.markdown("### 🔧 ARIMA Settings")
        use_auto_arima = st.checkbox("Auto-tune ARIMA (AIC grid)", value=True)
        forecast_days = st.slider("Forecast horizon (days)", 7, 21, 14)

        if not use_auto_arima:
            c1, c2, c3 = st.columns(3)
            p = c1.slider("p", 0, 3, 1)
            d = c2.slider("d", 0, 2, 1)
            q = c3.slider("q", 0, 3, 1)
            arima_order = (p, d, q)
        else:
            arima_order = None

        st.markdown("---")
        st.markdown("### 🧠 XAI Settings")
        enable_xai = st.checkbox("Enable XAI", value=True)

        st.markdown("---")
        st.info(f"""
        **📊 Coverage**
       
        **{len(df['Station'].unique())}** stations
       
        **{df['Date'].min().strftime('%b %Y')} → {df['Date'].max().strftime('%b %Y')}**
       
        **Models:** ARIMA (primary), RF, XGB, DT, LR
       
        **XAI:** LIME · SHAP · Permutation Importance
        """)

        c1, c2 = st.columns(2)
        c1.metric("Stations", len(df['Station'].unique()))
        c2.metric("Records", f"{len(df):,}")

    # ==================== TABS ====================
    selected_date_dt = pd.to_datetime(selected_date)
    aqi, pollutants = calculate_daily_aqi(df, selected_station, selected_date_dt)

    if aqi is None:
        st.warning(f"No data for {selected_station} on {selected_date.strftime('%B %d, %Y')}")
        return

    aqi_info = get_aqi_category(aqi)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📅 Daily View",
        "🔮 ARIMA Forecast",
        "🧠 ARIMA XAI",
        "🍋 LIME + ⚡ SHAP",
        "📊 Analytics",
        "🗺️ Station Map",
        "🏆 Model Comparison"
    ])

    # ==================== TAB 1: DAILY VIEW ====================
    with tab1:
        st.markdown(f"## {selected_station}")
        st.markdown(f"#### {selected_date.strftime('%A, %B %d, %Y')}")

        col1, col2 = st.columns([1, 2])
        with col1:
            create_aqi_card(aqi, selected_station, aqi_info, selected_date.strftime('%B %d, %Y'))
            st.markdown(f"""
            <div class="glass-card">
                <p style="color:#c5d8e8;line-height:1.6;">{aqi_info['description']}</p>
                <p style="color:#6a9ab8;line-height:1.6;margin-top:8px;">
                    <strong>💊 Advice:</strong> {aqi_info['health_impact']}
                </p>
            </div>""", unsafe_allow_html=True)
            create_pollutant_bars(pollutants)
            st.markdown("---")
            display_top_cities(get_top_cities(df, selected_date_dt))

        with col2:
            st.markdown("### 📊 Hourly AQI Pattern")
            create_hourly_chart(get_hourly_data(df, selected_station, selected_date_dt),
                                selected_station, selected_date_dt, "_d1")
            st.markdown("### 📈 Historical 30-Day Trend")
            create_historical_chart(get_historical_data(df, selected_station), selected_station, "_d1")
            st.markdown("### 🗺️ All-India AQI Map")
            create_india_map(df, selected_date_dt, "_d1")

    # ==================== TAB 2: ARIMA FORECAST ====================
    with tab2:
        st.markdown(f"## 🔮 ARIMA Forecast — {selected_station}")

        daily_series = df[df['Station'] == selected_station].groupby('Date')['AQI'].mean().sort_index()

        if len(daily_series) < 30:
            st.warning(f"Need ≥30 days of data. Currently have {len(daily_series)}.")
        else:
            # ADF test
            with st.expander("📈 Stationarity Check (ADF Test)", expanded=False):
                stat_result = check_stationarity(daily_series)
                col1, col2, col3 = st.columns(3)
                col1.metric("ADF Statistic", f"{stat_result['adf_statistic']:.3f}")
                col2.metric("p-value", f"{stat_result['p_value']:.4f}")
                col3.metric("Stationary?", "✅ Yes" if stat_result['is_stationary'] else "❌ No")
                st.markdown(f"""
                <div class="{'success-card' if stat_result['is_stationary'] else 'warning-card'}">
                    {"✅ Series is stationary — d=0 may be appropriate." if stat_result['is_stationary']
                     else "⚠️ Series is non-stationary — differencing (d≥1) recommended."}
                </div>""", unsafe_allow_html=True)

            # Find best order
            with st.spinner("🔬 Tuning ARIMA..."):
                if use_auto_arima:
                    best_order, best_aic = find_best_arima_order(daily_series)
                    st.success(f"✅ Best order: ARIMA{best_order} | AIC={best_aic:.2f}")
                else:
                    best_order = arima_order
                    best_aic = None

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("AR(p)", best_order[0])
            col2.metric("I(d)",  best_order[1])
            col3.metric("MA(q)", best_order[2])
            col4.metric("AIC",   f"{best_aic:.1f}" if best_aic else "—")

            # Walk-forward validation
            with st.spinner("📊 Running walk-forward validation..."):
                val_result = walk_forward_arima_validation(daily_series, best_order, n_test=14)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² (WF)", f"{val_result['R2']:.3f}")
            c2.metric("MAE",      f"{val_result['MAE']:.1f}")
            c3.metric("RMSE",     f"{val_result['RMSE']:.1f}")
            c4.metric("MAPE",     f"{val_result['MAPE']:.1f}%")

            # Forecast
            with st.spinner("🔮 Generating forecast..."):
                fc = arima_multi_step_forecast(daily_series, best_order, steps=forecast_days)

            create_arima_forecast_chart(daily_series, fc, selected_station, val_result)

            if fc:
                # Forecast table
                st.markdown("### 📋 Forecast Details")
                fc_df = pd.DataFrame({
                    'Date':     [d.strftime('%a %b %d') for d in fc['dates']],
                    'Forecast': np.maximum(fc['mean'], 0).round(0).astype(int),
                    'Low 80%':  np.maximum(fc['lower_80'], 0).round(0).astype(int),
                    'High 80%': np.maximum(fc['upper_80'], 0).round(0).astype(int),
                    'Low 95%':  np.maximum(fc['lower_95'], 0).round(0).astype(int),
                    'High 95%': np.maximum(fc['upper_95'], 0).round(0).astype(int),
                    'Category': [get_aqi_category(a)['category'] for a in fc['mean']],
                })

                col1, col2 = st.columns([2, 1])
                with col1:
                    for _, row in fc_df.iterrows():
                        cat_info = get_aqi_category(float(row['Forecast']))
                        st.markdown(f"""
                        <div class="glass-card" style="padding:0.75rem 1rem;margin:0.3rem 0;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div>
                                    <div style="font-family:Space Mono,monospace;color:#7ecfff;
                                         font-size:14px;">{row['Date']}</div>
                                    <div style="color:#4a7a98;font-size:12px;">
                                        95% CI: {row['Low 95%']} – {row['High 95%']} &nbsp;|&nbsp;
                                        80% CI: {row['Low 80%']} – {row['High 80%']}
                                    </div>
                                </div>
                                <div style="text-align:right;">
                                    <span style="font-family:Space Mono,monospace;font-size:1.5rem;
                                         color:{cat_info['color']};font-weight:700;">{row['Forecast']}</span>
                                    <span style="display:block;color:#c5d8e8;font-size:12px;">
                                        {cat_info['category']} {cat_info['emoji']}</span>
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                with col2:
                    avg  = int(np.mean(fc['mean']))
                    peak = int(np.max(fc['mean']))
                    best = int(np.min(fc['mean']))
                    avg_unc = int(np.mean(fc['upper_95'] - fc['lower_95']) / 2)
                    st.metric("📈 Average", avg)
                    st.metric("🔴 Peak",    peak)
                    st.metric("🟢 Best",    best)
                    st.metric("📊 ±Uncert", avg_unc)

    # ==================== TAB 3: ARIMA XAI ====================
    with tab3:
        if not enable_xai:
            st.info("Enable XAI in the sidebar.")
        else:
            daily_series = df[df['Station'] == selected_station].groupby('Date')['AQI'].mean().sort_index()
            if len(daily_series) < 30:
                st.warning(f"Need ≥30 days for ARIMA XAI. Have {len(daily_series)}.")
            else:
                with st.spinner("Training ARIMA for XAI..."):
                    best_order, _ = find_best_arima_order(daily_series) if use_auto_arima \
                                    else (arima_order, None)
                    try:
                        arima_fit = ARIMA(daily_series.dropna(), order=best_order).fit(
                            method_kwargs={'warn_convergence': False})
                        explain_arima_model(arima_fit, daily_series, selected_station)
                    except Exception as e:
                        st.error(f"ARIMA XAI failed: {e}")

    # ==================== TAB 4: LIME + SHAP ====================
    with tab4:
        if not enable_xai:
            st.info("Enable XAI in the sidebar.")
        else:
            st.markdown(f"## 🧠 ML Model Explainability — LIME + SHAP")
            st.markdown(f"### {selected_station}")

            ml_data = prepare_ml_features(df, selected_station)
            if ml_data[0] is None:
                st.warning("Need ≥80 days of data for ML XAI.")
            else:
                X_train, X_test, y_train, y_test, feature_names = ml_data

                with st.spinner("Training ML models..."):
                    ml_models = train_ml_models(X_train, y_train)

                if not ml_models:
                    st.error("No ML models trained successfully.")
                else:
                    selected_ml = st.selectbox("Select Model for XAI:",
                                               list(ml_models.keys()), key="xai_model_select")
                    model = ml_models[selected_ml]

                    # Quick metrics
                    preds = model.predict(X_test)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("R²",   f"{r2_score(y_test, preds):.3f}")
                    c2.metric("MAE",  f"{mean_absolute_error(y_test, preds):.1f}")
                    c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.1f}")
                    c4.metric("MAPE", f"{np.mean(np.abs((y_test.values - preds)/np.maximum(y_test.values,1)))*100:.1f}%")

                    st.markdown("---")
                    create_ml_xai_dashboard(model, X_train, X_test, y_train, y_test,
                                            feature_names, selected_ml)

    # ==================== TAB 5: ANALYTICS ====================
    with tab5:
        month_names_full = ['January','February','March','April','May','June',
                            'July','August','September','October','November','December']
        st.markdown(f"## 📊 Analytics — {selected_station}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### {month_names_full[selected_month-1]} {selected_year}")
            md = df[(df['Year']==selected_year)&(df['Month']==selected_month)&(df['Station']==selected_station)]
            if len(md) > 0:
                m_aqi = calculate_aqi({'PM2.5': md['PM2.5'].mean(), 'PM10': md['PM10'].mean()})
                m_info = get_aqi_category(m_aqi)
                st.metric("Monthly Avg AQI", f"{m_aqi:.1f}", m_info['category'])
                st.write(f"Days with data: **{md['Date'].nunique()}**")
                st.write(f"Total records: **{len(md):,}**")

        with col2:
            st.markdown(f"#### Year {selected_year}")
            yd = df[(df['Year']==selected_year)&(df['Station']==selected_station)]
            if len(yd) > 0:
                y_aqi = calculate_aqi({'PM2.5': yd['PM2.5'].mean(), 'PM10': yd['PM10'].mean()})
                y_info = get_aqi_category(y_aqi)
                st.metric("Yearly Avg AQI", f"{y_aqi:.1f}", y_info['category'])
                st.write(f"Months with data: **{yd['Month'].nunique()}**")
                st.write(f"Days with data: **{yd['Date'].nunique()}**")

        st.markdown("---")
        st.markdown("### 📊 Year-over-Year Trend")
        yoy = []
        for yr in sorted(df['Year'].unique()):
            d = df[(df['Year']==yr)&(df['Station']==selected_station)]
            if len(d) > 0:
                yoy.append({'year': yr, 'aqi': calculate_aqi({'PM2.5': d['PM2.5'].mean(), 'PM10': d['PM10'].mean()})})
        if yoy:
            yoy_df = pd.DataFrame(yoy)
            fig = dark_fig(380, f"Year-over-Year AQI — {selected_station}")
            fig.add_trace(go.Bar(
                x=yoy_df['year'].astype(str), y=yoy_df['aqi'],
                marker=dict(color=yoy_df['aqi'],
                            colorscale=[[0,'#0d4060'],[0.5,'#3a7aaa'],[1,'#ef4444']],
                            line=dict(width=0)),
                hovertemplate='<b>%{x}</b><br>Avg AQI: %{y:.1f}<extra></extra>'
            ))
            fig.update_yaxes(title='Average AQI')
            st.plotly_chart(fig, use_container_width=True, key="yoy_chart")

    # ==================== TAB 6: STATION MAP ====================
    with tab6:
        st.markdown(f"## 🗺️ All-India AQI Map")
        st.markdown(f"#### {selected_date.strftime('%B %d, %Y')}")
        create_india_map(df, selected_date_dt, "_t6")

        top_c = get_top_cities(df, selected_date_dt, 20)
        if top_c is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔴 Most Polluted")
                for _, row in top_c.head(10).iterrows():
                    ai = get_aqi_category(row['aqi'])
                    st.markdown(f"""
                    <div class="glass-card" style="padding:0.6rem 1rem;margin:0.25rem 0;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="color:#c5d8e8;">{row['station']}</span>
                            <span style="color:{ai['color']};font-family:Space Mono,monospace;
                                 font-weight:700;">{int(row['aqi'])}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown("#### 🟢 Cleanest Cities")
                for _, row in top_c.sort_values('aqi').head(10).iterrows():
                    ai = get_aqi_category(row['aqi'])
                    st.markdown(f"""
                    <div class="glass-card" style="padding:0.6rem 1rem;margin:0.25rem 0;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="color:#c5d8e8;">{row['station']}</span>
                            <span style="color:{ai['color']};font-family:Space Mono,monospace;
                                 font-weight:700;">{int(row['aqi'])}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

    # ==================== TAB 7: MODEL COMPARISON ====================
    with tab7:
        st.markdown("## 🏆 Model Comparison — ARIMA vs ML")

        daily_series = df[df['Station'] == selected_station].groupby('Date')['AQI'].mean().sort_index()

        if len(daily_series) < 40:
            st.warning(f"Need ≥40 days. Have {len(daily_series)}.")
        else:
            with st.spinner("Running all models..."):
                best_order, best_aic = find_best_arima_order(daily_series) if use_auto_arima \
                                       else (arima_order, None)
                val = walk_forward_arima_validation(daily_series, best_order)

                ml_data = prepare_ml_features(df, selected_station)
                ml_results = {}
                if ml_data[0] is not None:
                    X_train, X_test, y_train, y_test, feature_names = ml_data
                    ml_models = train_ml_models(X_train, y_train)
                    for name, model in ml_models.items():
                        p = model.predict(X_test)
                        ml_results[name] = {
                            'R2':   r2_score(y_test, p),
                            'MAE':  mean_absolute_error(y_test, p),
                            'RMSE': np.sqrt(mean_squared_error(y_test, p)),
                            'MAPE': np.mean(np.abs((y_test.values - p)/np.maximum(y_test.values,1)))*100,
                        }

            # Table
            rows = [{'Model': f'ARIMA{best_order}', 'Type':'Time Series',
                     'R²': val['R2'], 'MAE': val['MAE'], 'RMSE': val['RMSE'], 'MAPE': f"{val['MAPE']:.1f}%"}]
            for name, m in ml_results.items():
                rows.append({'Model': name, 'Type': 'ML',
                             'R²': m['R2'], 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'MAPE': f"{m['MAPE']:.1f}%"})

            cmp_df = pd.DataFrame(rows).sort_values('R²', ascending=False)
            st.dataframe(cmp_df.style.format({'R²':'{:.3f}','MAE':'{:.1f}','RMSE':'{:.1f}'}),
                         use_container_width=True)

            # Bar chart
            fig = dark_fig(420, "R² Score Comparison (Higher = Better)")
            colors = ['#7ecfff' if 'ARIMA' in r else '#a78bfa' for r in cmp_df['Model']]
            fig.add_trace(go.Bar(x=cmp_df['Model'], y=cmp_df['R²'],
                                 marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
                                 hovertemplate='<b>%{x}</b><br>R²: %{y:.3f}<extra></extra>'))
            fig.update_yaxes(title='R² Score', range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True, key="cmp_bar")

            # Why ARIMA
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="arima-card">
                    <h4 style="color:#7ecfff;">✅ ARIMA Strengths</h4>
                    <ul style="color:#9cb8d8;font-size:13px;margin-top:10px;">
                        <li><b>Temporal awareness</b> — today depends on yesterday</li>
                        <li><b>Trend & seasonality</b> — I(d) removes unit roots</li>
                        <li><b>Confidence intervals</b> — uncertainty quantified</li>
                        <li><b>No feature engineering</b> — works on raw time series</li>
                        <li><b>Walk-forward validation</b> — proper TS evaluation</li>
                    </ul>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="warning-card">
                    <h4 style="color:#fbbf24;">⚠️ ML Limitations for Time Series</h4>
                    <ul style="color:#c5a060;font-size:13px;margin-top:10px;">
                        <li><b>No native ordering</b> — treats rows as i.i.d.</li>
                        <li><b>Lag leakage risk</b> — future data can bleed in</li>
                        <li><b>No uncertainty</b> — point estimates only</li>
                        <li><b>Feature dependence</b> — needs manual lags/rolls</li>
                        <li><b>Cross-val danger</b> — random split = data leakage</li>
                    </ul>
                </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:1.5rem;color:#4a7a98;">
        <p style="font-family:Space Mono,monospace;font-size:12px;margin-bottom:6px;">
            📊 CPCB Data &nbsp;|&nbsp; AQI by Indian Standards &nbsp;|&nbsp;
            ARIMA · LIME · SHAP · Walk-Forward Validation
        </p>
        <p style="font-size:11px;">Built with Streamlit · Made for India 🇮🇳</p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
