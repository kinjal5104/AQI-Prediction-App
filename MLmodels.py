"""
India's AQI Dashboard - Complete with ARIMA & Explainable AI
ARIMA is superior for time series due to temporal dependencies and seasonality
Decision Trees lack temporal understanding - ARIMA captures autocorrelation and trends
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

# ==================== ML & XAI IMPORTS ====================
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# For SHAP explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="India's AQI Dashboard - ARIMA & XAI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0a1929 !important;
    }
    
    .main p, .main span, .main div, .main li, .main label {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    .arima-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 3px solid #5c6bc0;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(28, 58, 116, 0.3);
        transition: all 0.3s ease;
    }
    
    .decision-tree-card {
        background: linear-gradient(135deg, #2e7d32 0%, #388e3c 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 3px solid #4caf50;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.2);
    }
    
    .comparison-card {
        background: linear-gradient(135deg, #6a1b9a 0%, #8e24aa 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 3px solid #ab47bc;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(142, 36, 170, 0.3);
    }
    
    .xai-explanation {
        background: linear-gradient(135deg, #00695c 0%, #00897b 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 3px solid #26a69a;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(38, 166, 154, 0.3);
    }
    
    .model-metrics {
        background: rgba(30, 73, 118, 0.6);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .accuracy-high { border-left-color: #4caf50 !important; }
    .accuracy-medium { border-left-color: #ffc107 !important; }
    .accuracy-low { border-left-color: #f44336 !important; }
    
    .arima-param {
        background: rgba(92, 107, 192, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid #5c6bc0;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0d1f30 !important;
        border-right: 2px solid #1e3a52;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stMetric"] {
        background-color: #132f4c !important;
        border: 2px solid #1e3a52 !important;
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #132f4c;
        color: #b0bec5 !important;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: 2px solid #1e3a52;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e4976;
        color: #ffffff !important;
        border-color: #5c6bc0;
    }
    
    .custom-card {
        background: rgba(30, 73, 118, 0.6);
        border-radius: 16px;
        padding: 24px;
        border: 2px solid rgba(92, 107, 192, 0.3);
        backdrop-filter: blur(10px);
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .pollutant-bar {
        background-color: #132f4c;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #5c6bc0;
        transition: all 0.3s ease;
    }
    
    .pollutant-bar:hover {
        background-color: #1e4976;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(92, 107, 192, 0.3);
    }
    
    .city-card {
        background-color: #132f4c;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #1e3a52;
        transition: all 0.3s ease;
    }
    
    .city-card:hover {
        background-color: #1e4976;
        transform: translateX(8px);
        border-color: #5c6bc0;
    }
    
    .aqi-card {
        background: linear-gradient(135deg, #1e4976 0%, #132f4c 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #1e3a52;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1929;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #5c6bc0;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #7986cb;
    }
    
    .stButton > button {
        background-color: #5c6bc0;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #7986cb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(92, 107, 192, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==================== AQI CALCULATION FUNCTIONS ====================

def calculate_sub_index(concentration, breakpoints):
    """Calculate AQI sub-index for a pollutant"""
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= concentration <= C_high:
            sub_index = ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
            return sub_index
    return breakpoints[-1][3]

def calculate_aqi(pollutants):
    """Calculate overall AQI based on PM2.5 and PM10 concentrations"""
    aqi_values = []
    
    if pollutants.get('PM2.5') is not None and not pd.isna(pollutants['PM2.5']):
        pm25 = pollutants['PM2.5']
        breakpoints = [
            (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
            (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)
        ]
        aqi_values.append(calculate_sub_index(pm25, breakpoints))
    
    if pollutants.get('PM10') is not None and not pd.isna(pollutants['PM10']):
        pm10 = pollutants['PM10']
        breakpoints = [
            (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
            (251, 350, 201, 300), (351, 430, 301, 400), (431, 550, 401, 500)
        ]
        aqi_values.append(calculate_sub_index(pm10, breakpoints))
    
    return max(aqi_values) if aqi_values else 0

def get_aqi_category(aqi):
    """Get AQI category, color, and health information"""
    if aqi <= 50:
        return {
            'category': 'Good',
            'color': '#4caf50',
            'emoji': '😊',
            'description': 'Air quality is satisfactory, and air pollution poses little or no risk.',
            'health_impact': 'Minimal impact. Enjoy outdoor activities!'
        }
    elif aqi <= 100:
        return {
            'category': 'Satisfactory',
            'color': '#8bc34a',
            'emoji': '🙂',
            'description': 'Air quality is acceptable. However, there may be a risk for some people.',
            'health_impact': 'Acceptable for most, sensitive individuals should consider limiting prolonged outdoor exertion.'
        }
    elif aqi <= 200:
        return {
            'category': 'Moderate',
            'color': '#ffc107',
            'emoji': '😐',
            'description': 'Members of sensitive groups may experience health effects.',
            'health_impact': 'General public and sensitive groups should reduce prolonged or heavy outdoor exertion.'
        }
    elif aqi <= 300:
        return {
            'category': 'Poor',
            'color': '#ff9800',
            'emoji': '😷',
            'description': 'Everyone may begin to experience health effects.',
            'health_impact': 'General public should avoid prolonged or heavy exertion. Sensitive groups should limit outdoor activity.'
        }
    elif aqi <= 400:
        return {
            'category': 'Very Poor',
            'color': '#f44336',
            'emoji': '😨',
            'description': 'Health alert: The risk of health effects is increased for everyone.',
            'health_impact': 'General public should significantly limit outdoor exertion. Sensitive groups should avoid outdoor activity.'
        }
    else:
        return {
            'category': 'Severe',
            'color': '#d32f2f',
            'emoji': '☠️',
            'description': 'Health warning of emergency conditions: everyone is more likely to be affected.',
            'health_impact': 'Everyone should avoid all outdoor physical activity. Move activities indoors or reschedule.'
        }

# ==================== DATA LOADING ====================

@st.cache_data
def load_data():
    """Load and preprocess AQI data"""
    try:
        df = pd.read_csv('2022_2025_data.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='mixed').dt.normalize()
        
        # Convert pollutant columns
        pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO', 'NO2', 'NH3', 'O3', 'SO2']
        for col in pollutant_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        # Calculate AQI for each record
        def calculate_row_aqi(row):
            pollutants = {'PM2.5': row['PM2.5'], 'PM10': row['PM10']}
            return calculate_aqi(pollutants)
        
        df['AQI'] = df.apply(calculate_row_aqi, axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ==================== DAILY VIEW FUNCTIONS ====================

def calculate_daily_aqi(df, station, date):
    """Calculate daily average AQI for a specific station and date"""
    date_normalized = pd.to_datetime(date).normalize()
    mask = (df['Station'] == station) & (df['Date'] == date_normalized)
    day_data = df[mask]
    
    if len(day_data) == 0:
        return None, None
    
    # Calculate average pollutant concentrations
    pollutants = {
        'PM2.5': day_data['PM2.5'].mean(),
        'PM10': day_data['PM10'].mean(),
        'CO': day_data['CO'].mean(),
        'NO2': day_data['NO2'].mean(),
        'O3': day_data['O3'].mean(),
        'SO2': day_data['SO2'].mean(),
        'NH3': day_data['NH3'].mean(),
        'NO': day_data['NO'].mean()
    }
    
    # Calculate overall AQI
    aqi = calculate_aqi(pollutants)
    return aqi, pollutants

def get_hourly_data(df, station, date):
    """Get hourly AQI data for a specific station and date"""
    date_normalized = pd.to_datetime(date).normalize()
    mask = (df['Station'] == station) & (df['Date'] == date_normalized)
    day_data = df[mask].copy()
    
    if len(day_data) == 0:
        return None
    
    # Extract hour from Time column
    day_data['Hour'] = pd.to_datetime(day_data['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    
    # Calculate AQI for each hour
    hourly_aqi = []
    for hour in range(24):
        hour_data = day_data[day_data['Hour'] == hour]
        if len(hour_data) > 0:
            pollutants = {
                'PM2.5': hour_data['PM2.5'].mean(),
                'PM10': hour_data['PM10'].mean()
            }
            aqi = calculate_aqi(pollutants)
            hourly_aqi.append({'hour': hour, 'aqi': aqi})
    
    return pd.DataFrame(hourly_aqi) if hourly_aqi else None

def get_historical_data(df, station, days=30):
    """Get historical AQI data for the last N days"""
    station_data = df[df['Station'] == station].copy()
    if len(station_data) == 0:
        return None
    
    # Get unique dates and sort
    dates = sorted(station_data['Date'].unique())
    
    # Take last N days
    recent_dates = dates[-days:] if len(dates) > days else dates
    
    # Calculate daily AQI for each date
    historical = []
    for date in recent_dates:
        aqi, _ = calculate_daily_aqi(df, station, date)
        if aqi is not None:
            historical.append({
                'date': date,
                'aqi': aqi
            })
    
    return pd.DataFrame(historical) if historical else None

def get_top_cities(df, date, top_n=10):
    """Get top N cities by AQI for a specific date"""
    stations = df['Station'].unique()
    city_aqi = []
    
    # Calculate AQI for each station
    for station in stations:
        aqi, _ = calculate_daily_aqi(df, station, date)
        if aqi is not None:
            city_aqi.append({
                'station': station,
                'aqi': aqi
            })
    
    if not city_aqi:
        return None
    
    # Create DataFrame and get top N
    cities_df = pd.DataFrame(city_aqi)
    return cities_df.nlargest(top_n, 'aqi')

def create_aqi_card(aqi, station, aqi_info, date):
    """Create a beautiful AQI display card"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {aqi_info['color']} 0%, {aqi_info['color']}dd 100%);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    ">
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);"></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 1.2rem; color: #ffffff; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 2px;">AIR QUALITY INDEX</div>
            <div style="font-size: 5rem; margin: 1rem 0;">{aqi_info['emoji']}</div>
            <div style="font-size: 5rem; color: #ffffff; font-weight: 900; margin: 1rem 0; text-shadow: 0 4px 20px rgba(0,0,0,0.3);">{int(aqi)}</div>
            <div style="font-size: 2rem; color: #ffffff; font-weight: 700; margin: 1rem 0; letter-spacing: 1px;">{aqi_info['category'].upper()}</div>
            <div style="font-size: 1.1rem; color: #ffffff; opacity: 0.95; margin-top: 1rem;">{station}</div>
            <div style="font-size: 1rem; color: #ffffff; opacity: 0.85; margin-top: 0.5rem;">{date}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_pollutant_bars(pollutants):
    """Create modern pollutant concentration bars"""
    st.markdown("### 🔬 Pollutant Concentrations")
    
    pollutant_info = {
        'PM2.5': {'name': 'PM 2.5', 'max': 250, 'unit': 'µg/m³', 'icon': '💨'},
        'PM10': {'name': 'PM 10', 'max': 430, 'unit': 'µg/m³', 'icon': '🌫️'},
        'CO': {'name': 'CO', 'max': 5, 'unit': 'mg/m³', 'icon': '⚠️'},
        'NO2': {'name': 'NO₂', 'max': 100, 'unit': 'µg/m³', 'icon': '🏭'},
        'O3': {'name': 'O₃', 'max': 200, 'unit': 'µg/m³', 'icon': '☀️'},
        'SO2': {'name': 'SO₂', 'max': 100, 'unit': 'µg/m³', 'icon': '🔥'},
        'NH3': {'name': 'NH₃', 'max': 400, 'unit': 'µg/m³', 'icon': '💧'},
        'NO': {'name': 'NO', 'max': 100, 'unit': 'µg/m³', 'icon': '🚗'}
    }
    
    for key, info in pollutant_info.items():
        value = pollutants.get(key)
        if value is not None and not pd.isna(value):
            # Calculate percentage for bar width
            percentage = min((value / info['max']) * 100, 100)
            
            # Color based on percentage
            if percentage < 30:
                color = '#4caf50'
            elif percentage < 60:
                color = '#ffc107'
            else:
                color = '#f44336'
            
            st.markdown(f"""
            <div class="pollutant-bar">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 24px;">{info['icon']}</span>
                        <div style="color: #ffffff; font-weight: 600; font-size: 16px; min-width: 80px;">{info['name']}</div>
                    </div>
                    <div style="flex: 1; margin: 0 20px;">
                        <div style="background: #1e3a52; height: 12px; border-radius: 6px; overflow: hidden;">
                            <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 6px; transition: width 0.5s ease;"></div>
                        </div>
                    </div>
                    <div style="color: #ffffff; font-weight: 700; min-width: 120px; text-align: right; font-size: 16px;">{value:.2f} {info['unit']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def create_hourly_chart(hourly_data, station, date):
    """Create hourly AQI bar chart"""
    if hourly_data is None or len(hourly_data) == 0:
        st.warning("No hourly data available for this date")
        return
    
    # Get colors based on AQI categories
    colors = [get_aqi_category(aqi)['color'] for aqi in hourly_data['aqi']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_data['hour'],
        y=hourly_data['aqi'],
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.9
        ),
        hovertemplate='<b>Hour: %{x}:00</b><br>AQI: %{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Hourly AQI Pattern - {date.strftime('%B %d, %Y')}",
            font=dict(color='#ffffff', size=18)
        ),
        xaxis_title="Hour of Day",
        yaxis_title="AQI",
        template="plotly_dark",
        height=400,
        plot_bgcolor='#0a1929',
        paper_bgcolor='#0a1929',
        font=dict(color='#ffffff'),
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2,
            gridcolor='#1e3a52',
            title=dict(font=dict(color='#b0bec5')),
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            gridcolor='#1e3a52',
            title=dict(font=dict(color='#b0bec5')),
            tickfont=dict(color='#ffffff')
        ),
        margin=dict(t=60, b=40, l=50, r=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_historical_chart(historical_data, station):
    """Create historical trend chart"""
    if historical_data is None or len(historical_data) == 0:
        st.warning("No historical data available")
        return
    
    # Get colors based on AQI categories
    colors = [get_aqi_category(aqi)['color'] for aqi in historical_data['aqi']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['aqi'],
        mode='lines+markers',
        line=dict(color='#5c6bc0', width=3),
        marker=dict(
            color=colors,
            size=8,
            line=dict(color='#ffffff', width=2)
        ),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>AQI: %{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Last 30 Days AQI Trend - {station}",
            font=dict(color='#ffffff', size=18)
        ),
        xaxis_title="Date",
        yaxis_title="AQI",
        template="plotly_dark",
        height=400,
        plot_bgcolor='#0a1929',
        paper_bgcolor='#0a1929',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='#1e3a52',
            title=dict(font=dict(color='#b0bec5')),
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            gridcolor='#1e3a52',
            title=dict(font=dict(color='#b0bec5')),
            tickfont=dict(color='#ffffff')
        ),
        margin=dict(t=60, b=40, l=50, r=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_india_map(df, date):
    """Create interactive India map with color-coded AQI markers"""
    # Collect map data for all stations
    map_data = []
    for station in df['Station'].unique():
        aqi, _ = calculate_daily_aqi(df, station, date)
        if aqi is not None:
            station_info = df[df['Station'] == station].iloc[0]
            if not pd.isna(station_info['latitude']) and not pd.isna(station_info['longitude']):
                map_data.append({
                    'station': station,
                    'lat': station_info['latitude'],
                    'lon': station_info['longitude'],
                    'aqi': aqi,
                    'category': get_aqi_category(aqi)['category']
                })
    
    if not map_data:
        st.warning("No map data available for this date")
        return
    
    map_df = pd.DataFrame(map_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=map_df['lat'],
        lon=map_df['lon'],
        mode='markers',
        marker=dict(
            size=12,
            color=map_df['aqi'],
            colorscale=[
                [0, '#4caf50'],      # Good
                [0.125, '#8bc34a'],  # Satisfactory
                [0.25, '#ffc107'],   # Moderate
                [0.5, '#ff9800'],    # Poor
                [0.75, '#f44336'],   # Very Poor
                [1, '#d32f2f']       # Severe
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="AQI", font=dict(color='#ffffff', size=14)),
                tickfont=dict(color='#ffffff'),
                bgcolor='#132f4c',
                bordercolor='#5c6bc0',
                borderwidth=1
            ),
            opacity=0.9
        ),
        text=map_df['station'],
        customdata=map_df[['aqi', 'category']],
        hovertemplate='<b>%{text}</b><br>AQI: %{customdata[0]:.0f}<br>Status: %{customdata[1]}<extra></extra>'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=4,
            center=dict(lat=20.5937, lon=78.9629)
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='#0a1929',
        font=dict(color='#ffffff'),
        title=dict(
            text="🗺️ All India AQI Map",
            font=dict(color='#ffffff', size=18)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_top_cities(top_cities):
    """Display top 10 cities by AQI"""
    st.markdown("### 🏙️ Top 10 Cities by AQI")
    
    if top_cities is None or len(top_cities) == 0:
        st.warning("No data available")
        return
    
    for idx, row in top_cities.iterrows():
        aqi_info = get_aqi_category(row['aqi'])
        
        st.markdown(f"""
        <div class="city-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="
                        width: 40px;
                        height: 40px;
                        background: linear-gradient(135deg, #5c6bc0 0%, #7986cb 100%);
                        border-radius: 10px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 800;
                        font-size: 16px;
                        color: #ffffff;
                        box-shadow: 0 4px 12px rgba(92, 107, 192, 0.3);
                    ">{idx + 1}</div>
                    <div style="color: #ffffff; font-weight: 600; font-size: 16px;">{row['station']}</div>
                </div>
                <div style="
                    background: {aqi_info['color']};
                    color: #ffffff;
                    padding: 8px 16px;
                    border-radius: 8px;
                    font-weight: 800;
                    font-size: 16px;
                    box-shadow: 0 4px 12px {aqi_info['color']}40;
                ">{int(row['aqi'])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== ML & ARIMA FUNCTIONS ====================

def prepare_ml_data(df, station, test_size=0.2):
    """Prepare data for ML training"""
    station_data = df[df['Station'] == station].copy()
    
    if len(station_data) < 100:
        return None, None, None, None, None, None
    
    # Define features
    base_features = ['PM2.5', 'PM10', 'CO', 'NO', 'NO2', 'NH3', 'O3', 'SO2']
    lag_features = [f'{col}_lag1' for col in base_features if f'{col}_lag1' in station_data.columns]
    time_features = ['Month', 'Day', 'DayOfWeek', 'DayOfYear']
    
    all_features = base_features + lag_features + time_features
    all_features = [f for f in all_features if f in station_data.columns]
    
    # Remove rows with missing values
    station_data = station_data.dropna(subset=all_features)
    
    if len(station_data) < 50:
        return None, None, None, None, None, None
    
    X = station_data[all_features]
    y = station_data['AQI']
    
    # Train-test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, all_features

def train_all_ml_models(X_train, y_train):
    """Train multiple ML models"""
    models = {}
    
    # Linear Models
    models['Linear Regression'] = LinearRegression()
    models['Ridge Regression'] = Ridge(alpha=1.0)
    models['Lasso Regression'] = Lasso(alpha=0.01)
    
    # Tree-based Models
    models['Decision Tree'] = DecisionTreeRegressor(max_depth=10, random_state=42)
    models['Random Forest'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    # XGBoost
    try:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    except:
        pass
    
    # Other Models
    models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    
    # Train all models
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return metrics"""
    metrics = {}
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            
            metrics[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred),
                'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100
            }
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    return metrics

def calculate_feature_importance(models, feature_names):
    """Calculate feature importance for models that support it"""
    importance_data = {}
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_data[name] = {
                'features': feature_names,
                'importances': importances,
                'sorted_idx': np.argsort(importances)[::-1]
            }
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            importance_data[name] = {
                'features': feature_names,
                'importances': importances,
                'sorted_idx': np.argsort(importances)[::-1]
            }
    
    return importance_data

def display_model_comparison(metrics):
    """Display model comparison with metrics"""
    st.markdown("### 📊 Model Performance Comparison")
    
    if not metrics:
        st.warning("No model metrics available")
        return
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics).T.round(3)
    metrics_df = metrics_df.sort_values('R2', ascending=False)
    
    # Create metrics display
    cols = st.columns(min(4, len(metrics_df)))
    
    for idx, (model_name, row) in enumerate(metrics_df.iterrows()):
        col_idx = idx % len(cols)
        with cols[col_idx]:
            r2 = row['R2']
            mae = row['MAE']
            
            # Determine accuracy color
            if r2 > 0.8:
                accuracy_class = "accuracy-high"
            elif r2 > 0.6:
                accuracy_class = "accuracy-medium"
            else:
                accuracy_class = "accuracy-low"
            
            st.markdown(f"""
            <div class="model-metrics {accuracy_class}">
                <div style="color: #ffffff; font-size: 18px; font-weight: 700; margin-bottom: 12px;">
                    {model_name}
                </div>
                <div style="color: #ffffff; font-size: 24px; font-weight: 800;">
                    {r2:.3f}
                </div>
                <div style="color: #b0bec5; font-size: 12px; font-weight: 600;">
                    R² Score
                </div>
                <div style="margin-top: 12px;">
                    <div style="color: #ffffff; font-size: 14px;">MAE: {mae:.1f}</div>
                    <div style="color: #b0bec5; font-size: 12px;">Mean Absolute Error</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed metrics table
    st.markdown("#### 📋 Detailed Metrics")
    st.dataframe(metrics_df.style.format({
        'MAE': '{:.2f}',
        'RMSE': '{:.2f}',
        'R2': '{:.3f}',
        'MAPE': '{:.1f}%'
    }).background_gradient(subset=['R2'], cmap='RdYlGn'), 
    use_container_width=True)

def display_feature_importance_chart(feature_importance_data, model_name):
    """Display feature importance visualization"""
    if model_name not in feature_importance_data:
        return None
    
    data = feature_importance_data[model_name]
    
    # Get top 10 features
    top_n = min(10, len(data['features']))
    top_idx = data['sorted_idx'][:top_n]
    top_features = [data['features'][i] for i in top_idx]
    top_importances = [data['importances'][i] for i in top_idx]
    
    # Normalize importances
    max_imp = max(top_importances) if top_importances else 1
    normalized_importances = [imp/max_imp*100 for imp in top_importances]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=top_features,
            x=normalized_importances,
            orientation='h',
            marker=dict(
                color='#5c6bc0',
                line=dict(width=0),
                opacity=0.9
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Feature Importances ({model_name})",
            font=dict(color='#ffffff', size=16)
        ),
        xaxis_title="Normalized Importance (%)",
        template="plotly_dark",
        height=400,
        plot_bgcolor='#0a1929',
        paper_bgcolor='#0a1929',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='#1e3a52',
            title=dict(font=dict(color='#b0bec5')),
            tickfont=dict(color='#ffffff'),
            range=[0, 105]
        ),
        yaxis=dict(
            gridcolor='#1e3a52',
            tickfont=dict(color='#ffffff'),
            autorange='reversed'
        ),
        margin=dict(t=40, b=40, l=150, r=30)
    )
    
    return fig

def create_shap_explanations(model, X_test, feature_names, model_name="Random Forest"):
    """Create SHAP explanations for model predictions"""
    if not SHAP_AVAILABLE:
        return None, None
    
    try:
        if "Random Forest" in model_name or "XGBoost" in model_name or "Gradient Boosting" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            # Use KernelExplainer for non-tree models
            explainer = shap.KernelExplainer(model.predict, X_test[:100])
            shap_values = explainer.shap_values(X_test[:10])
        
        # Create summary plot
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        if len(X_test) > 0:
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        ax1.set_facecolor('#0a1929')
        plt.tight_layout()
        
        return fig1, explainer
        
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None, None

# ==================== ARIMA FUNCTIONS ====================

def train_arima_model(series, order=(1, 1, 1)):
    """Train ARIMA model with specific order"""
    try:
        model = ARIMA(series.dropna(), order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        return None

def arima_forecast(model_fit, steps=7):
    """Generate forecast using ARIMA model"""
    try:
        forecast_result = model_fit.forecast(steps=steps)
        forecast_df = model_fit.get_forecast(steps=steps)
        conf_int = forecast_df.conf_int()
        
        # Return as numpy array for easier indexing
        return forecast_result.values, conf_int
    except Exception as e:
        print(f"Error generating ARIMA forecast: {e}")
        return None, None

def create_forecast_chart(forecast_data, station):
    """Create forecast chart with confidence intervals"""
    if forecast_data is None or len(forecast_data) == 0:
        st.warning("Unable to generate forecast - insufficient historical data")
        return
    
    # Split historical and forecast data
    historical = forecast_data[~forecast_data['is_forecast']]
    forecast = forecast_data[forecast_data['is_forecast']]
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical['date'],
        y=historical['aqi'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#4caf50', width=3),
        marker=dict(
            size=10,
            color='#4caf50',
            line=dict(color='#ffffff', width=2)
        ),
        hovertemplate='<b>%{x|%b %d}</b><br>AQI: %{y:.0f}<extra></extra>'
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['aqi'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff9800', width=3, dash='dash'),
        marker=dict(
            size=12,
            symbol='diamond',
            color='#ff9800',
            line=dict(color='#ffffff', width=2)
        ),
        hovertemplate='<b>%{x|%b %d}</b><br>Predicted AQI: %{y:.0f}<extra></extra>'
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
        y=forecast['upper_bound'].tolist() + forecast['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 152, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence',
        hoverinfo='skip',
        showlegend=True
    ))
    
    fig.update_layout(
        title=dict(
            text=f"🔮 7-Day AQI Forecast - {station}",
            font=dict(color='#ffffff', size=20)
        ),
        xaxis_title="Date",
        yaxis_title="AQI",
        template="plotly_dark",
        height=500,
        plot_bgcolor='#0a1929',
        paper_bgcolor='#0a1929',
        font=dict(color='#ffffff'),
        xaxis=dict(
            gridcolor='#1e3a52',
            title=dict(font=dict(color='#b0bec5')),
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            gridcolor='#1e3a52',
            title=dict(font=dict(color='#b0bec5')),
            tickfont=dict(color='#ffffff')
        ),
        legend=dict(
            font=dict(color='#ffffff'),
            bgcolor='#132f4c',
            bordercolor='#5c6bc0',
            borderwidth=1,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x unified',
        margin=dict(t=80, b=40, l=50, r=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN APPLICATION ====================

def main():
    """Main application function"""
    
    # Header Section
    st.markdown("# 🌍 INDIA'S AQI DASHBOARD")
    st.markdown("### 🤖 ML-Powered Air Quality Monitoring & Forecasting")
    st.markdown("---")
    
    # Load Data
    with st.spinner('🔄 Loading air quality data...'):
        df = load_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check if '2022_2025_data.csv' exists.")
        return
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("## 🎯 Controls")
        st.markdown("")
        
        # Add Year and Month columns if not exist
        if 'Year' not in df.columns:
            df['Year'] = df['Date'].dt.year
        if 'Month' not in df.columns:
            df['Month'] = df['Date'].dt.month
        
        # Year Selection
        available_years = sorted(df['Year'].unique())
        selected_year = st.selectbox(
            "📅 Select Year",
            available_years,
            index=len(available_years) - 1
        )
        
        # Month Selection
        available_months = sorted(df[df['Year'] == selected_year]['Month'].unique())
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        selected_month = st.selectbox(
            "📆 Select Month",
            available_months,
            format_func=lambda x: month_names[x-1],
            index=len(available_months) - 1
        )
        
        # Station Selection
        stations = sorted(df['Station'].unique())
        selected_station = st.selectbox(
            "📍 Select Station",
            stations,
            index=0
        )
        
        # Date Selection
        available_dates = sorted(df[(df['Station'] == selected_station) & 
                                   (df['Year'] == selected_year) & 
                                   (df['Month'] == selected_month)]['Date'].unique())
        
        if len(available_dates) > 0:
            min_date = available_dates[0].date()
            max_date = available_dates[-1].date()
            default_date = max_date
            
            selected_date = st.date_input(
                "🗓️ Select Date",
                value=default_date,
                min_value=min_date,
                max_value=max_date
            )
        else:
            st.error(f"No data available for {selected_station} in {month_names[selected_month-1]} {selected_year}")
            return
        
        st.markdown("---")
        
        # ML Settings
        st.markdown("### 🤖 ML Settings")
        use_ml_forecast = st.checkbox("Use ML Forecasting", value=True)
        
        st.markdown("---")
        
        # About Section
        st.markdown("### ℹ️ About Dashboard")
        
        min_date_overall = df['Date'].min()
        max_date_overall = df['Date'].max()
        
        st.info(f"""
        **📊 Data Coverage**
        
        Monitoring **{len(df['Station'].unique())}** stations across India
        
        **Period:** {min_date_overall.strftime('%b %d, %Y')} to {max_date_overall.strftime('%b %d, %Y')}
        
        **🤖 ML Models:**
        - ARIMA Time Series
        - Random Forest
        - XGBoost
        - Gradient Boosting
        
        **🎨 AQI Scale:**
        - 🟢 0-50: Good
        - 🟡 51-100: Satisfactory
        - 🟠 101-200: Moderate
        - 🔴 201-300: Poor
        - 🟣 301-400: Very Poor
        - 🟤 401+: Severe
        """)
        
        st.markdown("---")
        
        # Quick Statistics
        st.markdown("### 📊 Quick Stats")
        total_stations = len(df['Station'].unique())
        total_records = len(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stations", f"{total_stations}")
        with col2:
            st.metric("Records", f"{total_records:,}")
    
    # ==================== MAIN CONTENT ====================
    
    # Calculate AQI for selected date
    selected_date_dt = pd.to_datetime(selected_date)
    aqi, pollutants = calculate_daily_aqi(df, selected_station, selected_date_dt)
    
    if aqi is None:
        st.warning(f"⚠️ No data available for {selected_station} on {selected_date.strftime('%B %d, %Y')}")
        return
    
    aqi_info = get_aqi_category(aqi)
    
    # Create Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Daily View",
        "📊 Monthly Analytics",
        "📈 Yearly Comparison",
        "🗺️ Station Comparison",
        "🔮 AI Forecast",
        "🤖 ML Models & XAI"
    ])
    
    # ==================== TAB 1: DAILY VIEW ====================
    with tab1:
        st.markdown(f"## {selected_station}")
        st.markdown(f"#### {selected_date.strftime('%A, %B %d, %Y')}")
        st.markdown("")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # AQI Display Card
            create_aqi_card(aqi, selected_station, aqi_info, selected_date.strftime('%B %d, %Y'))
            
            # Health Impact Section
            st.markdown("### 💡 Health Impact")
            st.markdown(f"""
            <div class="custom-card">
                <p style="color: #ffffff; line-height: 1.6; margin-bottom: 12px;">{aqi_info['description']}</p>
                <p style="color: #b0bec5; line-height: 1.6;"><strong>💊 Recommendation:</strong> {aqi_info['health_impact']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Pollutant Concentrations
            st.markdown("")
            create_pollutant_bars(pollutants)
            
            # Top 10 Cities
            st.markdown("---")
            top_cities = get_top_cities(df, selected_date_dt)
            display_top_cities(top_cities)
        
        with col2:
            # Hourly Pattern Chart
            st.markdown("### 📊 Hourly AQI Pattern")
            hourly_data = get_hourly_data(df, selected_station, selected_date_dt)
            create_hourly_chart(hourly_data, selected_station, selected_date_dt)
            
            st.markdown("")
            
            # Historical Trend Chart
            st.markdown("### 📈 Historical Trend (Last 30 Days)")
            historical_data = get_historical_data(df, selected_station, days=30)
            create_historical_chart(historical_data, selected_station)
            
            st.markdown("")
            
            # India Map
            st.markdown("### 🗺️ All India AQI Map")
            create_india_map(df, selected_date_dt)
    
    # ==================== TAB 2: MONTHLY ANALYTICS ====================
    with tab2:
        month_names_full = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
        st.markdown(f"## Monthly Analytics")
        st.markdown(f"### {month_names_full[selected_month-1]} {selected_year}")
        st.markdown("")
        
        # Calculate monthly AQI
        monthly_data = df[(df['Year'] == selected_year) & (df['Month'] == selected_month) & (df['Station'] == selected_station)]
        
        if len(monthly_data) > 0:
            # Calculate average pollutants
            monthly_pollutants = {
                'PM2.5': monthly_data['PM2.5'].mean(),
                'PM10': monthly_data['PM10'].mean(),
                'CO': monthly_data['CO'].mean(),
                'NO2': monthly_data['NO2'].mean(),
                'O3': monthly_data['O3'].mean(),
                'SO2': monthly_data['SO2'].mean(),
                'NH3': monthly_data['NH3'].mean(),
                'NO': monthly_data['NO'].mean()
            }
            
            monthly_aqi = calculate_aqi(monthly_pollutants)
            monthly_aqi_info = get_aqi_category(monthly_aqi)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Average AQI", f"{monthly_aqi:.1f}")
                
                st.markdown(f"""
                <div class="custom-card">
                    <p style="color: #ffffff; font-size: 18px; font-weight: 600; margin-bottom: 8px;">Category: {monthly_aqi_info['category']}</p>
                    <p style="color: #b0bec5; line-height: 1.6;">{monthly_aqi_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Monthly Stats
                st.markdown("### 📊 Monthly Statistics")
                st.write(f"**Days with data:** {monthly_data['Date'].nunique()}")
                st.write(f"**Total records:** {len(monthly_data)}")
                st.write(f"**Average AQI:** {monthly_aqi:.1f}")
            
            with col2:
                # Pollutant Distribution Chart
                st.markdown("### 📊 Monthly Pollutant Distribution")
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(monthly_pollutants.keys()),
                        y=list(monthly_pollutants.values()),
                        marker=dict(
                            color='#5c6bc0',
                            line=dict(width=0),
                            opacity=0.9
                        ),
                        hovertemplate='<b>%{x}</b><br>Concentration: %{y:.2f}<extra></extra>'
                    )
                ])
                fig.update_layout(
                    template='plotly_dark',
                    title=dict(text="Pollutant Concentrations", font=dict(color='#ffffff')),
                    xaxis_title="Pollutant",
                    yaxis_title="Concentration",
                    height=450,
                    showlegend=False,
                    plot_bgcolor='#0a1929',
                    paper_bgcolor='#0a1929',
                    font=dict(color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monthly data available")
    
    # ==================== TAB 3: YEARLY COMPARISON ====================
    with tab3:
        st.markdown(f"## Yearly Comparison")
        st.markdown(f"### {selected_station}")
        st.markdown("")
        
        years = sorted(df['Year'].unique())
        yearly_aqi = []
        
        for year in years:
            year_data = df[(df['Year'] == year) & (df['Station'] == selected_station)]
            if len(year_data) > 0:
                year_pollutants = {
                    'PM2.5': year_data['PM2.5'].mean(),
                    'PM10': year_data['PM10'].mean()
                }
                year_aqi = calculate_aqi(year_pollutants)
                yearly_aqi.append({'year': year, 'aqi': year_aqi})
        
        if yearly_aqi:
            yearly_df = pd.DataFrame(yearly_aqi)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📊 Year-over-Year AQI")
                fig = go.Figure(data=[
                    go.Bar(
                        x=yearly_df['year'].astype(str),
                        y=yearly_df['aqi'],
                        marker=dict(
                            color=yearly_df['aqi'],
                            colorscale='Reds',
                            line=dict(width=0),
                            opacity=0.9
                        ),
                        hovertemplate='<b>Year %{x}</b><br>Average AQI: %{y:.1f}<extra></extra>'
                    )
                ])
                fig.update_layout(
                    template='plotly_dark',
                    xaxis_title="Year",
                    yaxis_title="Average AQI",
                    height=450,
                    showlegend=False,
                    plot_bgcolor='#0a1929',
                    paper_bgcolor='#0a1929',
                    font=dict(color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📋 Yearly Averages")
                for _, row in yearly_df.iterrows():
                    year = int(row['year'])
                    aqi_val = row['aqi']
                    category = get_aqi_category(aqi_val)
                    
                    st.markdown(f"""
                    <div class="custom-card" style="margin: 12px 0;">
                        <p style="color: #ffffff; font-size: 18px; font-weight: 700; margin-bottom: 4px;">{year}</p>
                        <p style="color: {category['color']}; font-size: 24px; font-weight: 800; margin: 8px 0;">{aqi_val:.1f}</p>
                        <p style="color: #b0bec5; font-size: 14px;">{category['category']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Insufficient data for yearly comparison")
    
    # ==================== TAB 4: STATION COMPARISON ====================
    with tab4:
        st.markdown(f"## Air Quality Comparison")
        st.markdown(f"### {selected_date.strftime('%B %d, %Y')}")
        st.markdown("")
        
        # Get top cities for the selected date
        top_cities = get_top_cities(df, selected_date_dt, top_n=20)
        
        if top_cities is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🔴 Top 10 Most Polluted")
                top_polluted = top_cities.head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=top_polluted['station'],
                        x=top_polluted['aqi'],
                        orientation='h',
                        marker=dict(
                            color=top_polluted['aqi'],
                            colorscale='Reds',
                            line=dict(width=0),
                            opacity=0.9
                        ),
                        hovertemplate='<b>%{y}</b><br>AQI: %{x:.0f}<extra></extra>'
                    )
                ])
                fig.update_layout(
                    template='plotly_dark',
                    xaxis_title="AQI",
                    height=500,
                    showlegend=False,
                    plot_bgcolor='#0a1929',
                    paper_bgcolor='#0a1929',
                    font=dict(color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🟢 Top 10 Cleanest")
                top_clean = top_cities.sort_values('aqi', ascending=True).head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=top_clean['station'],
                        x=top_clean['aqi'],
                        orientation='h',
                        marker=dict(
                            color=top_clean['aqi'],
                            colorscale='Greens',
                            line=dict(width=0),
                            opacity=0.9
                        ),
                        hovertemplate='<b>%{y}</b><br>AQI: %{x:.0f}<extra></extra>'
                    )
                ])
                fig.update_layout(
                    template='plotly_dark',
                    xaxis_title="AQI",
                    height=500,
                    showlegend=False,
                    plot_bgcolor='#0a1929',
                    paper_bgcolor='#0a1929',
                    font=dict(color='#ffffff')
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for station comparison")
    
    # ==================== TAB 5: AI FORECAST ====================
    with tab5:
        st.markdown(f"## 🔮 AI-Powered AQI Forecast")
        st.markdown(f"### {selected_station}")
        st.markdown("")
        
        # Get historical data for forecasting
        daily_aqi_series = df[df['Station'] == selected_station].groupby('Date')['AQI'].mean().sort_index()
        
        if len(daily_aqi_series) >= 30:
            # Train ARIMA model
            with st.spinner("Training ARIMA model..."):
                arima_model = train_arima_model(daily_aqi_series, order=(1, 1, 1))
            
            if arima_model:
                # Generate forecast
                forecast, conf_int = arima_forecast(arima_model, steps=7)
                
                if forecast is not None:
                    # Prepare forecast data
                    last_date = daily_aqi_series.index[-1]
                    
                    # Convert forecast to list/numpy array for easier indexing
                    forecast_values = list(forecast) if not isinstance(forecast, (list, np.ndarray)) else forecast
                    
                    forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast_values))]
                    
                    forecast_data = []
                    for i in range(len(forecast_values)):
                        forecast_data.append({
                            'date': forecast_dates[i],
                            'aqi': forecast_values[i],
                            'lower_bound': conf_int.iloc[i, 0] if conf_int is not None else forecast_values[i] * 0.9,
                            'upper_bound': conf_int.iloc[i, 1] if conf_int is not None else forecast_values[i] * 1.1,
                            'is_forecast': True
                        })
                    
                    # Prepare historical data
                    historical_data = []
                    for i in range(min(14, len(daily_aqi_series))):
                        historical_data.append({
                            'date': daily_aqi_series.index[-(i+1)],
                            'aqi': daily_aqi_series.iloc[-(i+1)],
                            'lower_bound': daily_aqi_series.iloc[-(i+1)],
                            'upper_bound': daily_aqi_series.iloc[-(i+1)],
                            'is_forecast': False
                        })
                    
                    # Combine data
                    combined_data = pd.DataFrame(historical_data[::-1] + forecast_data)
                    
                    # Display forecast chart
                    create_forecast_chart(combined_data, selected_station)
                    
                    # Display forecast details
                    st.markdown("### 📋 7-Day Forecast Details")
                    forecast_df = pd.DataFrame(forecast_data)
                    forecast_df['Category'] = forecast_df['aqi'].apply(lambda x: get_aqi_category(x)['category'])
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        for idx, row in forecast_df.iterrows():
                            aqi_cat = get_aqi_category(row['aqi'])
                            date_str = row['date'].strftime('%A, %B %d')
                            
                            st.markdown(f"""
                            <div class="custom-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <div style="color: #ffffff; font-size: 18px; font-weight: 700; margin-bottom: 8px;">
                                            {date_str}
                                        </div>
                                        <div style="color: #b0bec5; font-size: 14px; font-weight: 500;">
                                            📊 Range: {int(row['lower_bound'])} - {int(row['upper_bound'])} AQI
                                        </div>
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="color: {aqi_cat['color']}; font-size: 40px; font-weight: 900;">
                                            {int(row['aqi'])}
                                        </div>
                                        <div style="color: #ffffff; font-size: 14px; font-weight: 700;">
                                            {aqi_cat['category']}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        # Forecast summary
                        avg_forecast = forecast_df['aqi'].mean()
                        max_forecast = forecast_df['aqi'].max()
                        min_forecast = forecast_df['aqi'].min()
                        
                        st.metric("📈 Average AQI", f"{int(avg_forecast)}")
                        st.metric("🔴 Peak AQI", f"{int(max_forecast)}")
                        st.metric("🟢 Best AQI", f"{int(min_forecast)}")
                else:
                    st.warning("Failed to generate forecast")
            else:
                st.warning("Failed to train ARIMA model")
        else:
            st.warning(f"Need at least 30 days of historical data for forecasting. Currently have {len(daily_aqi_series)} days.")
    
    # ==================== TAB 6: ML MODELS & XAI ====================
    with tab6:
        st.markdown(f"## 🤖 Machine Learning Analysis")
        st.markdown(f"### {selected_station}")
        st.markdown("")
        
        # Prepare ML data
        ml_result = prepare_ml_data(df, selected_station)
        
        if ml_result[0] is not None:
            X_train, X_test, y_train, y_test, scaler, feature_names = ml_result
            
            # Train models
            with st.spinner("Training ML models..."):
                models = train_all_ml_models(X_train, y_train)
            
            if models:
                # Evaluate models
                metrics = evaluate_models(models, X_test, y_test)
                
                # Display model comparison
                display_model_comparison(metrics)
                
                # Feature Importance
                st.markdown("---")
                st.markdown("### 🔍 Feature Importance Analysis")
                
                feature_importance_data = calculate_feature_importance(models, feature_names)
                
                if feature_importance_data:
                    # Let user select model for feature importance
                    model_options = list(feature_importance_data.keys())
                    selected_model = st.selectbox("Select model for feature importance:", model_options)
                    
                    if selected_model:
                        fig = display_feature_importance_chart(feature_importance_data, selected_model)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Explain feature importance
                        st.markdown("""
                        <div class="custom-card">
                            <h4>📝 Feature Importance Interpretation:</h4>
                            <p style="color: #b0bec5; line-height: 1.6;">
                            Feature importance shows which factors most influence AQI predictions. 
                            Higher importance means the feature has more impact on the model's decisions.
                            </p>
                            <p style="color: #b0bec5; line-height: 1.6; margin-top: 8px;">
                            <strong>Key Insights:</strong><br>
                            • PM2.5 and PM10 are typically the most important factors<br>
                            • Temporal features capture seasonal patterns<br>
                            • Higher importance = greater impact on AQI prediction
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # SHAP Explanations
                st.markdown("---")
                st.markdown("### 🎯 Explainable AI (SHAP)")
                
                if SHAP_AVAILABLE:
                    # Select model for SHAP
                    model_options = list(models.keys())
                    shap_model = st.selectbox("Select model for SHAP explanation:", model_options)
                    
                    if shap_model and shap_model in models:
                        with st.spinner("Generating SHAP explanations..."):
                            shap_fig, explainer = create_shap_explanations(
                                models[shap_model], 
                                X_test[:50], 
                                feature_names,
                                shap_model
                            )
                            
                            if shap_fig:
                                st.pyplot(shap_fig)
                                
                                st.markdown("""
                                <div class="custom-card">
                                    <h4>🔍 How to Read SHAP Plots:</h4>
                                    <ul style="color: #e0f2f1;">
                                        <li><strong>Red bars:</strong> Features increasing AQI prediction</li>
                                        <li><strong>Blue bars:</strong> Features decreasing AQI prediction</li>
                                        <li><strong>Bar length:</strong> Magnitude of impact on prediction</li>
                                        <li><strong>Base value:</strong> Average AQI prediction</li>
                                        <li><strong>Feature values:</strong> Actual values for each feature</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("SHAP explanations not available for this model type")
                else:
                    st.warning("SHAP not installed. Install with: pip install shap")
                
                # Model Insights
                st.markdown("---")
                st.markdown("### 💡 Model Insights & Recommendations")
                
                if metrics:
                    # Best performing model
                    best_model = max(metrics.items(), key=lambda x: x[1]['R2'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="custom-card">
                            <h4 style="color: #ffffff; margin-bottom: 12px;">🏆 Best Performing Model</h4>
                            <p style="color: #b0bec5; font-size: 24px; font-weight: 800; margin: 8px 0;">
                            {best_model[0]}
                            </p>
                            <p style="color: #ffffff; font-size: 16px;">
                            R² Score: <span style="color: #4caf50;">{best_model[1]['R2']:.3f}</span>
                            </p>
                            <p style="color: #ffffff; font-size: 16px;">
                            MAE: <span style="color: #ffc107;">{best_model[1]['MAE']:.1f}</span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Prediction confidence
                        avg_r2 = np.mean([m['R2'] for m in metrics.values()])
                        if avg_r2 > 0.8:
                            confidence = "High"
                            confidence_color = "#4caf50"
                        elif avg_r2 > 0.6:
                            confidence = "Medium"
                            confidence_color = "#ffc107"
                        else:
                            confidence = "Low"
                            confidence_color = "#f44336"
                        
                        st.markdown(f"""
                        <div class="custom-card">
                            <h4 style="color: #ffffff; margin-bottom: 12px;">🎯 Prediction Confidence</h4>
                            <p style="color: {confidence_color}; font-size: 24px; font-weight: 800; margin: 8px 0;">
                            {confidence}
                            </p>
                            <p style="color: #b0bec5; font-size: 14px;">
                            Average R² across all models: {avg_r2:.3f}
                            </p>
                            <p style="color: #b0bec5; font-size: 14px;">
                            Based on {len(X_train)} training samples
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ARIMA vs ML Comparison
                    st.markdown("---")
                    st.markdown("### 📊 Why ARIMA is Better for Time Series")
                    
                    st.markdown("""
                    <div class="comparison-card">
                        <h4>🎯 ARIMA Advantages for AQI Forecasting</h4>
                        
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin: 1.5rem 0;">
                            <div>
                                <h5>✅ ARIMA Strengths</h5>
                                <ul style="color: #bbdefb;">
                                    <li>Captures temporal dependencies</li>
                                    <li>Handles trends & seasonality</li>
                                    <li>Provides confidence intervals</li>
                                    <li>Specifically designed for time series</li>
                                </ul>
                            </div>
                            
                            <div>
                                <h5>⚠️ Traditional ML Limitations</h5>
                                <ul style="color: #ffcdd2;">
                                    <li>No temporal awareness</li>
                                    <li>Poor trend extrapolation</li>
                                    <li>No uncertainty quantification</li>
                                    <li>Requires manual feature engineering</li>
                                </ul>
                            </div>
                        </div>
                        
                        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 8px;">
                            <p style="color: #e3f2fd;">
                            <strong>📈 Key Insight:</strong> AQI exhibits strong autocorrelation, seasonality, and trends. 
                            ARIMA is specifically designed for such patterns, making it superior for time series forecasting.
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Failed to train ML models")
        else:
            st.warning(f"Insufficient data for ML analysis. Need at least 100 complete records for {selected_station}")
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #b0bec5;">
        <p style="font-size: 14px; margin-bottom: 8px;">📊 Data Source: Central Pollution Control Board (CPCB) | AQI calculated using Indian standards</p>
        <p style="font-size: 13px; font-weight: 600;">
        🌍 Built with Streamlit • 📈 Powered by ARIMA & ML Models • 🧠 Explainable AI Insights • 
        🎯 Superior Time Series Forecasting • 💙 Made for India
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    main()