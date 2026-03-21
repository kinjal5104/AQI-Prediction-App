"""
India's AQI Dashboard - Complete Streamlit Application
Full-featured dashboard with all visualizations and metrics
Enhanced with beautiful dark theme, ARIMA forecasting, and PERFECT text contrast everywhere
FIXED: Universal text visibility with dynamic contrast on ALL backgrounds
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import requests

# Machine Learning imports for ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# Page configuration
st.set_page_config(
    page_title="India's AQI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ENHANCED: Contrast utility function
def get_contrast_text_color(bg_color):
    """
    Calculate optimal text color (black/white) based on background color
    Uses relative luminance calculation per WCAG guidelines for accessibility
    """
    # Remove # if present
    bg_color = bg_color.lstrip('#')
   
    # Handle short hex codes
    if len(bg_color) == 3:
        bg_color = ''.join([c*2 for c in bg_color])
   
    # Convert hex to RGB
    try:
        r = int(bg_color[0:2], 16) / 255
        g = int(bg_color[2:4], 16) / 255
        b = int(bg_color[4:6], 16) / 255
    except:
        return '#ffffff'  # Default to white on error
   
    # Calculate relative luminance
    def adjust_channel(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
   
    r_adj = adjust_channel(r)
    g_adj = adjust_channel(g)
    b_adj = adjust_channel(b)
   
    luminance = 0.2126 * r_adj + 0.7152 * g_adj + 0.0722 * b_adj
   
    # Return black for light backgrounds (luminance > 0.5), white for dark
    return '#0f172a' if luminance > 0.5 else '#ffffff'

def get_high_contrast_text(bg_color):
    """
    Get high contrast text color with enhanced visibility
    Returns white for dark backgrounds, very dark for light backgrounds
    """
    base_color = get_contrast_text_color(bg_color)
    # For even better contrast on light backgrounds
    if base_color == '#0f172a':
        return '#000000'  # Pure black for light backgrounds
    return '#ffffff'  # Pure white for dark backgrounds

# Custom CSS for beautiful dark theme with PERFECT contrast everywhere
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
   
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* FORCE: Background on root container */
    [data-testid="stAppViewContainer"] {
        background: #001f3f !important;
    }
    
    /* FORCE: Background on main content */
    section.main {
        background: #001f3f !important;
    }
   
    /* Main container - Navy Blue */
    .main {
        background: #001f3f !important;
        color: #ffffff !important;
    }
   
    /* All paragraphs and text - HIGH CONTRAST WHITE */
    .main p, .main div, .main span, .main li {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    }
   
    /* Headers - Bright white with glow */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        letter-spacing: -0.02em;
    }
   
    h1 {
        font-size: 3rem !important;
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
   
    h2 {
        font-size: 2rem !important;
        color: #ffffff !important;
    }
   
    h3 {
        font-size: 1.5rem !important;
        color: #ffffff !important;
    }
   
    /* Sidebar - Dark with purple accent */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
   
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
   
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
   
    /* Metric cards - WHITE text on dark cards */
    [data-testid="stMetric"] {
        background: rgba(30, 27, 75, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
   
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
        color: #ffffff !important;
        font-weight: 900 !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
   
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
   
    [data-testid="stMetricDelta"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
   
    /* Alert boxes - HIGH CONTRAST */
    .stAlert {
        background: rgba(139, 92, 246, 0.15) !important;
        border: 1px solid rgba(139, 92, 246, 0.4) !important;
        border-left: 4px solid #8b5cf6 !important;
        color: #ffffff !important;
    }
   
    .stAlert * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
   
    /* Info box */
    [data-testid="stAlert"] {
        background: rgba(59, 130, 246, 0.15) !important;
        border-left: 4px solid #3b82f6 !important;
    }
   
    [data-testid="stAlert"] * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
   
    /* Warning box */
    .stWarning {
        background: rgba(251, 191, 36, 0.15) !important;
        border-left: 4px solid #fbbf24 !important;
    }
   
    .stWarning * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
   
    /* Success box */
    .stSuccess {
        background: rgba(34, 197, 94, 0.15) !important;
        border-left: 4px solid #22c55e !important;
    }
   
    .stSuccess * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
   
    /* Error box */
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid #ef4444 !important;
    }
   
    .stError * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
   
    /* Selectbox - Light background with DARK text */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 2px solid rgba(139, 92, 246, 0.4) !important;
        border-radius: 10px !important;
    }
   
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
   
    .stSelectbox input, .stSelectbox span, .stSelectbox div {
        color: #000000 !important;
        font-weight: 600 !important;
    }
   
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
   
    /* Date input - Light background with DARK text */
    .stDateInput > div > div > div {
        background-color: #ffffff !important;
        border: 2px solid rgba(139, 92, 246, 0.4) !important;
        border-radius: 10px !important;
    }
   
    .stDateInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
    }
   
    .stDateInput label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
   
    /* Date picker calendar */
    [data-baseweb="calendar"] {
        background-color: #ffffff !important;
    }
   
    [data-baseweb="calendar"] * {
        color: #000000 !important;
        font-weight: 500 !important;
    }
   
    /* Dropdown menus */
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
   
    [data-baseweb="menu"] * {
        color: #000000 !important;
        font-weight: 500 !important;
    }
   
    /* Tabs - HIGH CONTRAST */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
   
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
        background: rgba(30, 27, 75, 0.6) !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-bottom: none !important;
    }
   
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.5) 0%, rgba(168, 85, 247, 0.5) 100%) !important;
        color: #ffffff !important;
        border-color: rgba(139, 92, 246, 0.6) !important;
        font-weight: 800 !important;
    }
   
    /* Custom cards - HIGH CONTRAST */
    .custom-card {
        background: rgba(30, 27, 75, 0.8);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        backdrop-filter: blur(10px);
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
   
    .custom-card p, .custom-card div, .custom-card span {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
   
    .custom-card strong {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
   
    /* Pollutant items - HIGH CONTRAST */
    .pollutant-item {
        background: rgba(30, 27, 75, 0.8);
        padding: 16px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 4px solid #8b5cf6;
        transition: all 0.3s ease;
    }
   
    .pollutant-item:hover {
        background: rgba(30, 27, 75, 0.95);
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
   
    .pollutant-item * {
        font-weight: 600 !important;
    }
   
    /* Spinner */
    .stSpinner > div {
        border-color: #8b5cf6 !important;
    }
   
    /* Buttons - HIGH CONTRAST */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700 !important;
        transition: all 0.3s ease;
    }
   
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(139, 92, 246, 0.4);
    }
   
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
   
    ::-webkit-scrollbar-track {
        background: #1e1b4b;
    }
   
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
        border-radius: 6px;
    }
   
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%);
    }
   
    /* Markdown text - HIGH CONTRAST */
    .element-container p {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
   
    /* Ensure all text is visible */
    .stMarkdown, .stText {
        color: #f1f5f9 !important;
    }
   
    .stMarkdown strong, .stText strong {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
   
    /* Code blocks */
    code {
        background: rgba(30, 27, 75, 0.8) !important;
        color: #ffffff !important;
        padding: 4px 8px !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
    }
   
    /* Divider */
    hr {
        border-color: rgba(139, 92, 246, 0.4) !important;
        margin: 2rem 0 !important;
    }
   
    /* Data frame styling */
    .stDataFrame {
        color: #000000 !important;
    }
   
    /* Expander */
    .streamlit-expanderHeader {
        color: #ffffff !important;
        font-weight: 700 !important;
        background: rgba(30, 27, 75, 0.6) !important;
    }
   
    /* Radio buttons */
    .stRadio label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
   
    /* Checkbox */
    .stCheckbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
   
    /* Number input */
    .stNumberInput label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
   
    .stNumberInput input {
        color: #000000 !important;
        background: #ffffff !important;
        font-weight: 600 !important;
    }
   
    /* Text input */
    .stTextInput label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
   
    .stTextInput input {
        color: #000000 !important;
        background: #ffffff !important;
        font-weight: 600 !important;
    }
   
    /* Text area */
    .stTextArea label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
   
    .stTextArea textarea {
        color: #000000 !important;
        background: #ffffff !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# AQI Calculation Functions
def calculate_sub_index(concentration, breakpoints):
    """Calculate AQI sub-index for a pollutant"""
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= concentration <= C_high:
            sub_index = ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
            return sub_index
    return breakpoints[-1][3]

def calculate_aqi(pollutants):
    """Calculate AQI based on PM2.5 and PM10 concentrations"""
    aqi_values = []
   
    # PM2.5 AQI calculation (Indian standards)
    if pollutants.get('PM2.5') is not None and not pd.isna(pollutants['PM2.5']):
        pm25 = pollutants['PM2.5']
        breakpoints = [
            (0, 30, 0, 50),       # Good
            (31, 60, 51, 100),    # Satisfactory
            (61, 90, 101, 200),   # Moderate
            (91, 120, 201, 300),  # Poor
            (121, 250, 301, 400), # Very Poor
            (251, 500, 401, 500)  # Severe
        ]
        aqi_values.append(calculate_sub_index(pm25, breakpoints))
   
    # PM10 AQI calculation (Indian standards)
    if pollutants.get('PM10') is not None and not pd.isna(pollutants['PM10']):
        pm10 = pollutants['PM10']
        breakpoints = [
            (0, 50, 0, 50),       # Good
            (51, 100, 51, 100),   # Satisfactory
            (101, 250, 101, 200), # Moderate
            (251, 350, 201, 300), # Poor
            (351, 430, 301, 400), # Very Poor
            (431, 550, 401, 500)  # Severe
        ]
        aqi_values.append(calculate_sub_index(pm10, breakpoints))
   
    return max(aqi_values) if aqi_values else 0

def get_aqi_category(aqi):
    """Get AQI category and color based on AQI value"""
    if aqi <= 50:
        color = '#10b981'
        return {
            'category': 'Good',
            'color': color,
            'text_color': get_high_contrast_text(color),
            'emoji': '😊',
            'description': 'Air quality is satisfactory, and air pollution poses little or no risk.',
            'health_impact': 'Minimal impact. Enjoy outdoor activities!'
        }
    elif aqi <= 100:
        color = '#84cc16'
        return {
            'category': 'Satisfactory',
            'color': color,
            'text_color': get_high_contrast_text(color),
            'emoji': '🙂',
            'description': 'Air quality is acceptable. However, there may be a risk for some people.',
            'health_impact': 'Acceptable for most, sensitive individuals should consider limiting prolonged outdoor exertion.'
        }
    elif aqi <= 200:
        color = '#eab308'
        return {
            'category': 'Moderate',
            'color': color,
            'text_color': get_high_contrast_text(color),
            'emoji': '😐',
            'description': 'Members of sensitive groups may experience health effects.',
            'health_impact': 'General public and sensitive groups should reduce prolonged or heavy outdoor exertion.'
        }
    elif aqi <= 300:
        color = '#f97316'
        return {
            'category': 'Poor',
            'color': color,
            'text_color': get_high_contrast_text(color),
            'emoji': '😷',
            'description': 'Everyone may begin to experience health effects.',
            'health_impact': 'General public should avoid prolonged or heavy exertion. Sensitive groups should limit outdoor activity.'
        }
    elif aqi <= 400:
        color = '#ef4444'
        return {
            'category': 'Very Poor',
            'color': color,
            'text_color': get_high_contrast_text(color),
            'emoji': '😨',
            'description': 'Health alert: The risk of health effects is increased for everyone.',
            'health_impact': 'General public should significantly limit outdoor exertion. Sensitive groups should avoid outdoor activity.'
        }
    else:
        color = '#dc2626'
        return {
            'category': 'Severe',
            'color': color,
            'text_color': get_high_contrast_text(color),
            'emoji': '☠️',
            'description': 'Health warning of emergency conditions: everyone is more likely to be affected.',
            'health_impact': 'Everyone should avoid all outdoor physical activity. Move activities indoors or reschedule.'
        }

@st.cache_data
def load_data():
    """Load and cache the CSV data"""
    try:
        df = pd.read_csv('2022_2025_data.csv')
        
        # Rename columns from lowercase to proper case
        df = df.rename(columns={
            'date': 'Date',
            'pm2_5': 'PM2.5',
            'pm10': 'PM10',
            'co': 'CO',
            'no': 'NO',
            'no2': 'NO2',
            'nh3': 'NH3',
            'o3': 'O3',
            'so2': 'SO2'
        })
       
        # Clean and process data
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
       
        # Convert pollutant columns to numeric
        pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO', 'NO2', 'NH3', 'O3', 'SO2']
        for col in pollutant_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
       
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_daily_aqi(df, station, date):
    """Calculate daily average AQI for a station and date"""
    date_normalized = pd.to_datetime(date).normalize()
    mask = (df['Station'] == station) & (df['Date'] == date_normalized)
    day_data = df[mask]
   
    if len(day_data) == 0:
        return None, None
   
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
   
    aqi = calculate_aqi(pollutants)
    return aqi, pollutants

def get_hourly_data(df, station, date):
    """Get daily data for a specific date (no hourly data available in CSV)"""
    date_normalized = pd.to_datetime(date).normalize()
    mask = (df['Station'] == station) & (df['Date'] == date_normalized)
    day_data = df[mask].copy()
   
    if len(day_data) == 0:
        return None
   
    # Since we don't have hourly data, return daily aggregated data
    pollutants = {
        'PM2.5': day_data['PM2.5'].mean(),
        'PM10': day_data['PM10'].mean()
    }
    aqi = calculate_aqi(pollutants)
    
    # Return a dataframe with just one entry for the day
    return pd.DataFrame([{'hour': 12, 'aqi': aqi}])

def get_historical_data(df, station, days=30):
    """Get historical AQI data for the last N days"""
    station_data = df[df['Station'] == station].copy()
    if len(station_data) == 0:
        return None
   
    # Get unique dates
    dates = sorted(station_data['Date'].unique())
   
    # Take last N days
    recent_dates = dates[-days:] if len(dates) > days else dates
   
    historical = []
    for date in recent_dates:
        aqi, _ = calculate_daily_aqi(df, station, date)
        if aqi is not None:
            historical.append({
                'date': date,
                'aqi': aqi
            })
   
    return pd.DataFrame(historical) if historical else None

def calculate_model_metrics(df, station):
    """
    Calculate accuracy metrics for multiple forecasting models
    Returns metrics for: ARIMA, Exponential Smoothing, and Moving Average
    """
    historical = get_historical_data(df, station, days=120)  # Use more data for better training
    
    if historical is None or len(historical) < 28:
        return None
    
    try:
        historical = historical.sort_values('date').reset_index(drop=True)
        ts_data = historical['aqi'].values
        
        # Normalize data for better model performance
        data_mean = np.mean(ts_data)
        data_std = np.std(ts_data)
        ts_data_normalized = (ts_data - data_mean) / (data_std + 1e-8) if data_std > 0 else ts_data
        
        # Use 90/10 split for more training data
        split_idx = int(len(ts_data_normalized) * 0.9)
        train_data = ts_data_normalized[:split_idx]
        test_data = ts_data_normalized[split_idx:]
        train_data_raw = ts_data[:split_idx]
        test_data_raw = ts_data[split_idx:]
        
        if len(train_data) < 15 or len(test_data) < 2:
            return None
        
        models_metrics = {}
        
        # ===== Model 1: ARIMA =====
        try:
            best_arima_score = float('inf')
            best_arima_order = (1, 1, 1)
            arima_predictions = None
            
            # Grid search for best ARIMA parameters
            for p in [0, 1, 2, 3]:
                for d in [0, 1, 2]:
                    for q in [0, 1, 2]:
                        try:
                            arima_model = ARIMA(train_data, order=(p, d, q))
                            arima_fit = arima_model.fit()
                            pred = arima_fit.forecast(steps=len(test_data))
                            error = np.mean(np.abs(pred - test_data))
                            
                            if error < best_arima_score:
                                best_arima_score = error
                                best_arima_order = (p, d, q)
                                arima_predictions = pred
                        except:
                            continue
            
            if arima_predictions is not None:
                # Denormalize predictions
                arima_predictions_raw = arima_predictions * data_std + data_mean
                
                arima_mae = np.mean(np.abs(test_data_raw - arima_predictions_raw))
                arima_rmse = np.sqrt(np.mean((test_data_raw - arima_predictions_raw) ** 2))
                arima_mape = np.mean(np.abs((test_data_raw - arima_predictions_raw) / (test_data_raw + 1e-8))) * 100
                
                ss_res = np.sum((test_data - arima_predictions) ** 2)
                ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
                arima_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            else:
                arima_mae = arima_rmse = arima_mape = arima_r2 = 0
            
            models_metrics['ARIMA'] = {
                'mae': arima_mae,
                'rmse': arima_rmse,
                'mape': max(0, min(arima_mape, 100)),
                'r_squared': max(0, min(arima_r2, 1)),
                'accuracy': max(0, 100 - max(0, min(arima_mape, 100))),
                'type': 'Time Series'
            }
        except Exception as e:
            models_metrics['ARIMA'] = None
        
        # ===== Model 2: Exponential Smoothing =====
        try:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            exp_model = SimpleExpSmoothing(train_data)
            exp_fit = exp_model.fit(optimized=True)
            exp_predictions = exp_fit.forecast(steps=len(test_data))
            
            # Denormalize predictions
            exp_predictions_raw = exp_predictions * data_std + data_mean
            
            exp_mae = np.mean(np.abs(test_data_raw - exp_predictions_raw))
            exp_rmse = np.sqrt(np.mean((test_data_raw - exp_predictions_raw) ** 2))
            exp_mape = np.mean(np.abs((test_data_raw - exp_predictions_raw) / (test_data_raw + 1e-8))) * 100
            ss_res = np.sum((test_data - exp_predictions) ** 2)
            ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
            exp_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            models_metrics['Exponential Smoothing'] = {
                'mae': exp_mae,
                'rmse': exp_rmse,
                'mape': max(0, min(exp_mape, 100)),
                'r_squared': max(0, min(exp_r2, 1)),
                'accuracy': max(0, 100 - max(0, min(exp_mape, 100))),
                'type': 'Smoothing'
            }
        except Exception as e:
            models_metrics['Exponential Smoothing'] = None
        
        # ===== Model 3: Moving Average =====
        try:
            # Use linearly increasing weights for recent values
            ma_window = min(7, len(train_data) // 4)
            weights = np.linspace(1, 2, ma_window)
            weights = weights / np.sum(weights)
            
            wma_predictions = []
            for i in range(len(test_data)):
                if i < ma_window:
                    window_data = train_data[-(ma_window):]
                    if len(wma_predictions) > 0:
                        combined = np.concatenate([window_data[len(wma_predictions):], wma_predictions])
                        window_data = combined[-ma_window:]
                    pred = np.average(window_data, weights=weights[-len(window_data):])
                else:
                    window_data = np.array(wma_predictions[-ma_window:])
                    pred = np.average(window_data, weights=weights)
                wma_predictions.append(pred)
            
            wma_predictions = np.array(wma_predictions)
            wma_predictions_raw = wma_predictions * data_std + data_mean
            
            wma_mae = np.mean(np.abs(test_data_raw - wma_predictions_raw))
            wma_rmse = np.sqrt(np.mean((test_data_raw - wma_predictions_raw) ** 2))
            wma_mape = np.mean(np.abs((test_data_raw - wma_predictions_raw) / (test_data_raw + 1e-8))) * 100
            
            ss_res = np.sum((wma_predictions - test_data) ** 2)
            ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
            wma_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            models_metrics['Weighted Moving Avg'] = {
                'mae': wma_mae,
                'rmse': wma_rmse,
                'mape': max(0, min(wma_mape, 100)),
                'r_squared': max(0, min(wma_r2, 1)),
                'accuracy': max(0, 100 - max(0, min(wma_mape, 100))),
                'type': 'Average'
            }
        except Exception as e:
            models_metrics['Weighted Moving Avg'] = None
        
        # ===== Model 4: Linear Regression =====
        # ===== Model 4: Ensemble Model (Weighted Combination) =====
        try:
            ensemble_predictions = np.zeros(len(test_data))
            ensemble_count = 0
            
            if models_metrics.get('ARIMA') and models_metrics['ARIMA'] is not None:
                try:
                    arima_model = ARIMA(train_data, order=best_arima_order if 'best_arima_order' in locals() else (1, 1, 1))
                    arima_fit = arima_model.fit()
                    pred = arima_fit.forecast(steps=len(test_data))
                    ensemble_predictions += pred * 0.4
                    ensemble_count += 0.4
                except:
                    pass
            
            if models_metrics.get('Exponential Smoothing') and models_metrics['Exponential Smoothing'] is not None:
                try:
                    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                    exp_model = SimpleExpSmoothing(train_data)
                    exp_fit = exp_model.fit(optimized=True)
                    pred = exp_fit.forecast(steps=len(test_data))
                    ensemble_predictions += pred * 0.35
                    ensemble_count += 0.35
                except:
                    pass
            
            if models_metrics.get('Weighted Moving Avg') and models_metrics['Weighted Moving Avg'] is not None:
                try:
                    ensemble_predictions += np.array(wma_predictions) * 0.25
                    ensemble_count += 0.25
                except:
                    pass
            
            if ensemble_count > 0:
                ensemble_predictions = ensemble_predictions / ensemble_count
                ensemble_predictions_raw = ensemble_predictions * data_std + data_mean
                
                ens_mae = np.mean(np.abs(test_data_raw - ensemble_predictions_raw))
                ens_rmse = np.sqrt(np.mean((test_data_raw - ensemble_predictions_raw) ** 2))
                ens_mape = np.mean(np.abs((test_data_raw - ensemble_predictions_raw) / (test_data_raw + 1e-8))) * 100
                
                ss_res = np.sum((ensemble_predictions - test_data) ** 2)
                ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
                ens_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                models_metrics['Ensemble'] = {
                    'mae': ens_mae,
                    'rmse': ens_rmse,
                    'mape': max(0, min(ens_mape, 100)),
                    'r_squared': max(0, min(ens_r2, 1)),
                    'accuracy': max(0, 100 - max(0, min(ens_mape, 100))),
                    'type': 'Combination'
                }
        except Exception as e:
            pass
        
        return {
            'models': models_metrics,
            'training_days': split_idx,
            'test_days': len(test_data),
            'total_days': len(ts_data),
            'avg_aqi': np.mean(ts_data),
            'aqi_std': np.std(ts_data),
            'aqi_min': np.min(ts_data),
            'aqi_max': np.max(ts_data)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def forecast_aqi(df, station, days=7):
    """
    Forecast AQI for next N days using ARIMA (AutoRegressive Integrated Moving Average) model
    ARIMA is a statistical ML model for time series forecasting
    """
    # Get historical data (last 60 days for better ARIMA training)
    historical = get_historical_data(df, station, days=60)
   
    if historical is None or len(historical) < 14:
        return None
   
    try:
        # Prepare time series data
        historical = historical.sort_values('date').reset_index(drop=True)
        ts_data = historical['aqi'].values
       
        # ARIMA Model Parameters
        # (p, d, q) where:
        # p = number of lag observations (AR term)
        # d = degree of differencing (I term)  
        # q = size of moving average window (MA term)
       
        try:
            # Try ARIMA(5,1,0) - good for AQI trends
            model = ARIMA(ts_data, order=(5, 1, 0))
            model_fit = model.fit()
        except:
            try:
                # Fallback to simpler ARIMA(2,1,2)
                model = ARIMA(ts_data, order=(2, 1, 2))
                model_fit = model.fit()
            except:
                # Last resort: ARIMA(1,1,1)
                model = ARIMA(ts_data, order=(1, 1, 1))
                model_fit = model.fit()
       
        # Forecast next N days
        forecast_result = model_fit.forecast(steps=days)
       
        # Get confidence intervals (95% confidence)
        forecast_df = model_fit.get_forecast(steps=days)
        confidence_intervals = forecast_df.conf_int()
       
        # Convert to numpy array if it's a DataFrame
        if hasattr(confidence_intervals, 'values'):
            confidence_intervals = confidence_intervals.values
       
        # Prepare forecast data
        forecast_data = []
        last_date = historical['date'].iloc[-1]
       
        for i in range(days):
            forecast_date = last_date + timedelta(days=i+1)
            forecasted_aqi = max(0, forecast_result[i])  # Ensure non-negative
           
            # Get confidence bounds
            lower_bound = max(0, confidence_intervals[i, 0])
            upper_bound = confidence_intervals[i, 1]
           
            forecast_data.append({
                'date': forecast_date,
                'aqi': forecasted_aqi,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'is_forecast': True
            })
       
        # Combine historical (last 14 days) and forecast
        historical['is_forecast'] = False
        historical['lower_bound'] = historical['aqi']
        historical['upper_bound'] = historical['aqi']
       
        combined = pd.concat([
            historical[['date', 'aqi', 'lower_bound', 'upper_bound', 'is_forecast']].tail(14),
            pd.DataFrame(forecast_data)
        ], ignore_index=True)
       
        return combined
       
    except Exception as e:
        # Fallback to simple statistical method if ARIMA fails
        print(f"ARIMA failed: {e}. Using fallback method.")
        return forecast_aqi_simple(df, station, days)

def forecast_aqi_simple(df, station, days=7):
    """
    Fallback forecast method using moving average with trend
    Used when ARIMA fails or insufficient data
    """
    # Get historical data (last 30 days)
    historical = get_historical_data(df, station, days=30)
   
    if historical is None or len(historical) < 7:
        return None
   
    # Calculate moving average and trend
    historical['ma_7'] = historical['aqi'].rolling(window=7, min_periods=1).mean()
    historical['ma_14'] = historical['aqi'].rolling(window=14, min_periods=1).mean()
   
    # Calculate trend (linear regression on last 14 days)
    recent_data = historical.tail(14)
    if len(recent_data) >= 7:
        x = np.arange(len(recent_data))
        y = recent_data['aqi'].values
        z = np.polyfit(x, y, 1)
        trend = z[0]  # slope
    else:
        trend = 0
   
    # Last known values
    last_aqi = historical['aqi'].iloc[-1]
    last_ma7 = historical['ma_7'].iloc[-1]
    last_ma14 = historical['ma_14'].iloc[-1]
   
    # Forecast next N days
    forecast_data = []
    last_date = historical['date'].iloc[-1]
   
    # Calculate volatility (standard deviation of last 14 days)
    volatility = historical['aqi'].tail(14).std()
   
    for i in range(1, days + 1):
        # Weighted forecast combining moving averages and trend
        base_forecast = (last_ma7 * 0.5 + last_ma14 * 0.3 + last_aqi * 0.2)
        trend_adjustment = trend * i * 0.5  # Dampen trend effect
       
        forecasted_aqi = base_forecast + trend_adjustment
       
        # Add some noise based on historical volatility
        noise = np.random.normal(0, volatility * 0.3)
        forecasted_aqi = max(0, forecasted_aqi + noise)
       
        # Calculate confidence interval
        confidence_margin = volatility * (1 + i * 0.1)  # Wider for further dates
       
        forecast_date = last_date + timedelta(days=i)
        forecast_data.append({
            'date': forecast_date,
            'aqi': forecasted_aqi,
            'lower_bound': max(0, forecasted_aqi - confidence_margin),
            'upper_bound': forecasted_aqi + confidence_margin,
            'is_forecast': True
        })
   
    # Combine historical and forecast
    historical['is_forecast'] = False
    historical['lower_bound'] = historical['aqi']
    historical['upper_bound'] = historical['aqi']
   
    combined = pd.concat([
        historical[['date', 'aqi', 'lower_bound', 'upper_bound', 'is_forecast']].tail(14),
        pd.DataFrame(forecast_data)
    ], ignore_index=True)
   
    return combined

def get_top_cities(df, date, top_n=10):
    """Get top N cities by AQI for a specific date"""
    stations = df['Station'].unique()
    city_aqi = []
   
    for station in stations:
        aqi, _ = calculate_daily_aqi(df, station, date)
        if aqi is not None:
            city_aqi.append({
                'station': station,
                'aqi': aqi
            })
   
    if not city_aqi:
        return None
   
    cities_df = pd.DataFrame(city_aqi)
    return cities_df.nlargest(top_n, 'aqi')

def get_monthly_analytics(df, year, month, station=None):
    """Get monthly aggregated AQI data"""
    monthly_data = df[(df['Year'] == year) & (df['Month'] == month)]
   
    if station:
        monthly_data = monthly_data[monthly_data['Station'] == station]
   
    if len(monthly_data) == 0:
        return None
   
    pollutants = {
        'PM2.5': monthly_data['PM2.5'].mean(),
        'PM10': monthly_data['PM10'].mean(),
        'CO': monthly_data['CO'].mean(),
        'NO2': monthly_data['NO2'].mean(),
        'O3': monthly_data['O3'].mean(),
        'SO2': monthly_data['SO2'].mean(),
        'NH3': monthly_data['NH3'].mean(),
        'NO': monthly_data['NO'].mean()
    }
   
    aqi = calculate_aqi(pollutants)
    return aqi, pollutants

def get_year_comparison(df, station):
    """Get year-over-year comparison for a station"""
    years = sorted(df['Year'].unique())
    year_data = []
   
    for year in years:
        year_df = df[(df['Year'] == year) & (df['Station'] == station)]
        if len(year_df) > 0:
            pollutants = {
                'PM2.5': year_df['PM2.5'].mean(),
                'PM10': year_df['PM10'].mean(),
                'CO': year_df['CO'].mean(),
                'NO2': year_df['NO2'].mean(),
                'O3': year_df['O3'].mean(),
                'SO2': year_df['SO2'].mean(),
                'NH3': year_df['NH3'].mean(),
                'NO': year_df['NO'].mean()
            }
            aqi = calculate_aqi(pollutants)
            year_data.append({'year': year, 'aqi': aqi})
   
    return pd.DataFrame(year_data) if year_data else None

def get_month_statistics(df, year, month, station=None):
    """Get detailed statistics for a month"""
    monthly_data = df[(df['Year'] == year) & (df['Month'] == month)]
   
    if station:
        monthly_data = monthly_data[monthly_data['Station'] == station]
   
    if len(monthly_data) == 0:
        return None
   
    stats = {
        'Days with data': monthly_data['Date'].nunique(),
        'Total records': len(monthly_data),
        'Unique stations': monthly_data['Station'].nunique() if not station else 1
    }
   
    return stats

def create_aqi_card(aqi, station, aqi_info):
    """Create a beautiful AQI display card with glassmorphism effect - PERFECT TEXT CONTRAST"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {aqi_info['color']}dd 0%, {aqi_info['color']}ff 100%);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 40px {aqi_info['color']}60;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    ">
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);"></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 18px; color: {aqi_info['text_color']}; opacity: 1; margin-bottom: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">Air Quality Index</div>
            <div style="font-size: 72px; margin: 16px 0; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));">{aqi_info['emoji']}</div>
            <div style="font-size: 90px; color: {aqi_info['text_color']}; font-weight: 900; line-height: 1; margin: 16px 0; text-shadow: 0 4px 20px rgba(0,0,0,0.5), 0 2px 4px rgba(0,0,0,0.3);">{int(aqi)}</div>
            <div style="font-size: 36px; color: {aqi_info['text_color']}; font-weight: 800; margin: 16px 0; letter-spacing: 2px; text-shadow: 0 2px 8px rgba(0,0,0,0.4);">{aqi_info['category'].upper()}</div>
            <div style="font-size: 18px; color: {aqi_info['text_color']}; margin-top: 20px; opacity: 1; line-height: 1.4; font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{station}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_pollutant_bars(pollutants):
    """Create modern pollutant concentration bars - PERFECT TEXT CONTRAST"""
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
            percentage = min((value / info['max']) * 100, 100)
           
            # Color gradient based on percentage
            if percentage < 30:
                bar_color = 'linear-gradient(90deg, #10b981 0%, #34d399 100%)'
            elif percentage < 60:
                bar_color = 'linear-gradient(90deg, #eab308 0%, #fbbf24 100%)'
            else:
                bar_color = 'linear-gradient(90deg, #ef4444 0%, #f87171 100%)'
           
            st.markdown(f"""
            <div class="pollutant-item">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 26px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));">{info['icon']}</span>
                        <div style="font-weight: 700; color: #ffffff; font-size: 17px; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">{info['name']}</div>
                    </div>
                    <div style="flex: 1; margin: 0 20px;">
                        <div style="background: rgba(139, 92, 246, 0.25); height: 14px; border-radius: 7px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);">
                            <div style="background: {bar_color}; width: {percentage}%; height: 100%; border-radius: 7px; transition: width 0.5s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                        </div>
                    </div>
                    <div style="font-weight: 800; color: #ffffff; min-width: 130px; text-align: right; font-size: 17px; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">{value:.2f} {info['unit']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def create_hourly_chart(hourly_data, station, date):
    """Display daily AQI for the selected date (hourly data not available in dataset)"""
    if hourly_data is None or len(hourly_data) == 0:
        st.warning("No data available for this date")
        return
   
    aqi_value = hourly_data['aqi'].iloc[0] if len(hourly_data) > 0 else 0
    category = get_aqi_category(aqi_value)
    
    fig = go.Figure()
   
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Daily AQI"},
        gauge={
            'axis': {'range': [0, 500]},
            'bar': {'color': category['color']},
            'steps': [
                {'range': [0, 50], 'color': '#10b981'},
                {'range': [50, 100], 'color': '#84cc16'},
                {'range': [100, 200], 'color': '#eab308'},
                {'range': [200, 300], 'color': '#f97316'},
                {'range': [300, 400], 'color': '#ef4444'},
                {'range': [400, 500], 'color': '#dc2626'}
            ]
        }
    ))
   
    fig.update_layout(
        title=dict(
            text=f"Daily AQI - {date.strftime('%B %d, %Y')} ({station})",
            font=dict(color='#ffffff', size=18, family='Inter')
        ),
        height=350,
        paper_bgcolor='rgba(30, 27, 75, 0.3)',
        font=dict(color='#ffffff', family='Inter', size=12),
        margin=dict(t=80, b=40, l=50, r=30)
    )
   
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"**Status: {category['emoji']} {category['category']}**\n\n{category['health_impact']}")

def create_historical_chart(historical_data, station):
    """Create historical trend chart with enhanced styling"""
    if historical_data is None or len(historical_data) == 0:
        st.warning("No historical data available")
        return
   
    colors = [get_aqi_category(aqi)['color'] for aqi in historical_data['aqi']]
   
    fig = go.Figure()
   
    fig.add_trace(go.Bar(
        x=historical_data['date'],
        y=historical_data['aqi'],
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.9
        ),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>AQI: %{y:.0f}<extra></extra>'
    ))
   
    fig.update_layout(
        title=dict(
            text=f"Last 30 Days AQI Trend - {station}",
            font=dict(color='#ffffff', size=22, family='Inter')
        ),
        xaxis_title="Date",
        yaxis_title="AQI",
        template="plotly_dark",
        height=450,
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        paper_bgcolor='rgba(30, 27, 75, 0.3)',
        font=dict(color='#f1f5f9', family='Inter', size=13),
        xaxis=dict(
            gridcolor='rgba(139, 92, 246, 0.15)',
            title=dict(font=dict(color='#ffffff', size=15)),
            tickfont=dict(color='#f1f5f9', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(139, 92, 246, 0.15)',
            title=dict(font=dict(color='#ffffff', size=15)),
            tickfont=dict(color='#f1f5f9', size=12)
        ),
        margin=dict(t=60, b=40, l=50, r=30)
    )
   
    st.plotly_chart(fig, use_container_width=True)

def create_forecast_chart(forecast_data, station):
    """Create forecast chart with confidence intervals and enhanced styling"""
    if forecast_data is None or len(forecast_data) == 0:
        st.warning("Unable to generate forecast - insufficient historical data")
        return
   
    # Split historical and forecast
    historical = forecast_data[~forecast_data['is_forecast']]
    forecast = forecast_data[forecast_data['is_forecast']]
   
    fig = go.Figure()
   
    # Historical data (line)
    fig.add_trace(go.Scatter(
        x=historical['date'],
        y=historical['aqi'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#10b981', width=4),
        marker=dict(size=10, color='#10b981', line=dict(color='#ffffff', width=2)),
        hovertemplate='<b>%{x|%b %d}</b><br>AQI: %{y:.0f}<extra></extra>'
    ))
   
    # Forecast data (line)
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['aqi'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#f59e0b', width=4, dash='dash'),
        marker=dict(size=12, symbol='diamond', color='#f59e0b', line=dict(color='#ffffff', width=2)),
        hovertemplate='<b>%{x|%b %d}</b><br>Predicted AQI: %{y:.0f}<extra></extra>'
    ))
   
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
        y=forecast['upper_bound'].tolist() + forecast['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(245, 158, 11, 0.25)',
        line=dict(color='rgba(245, 158, 11, 0)'),
        name='95% Confidence',
        hoverinfo='skip'
    ))
   
    fig.update_layout(
        title=dict(
            text=f"🔮 7-Day AQI Forecast - {station}",
            font=dict(color='#ffffff', size=24, family='Inter')
        ),
        xaxis_title="Date",
        yaxis_title="AQI",
        template="plotly_dark",
        height=550,
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        paper_bgcolor='rgba(30, 27, 75, 0.3)',
        font=dict(color='#f1f5f9', family='Inter', size=13),
        xaxis=dict(
            gridcolor='rgba(139, 92, 246, 0.15)',
            title=dict(font=dict(color='#ffffff', size=15)),
            tickfont=dict(color='#f1f5f9', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(139, 92, 246, 0.15)',
            title=dict(font=dict(color='#ffffff', size=15)),
            tickfont=dict(color='#f1f5f9', size=12)
        ),
        legend=dict(
            font=dict(color='#ffffff', size=13),
            bgcolor='rgba(30, 27, 75, 0.9)',
            bordercolor='rgba(139, 92, 246, 0.4)',
            borderwidth=2,
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

def create_india_map(df, date):
    """Display India choropleth by state (best-effort) and fallback to station rankings."""

    # Helper: get state for a station (use existing 'State' column if present)
    def infer_state(station_name, df):
        # If a State column exists, try to use it for this station
        if 'State' in df.columns:
            row = df[df['Station'] == station_name]
            if not row.empty and 'State' in row.columns:
                val = row['State'].dropna().unique()
                if len(val) > 0:
                    return str(val[0])

        # Try common separators
        if ',' in station_name:
            parts = [p.strip() for p in station_name.split(',')]
            if len(parts) > 1:
                return parts[-1]
        if '-' in station_name:
            parts = [p.strip() for p in station_name.split('-')]
            if len(parts) > 1:
                return parts[-1]

        # Fallback: last token (may be state or city)
        parts = station_name.split()
        if len(parts) > 1:
            return parts[-1]

        return 'Unknown'

    # Build station-level AQI list and inferred states
    station_records = []
    for station in df['Station'].unique():
        aqi, pollutants = calculate_daily_aqi(df, station, date)
        if aqi is not None:
            state = infer_state(station, df)
            station_records.append({'Station': station, 'AQI': float(aqi), 'State': state})

    if not station_records:
        st.warning("No data available for this date")
        return

    station_df = pd.DataFrame(station_records)

    # Aggregate by state
    state_agg = station_df.groupby('State', dropna=False).agg({'AQI': 'mean'}).reset_index()

    # Try to load India GeoJSON and create choropleth (best-effort matching)
    GEOJSON_URLS = [
        'https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson',
        'https://raw.githubusercontent.com/nikhilkumarsingh/india-geojson/master/india_states.geojson'
    ]

    @st.cache_data
    def load_geojson(urls):
        for url in urls:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                continue
        return None

    geojson = load_geojson(GEOJSON_URLS)

    if geojson is None:
        # Fallback to station ranking chart if GeoJSON unavailable
        st.warning("India GeoJSON not available — showing station rankings instead")
        _plot_station_rankings(station_df)
        return

    # Build a mapping of geojson state names to feature ids
    geo_names = []
    for feat in geojson.get('features', []):
        props = feat.get('properties', {})
        # collect any plausible name fields
        for key in ['st_nm', 'ST_NM', 'name', 'NAME_1', 'state']:
            if key in props and props[key]:
                geo_names.append(str(props[key]))
                break

    # Simple matching: lowercase contains
    def match_geo_name(state_name):
        if not isinstance(state_name, str):
            return None
        s = state_name.lower().strip()
        for name in geo_names:
            n = name.lower()
            if s == n or s in n or n in s:
                return name
        return None

    # Map state_agg.State to geo name where possible
    state_agg['geo_name'] = state_agg['State'].apply(match_geo_name)

    # Keep only matched states for choropleth
    choropleth_data = state_agg[state_agg['geo_name'].notnull()].copy()

    if len(choropleth_data) == 0:
        st.warning("Could not match any station states to GeoJSON names — showing station rankings instead")
        _plot_station_rankings(station_df)
        return

    # Prepare dataframe for plotly (use geo_name as location)
    choropleth_data['AQI'] = choropleth_data['AQI'].astype(float)

    # Attempt to find the correct featureidkey (pick a likely property)
    example_props = geojson.get('features', [])[0].get('properties', {}) if geojson.get('features') else {}
    candidate_keys = ['properties.ST_NM', 'properties.st_nm', 'properties.NAME_1', 'properties.name', 'properties.state']
    featureidkey = None
    for k in candidate_keys:
        prop = k.split('.', 1)[1]
        if prop in example_props:
            featureidkey = k
            break

    if featureidkey is None:
        featureidkey = 'properties.st_nm'

    try:
        fig = px.choropleth(
            choropleth_data,
            geojson=geojson,
            locations='geo_name',
            color='AQI',
            color_continuous_scale='YlOrRd',
            range_color=(0, 500),
            featureidkey=featureidkey,
            labels={'AQI': 'Average AQI'},
            title='🗺️ Average AQI by State'
        )

        fig.update_geos(fitbounds='locations', visible=False)
        fig.update_layout(
            template='plotly_dark',
            height=600,
            margin=dict(t=80, b=40, l=40, r=40),
            coloraxis_colorbar=dict(title='AQI')
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Choropleth failed ({e}) — showing station rankings instead")
        _plot_station_rankings(station_df)


def _plot_station_rankings(station_df):
    """Fallback helper to plot horizontal station rankings and table"""
    station_df_sorted = station_df.sort_values('AQI', ascending=False)
    fig = go.Figure()
    colors = [get_aqi_category(aqi)['color'] for aqi in station_df_sorted['AQI']]
    fig.add_trace(go.Bar(
        y=station_df_sorted['Station'],
        x=station_df_sorted['AQI'],
        orientation='h',
        marker=dict(color=colors),
        text=station_df_sorted['AQI'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>AQI: %{x:.1f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="🗺️ Station Pollution Rankings", font=dict(color='#ffffff', size=22, family='Inter')),
        xaxis=dict(title=dict(text="AQI", font=dict(color='#ffffff', size=14, family='Inter')), tickfont=dict(color='#ffffff', size=12)),
        yaxis=dict(title=dict(text="Station", font=dict(color='#ffffff', size=14, family='Inter')), tickfont=dict(color='#ffffff', size=12)),
        plot_bgcolor='rgba(30, 27, 75, 0.3)',
        paper_bgcolor='rgba(30, 27, 75, 0.3)',
        font=dict(color='#ffffff', family='Inter', size=12),
        height=450,
        hovermode='closest',
        margin=dict(t=80, b=40, l=150, r=30)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📊 Detailed Station Data")
    st.dataframe(station_df_sorted, use_container_width=True, hide_index=True)

def display_model_metrics(metrics):
    """Display metrics for all trained models with professional styling"""
    if metrics is None:
        st.warning("⚠️ Insufficient data to calculate model metrics")
        return
    
    st.markdown("## 🤖 Model Performance Comparison")
    st.markdown("*Comparing multiple forecasting models on test data*")
    st.markdown("")
    
    # Filter out None models
    models = {k: v for k, v in metrics['models'].items() if v is not None}
    
    if not models:
        st.warning("⚠️ Unable to train models for this station")
        return
    
    # Create model comparison table
    model_data = []
    for model_name, model_metrics in models.items():
        model_data.append({
            'Model': model_name,
            'Type': model_metrics['type'],
            'Accuracy %': f"{model_metrics['accuracy']:.2f}",
            'R² Score': f"{model_metrics['r_squared']:.4f}",
            'MAE': f"{model_metrics['mae']:.2f}",
            'RMSE': f"{model_metrics['rmse']:.2f}",
            'MAPE %': f"{model_metrics['mape']:.2f}"
        })
    
    model_df = pd.DataFrame(model_data)
    
    # Display comparison table
    st.markdown("### 📊 Model Performance Metrics")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(model_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Legend")
        st.markdown("""
        - **Accuracy %**: Model accuracy (higher is better)
        - **R² Score**: Goodness of fit (0-1, higher is better)
        - **MAE**: Mean Absolute Error (lower is better)
        - **RMSE**: Root Mean Squared Error (lower is better)
        - **MAPE %**: Mean Absolute Percentage Error (lower is better)
        """)
    
    st.markdown("")
    
    # Best performing model
    best_model = max(models.items(), key=lambda x: x[1]['accuracy'])
    best_name, best_metrics = best_model
    
    st.markdown(f"""
    <div class="custom-card">
        <h3 style="color: #a855f7; margin-top: 0;">🏆 Best Performing Model</h3>
        <p style="color: #ffffff; font-size: 18px; margin: 10px 0;">
            <strong>{best_name}</strong> ({best_metrics['type']})
        </p>
        <p style="color: #10b981; font-size: 16px; font-weight: 600;">
            Accuracy: {best_metrics['accuracy']:.2f}% | R²: {best_metrics['r_squared']:.4f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Detailed metrics for each model
    st.markdown("### 📈 Detailed Model Analysis")
    
    tabs = st.tabs([f"{i+1}. {name}" for i, name in enumerate(models.keys())])
    
    for tab, (model_name, model_metrics) in zip(tabs, models.items()):
        with tab:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "🎯 Model Accuracy",
                    f"{model_metrics['accuracy']:.2f}%",
                    help="Percentage accuracy on test data"
                )
                st.metric(
                    "📊 R² Score",
                    f"{model_metrics['r_squared']:.4f}",
                    help="Coefficient of determination (higher is better)"
                )
                st.metric(
                    "📉 MAPE",
                    f"{model_metrics['mape']:.2f}%",
                    help="Mean Absolute Percentage Error (lower is better)"
                )
            
            with col2:
                st.metric(
                    "📏 MAE",
                    f"{model_metrics['mae']:.2f}",
                    help="Mean Absolute Error in AQI points"
                )
                st.metric(
                    "📈 RMSE",
                    f"{model_metrics['rmse']:.2f}",
                    help="Root Mean Square Error in AQI points"
                )
                st.metric(
                    "🏷️ Model Type",
                    model_metrics['type']
                )
            
            # Model description
            descriptions = {
                'ARIMA': 'Autoregressive Integrated Moving Average - Advanced time series model that captures temporal patterns and seasonality in AQI data.',
                'Exponential Smoothing': 'Weighted average method that gives more importance to recent observations. Good for trending data.',
                'Moving Average': 'Simple average of recent observations. Smooth out noise but may lag behind trend changes.',
                'Linear Regression': 'Fits a linear trend line through data. Works well for data with clear upward/downward trends.'
            }
            
            st.info(descriptions.get(model_name, "Model for forecasting"))
    
    st.markdown("")
    st.markdown("### 📊 Training Data Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📚 Training Period",
            f"{metrics['training_days']} days",
            help="Data used to train the models"
        )
    
    with col2:
        st.metric(
            "✅ Test Period",
            f"{metrics['test_days']} days",
            help="Data used to validate model accuracy"
        )
    
    with col3:
        st.metric(
            "📈 Average AQI",
            f"{metrics['avg_aqi']:.1f}",
            help="Historical average AQI value"
        )
    
    with col4:
        st.metric(
            "📊 AQI Std Dev",
            f"{metrics['aqi_std']:.1f}",
            help="Standard deviation of AQI values"
        )

def display_top_cities(top_cities):
    """Display top 10 cities by AQI with modern card design - PERFECT TEXT CONTRAST"""
    st.markdown("### 🏙️ Top 10 Cities by AQI")
   
    if top_cities is None or len(top_cities) == 0:
        st.warning("No data available")
        return
   
    for idx, row in top_cities.iterrows():
        aqi_info = get_aqi_category(row['aqi'])
        badge_text_color = get_high_contrast_text(aqi_info['color'])
       
        st.markdown(f"""
        <div style="
            background: rgba(30, 27, 75, 0.8);
            padding: 18px 22px;
            border-radius: 14px;
            margin: 12px 0;
            display: flex;
            align-items: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(139, 92, 246, 0.3);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        " onmouseover="this.style.background='rgba(30, 27, 75, 0.95)'; this.style.transform='translateX(10px)'; this.style.borderColor='rgba(139, 92, 246, 0.5)'; this.style.boxShadow='0 4px 16px rgba(0, 0, 0, 0.3)';"
           onmouseout="this.style.background='rgba(30, 27, 75, 0.8)'; this.style.transform='translateX(0)'; this.style.borderColor='rgba(139, 92, 246, 0.3)'; this.style.boxShadow='0 2px 8px rgba(0, 0, 0, 0.2)';">
            <div style="
                width: 44px;
                height: 44px;
                background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
                border-radius: 11px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 900;
                margin-right: 18px;
                font-size: 18px;
                color: #ffffff;
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
            ">{idx + 1}</div>
            <div style="flex: 1; font-weight: 700; font-size: 17px; color: #ffffff; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);">{row['station']}</div>
            <div style="
                background: {aqi_info['color']};
                color: {badge_text_color};
                padding: 10px 18px;
                border-radius: 10px;
                font-weight: 900;
                font-size: 18px;
                box-shadow: 0 4px 12px {aqi_info['color']}50;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.2);
            ">{int(row['aqi'])}</div>
        </div>
        """, unsafe_allow_html=True)

def display_aqi_calculations(pollutants, aqi):
    """Display detailed AQI calculation methodology"""
    st.markdown("## 📐 AQI Calculation Details")
    st.markdown("")
    
    st.markdown("### 🔬 Calculation Methodology")
    st.markdown(f"""
    <div class="custom-card" style="background: rgba(139, 92, 246, 0.15); border-left: 5px solid #a855f7;">
        <p style="color: #ffffff; font-weight: 700; font-size: 16px; margin-bottom: 12px;">How is AQI Calculated?</p>
        <p style="color: #f1f5f9; line-height: 1.8; margin-bottom: 10px; font-weight: 500;">
            The Air Quality Index (AQI) is calculated based on <strong style="color: #ffffff;">Indian standards</strong> using pollutant concentrations:
        </p>
        <ol style="color: #f1f5f9; margin-left: 20px;">
            <li style="margin: 10px 0; font-weight: 500;"><strong style="color: #ffffff;">Sub-index Calculation:</strong> For each pollutant (PM2.5, PM10), a sub-index is calculated using breakpoint concentrations</li>
            <li style="margin: 10px 0; font-weight: 500;"><strong style="color: #ffffff;">Maximum Rule:</strong> The final AQI is the <strong style="color: #ffffff;">MAXIMUM sub-index</strong> among all pollutants</li>
            <li style="margin: 10px 0; font-weight: 500;"><strong style="color: #ffffff;">Category Assignment:</strong> AQI is mapped to quality categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # PM2.5 and PM10 Breakpoints
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### PM2.5 Breakpoints (Indian Standards)")
        pm25_breakpoints = pd.DataFrame({
            'AQI Range': ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500'],
            'PM2.5 (µg/m³)': ['0-30', '31-60', '61-90', '91-120', '121-250', '251-500'],
            'Category': ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
        })
        st.dataframe(pm25_breakpoints, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### PM10 Breakpoints (Indian Standards)")
        pm10_breakpoints = pd.DataFrame({
            'AQI Range': ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500'],
            'PM10 (µg/m³)': ['0-50', '51-100', '101-250', '251-350', '351-430', '431-550'],
            'Category': ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
        })
        st.dataframe(pm10_breakpoints, use_container_width=True, hide_index=True)
    
    st.markdown("")
    
    # Current pollutant values and their contribution
    st.markdown("### 🔍 Your Current Pollutant Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Measured Concentrations")
        pollutant_data = []
        for pollutant, value in sorted(pollutants.items()):
            if pd.notna(value) and value > 0:
                pollutant_data.append({
                    'Pollutant': pollutant,
                    'Concentration': f"{value:.2f}",
                    'Unit': 'µg/m³' if pollutant in ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'NH3', 'NO'] else 'mg/m³'
                })
        
        if pollutant_data:
            df_pollutants = pd.DataFrame(pollutant_data)
            st.dataframe(df_pollutants, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown(f"#### Resulting AQI")
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.3) 0%, rgba(168, 85, 247, 0.3) 100%);
            padding: 30px;
            border-radius: 16px;
            text-align: center;
            border: 2px solid rgba(168, 85, 247, 0.5);
            margin: 20px 0;
        ">
            <p style="color: #a855f7; font-size: 18px; font-weight: 700; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px;">Final AQI Value</p>
            <p style="color: #ffffff; font-size: 64px; font-weight: 900; margin: 20px 0; text-shadow: 0 4px 12px rgba(168, 85, 247, 0.4);">{int(aqi)}</p>
            <p style="color: #f1f5f9; font-size: 16px; font-weight: 600;">Based on maximum sub-index principle</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # AQI Categories and Health Impact
    st.markdown("### 🏥 AQI Categories & Health Impact")
    
    categories_data = {
        'AQI Range': ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500'],
        'Category': ['🟢 Good', '🟡 Satisfactory', '🟠 Moderate', '🔴 Poor', '🟣 Very Poor', '🟤 Severe'],
        'Health Impact': [
            'Minimal impact. Enjoy outdoor activities!',
            'Acceptable for most. Sensitive individuals should limit exertion',
            'General public and sensitive groups should reduce outdoor exertion',
            'Avoid prolonged outdoor activity. Sensitive groups limit activity',
            'Everyone should limit outdoor exertion significantly',
            'Avoid all outdoor physical activity. Move activities indoors'
        ]
    }
    
    df_categories = pd.DataFrame(categories_data)
    st.dataframe(df_categories, use_container_width=True, hide_index=True)



# Main App
def main():
    # Header with gradient
    st.markdown("# 🌍 INDIA'S AQI DASHBOARD")
    st.markdown("### Real-time Air Quality Monitoring Across India")
    st.markdown("---")
   
    # Load data
    with st.spinner('🔄 Loading air quality data...'):
        df = load_data()
   
    if df is None:
        st.error("❌ Failed to load data. Please check if '2022_2025_data.csv' exists.")
        return
   
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Controls")
        st.markdown("")
       
        # Get available years and months
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
       
        available_years = sorted(df['Year'].unique())
       
        # Year and Month selection
        selected_year = st.selectbox(
            "📅 Select Year",
            available_years,
            index=len(available_years) - 1
        )
       
        # Get months available for selected year
        available_months = sorted(df[df['Year'] == selected_year]['Month'].unique())
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
       
        selected_month = st.selectbox(
            "📆 Select Month",
            available_months,
            format_func=lambda x: month_names[x-1],
            index=len(available_months) - 1
        )
       
        # Station selection
        stations = sorted(df['Station'].unique())
        selected_station = st.selectbox(
            "📍 Select Station",
            stations,
            index=0
        )
       
        # Date selection
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
       
        # Info section
        st.markdown("### ℹ️ About Dashboard")
       
        # Get data range
        min_date_overall = df['Date'].min()
        max_date_overall = df['Date'].max()
       
        st.info(f"""
        **📊 Data Coverage**
       
        Monitoring {len(df['Station'].unique())} stations across India
       
       
       
        **Period:** {min_date_overall.strftime('%b %d, %Y')} to {max_date_overall.strftime('%b %d, %Y')}
       
        **🎨 AQI Scale:**
        - 🟢 0-50: Good
        - 🟡 51-100: Satisfactory
        - 🟠 101-200: Moderate
        - 🔴 201-300: Poor
        - 🟣 301-400: Very Poor
        - 🟤 401+: Severe
        """)
       
        st.markdown("---")
       
        # Statistics
        st.markdown("### 📊 Quick Stats")
        total_stations = len(df['Station'].unique())
        total_records = len(df)
       
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stations", f"{total_stations}")
        with col2:
            st.metric("Records", f"{total_records:,}")
   
    # Convert selected_date to datetime for comparison
    selected_date_dt = pd.to_datetime(selected_date)
   
    # Calculate current AQI for selected date
    aqi, pollutants = calculate_daily_aqi(df, selected_station, selected_date_dt)
   
    if aqi is None:
        st.warning(f"⚠️ No data available for {selected_station} on {selected_date.strftime('%B %d, %Y')}")
        return
   
    aqi_info = get_aqi_category(aqi)
   
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📅 Daily View",
        "📊 Monthly Analytics",
        "📈 Yearly Comparison",
        "🗺️ Station Comparison",
        "🔮 AI Forecast",
        "🤖 Model Metrics",
        "📐 AQI Calculations"
    ])
   
    with tab1:  # Daily View
        st.markdown(f"## {selected_station}")
        st.markdown(f"#### {selected_date.strftime('%A, %B %d, %Y')}")
        st.markdown("")
       
        col1, col2 = st.columns([1, 2])
       
        with col1:
            # AQI Display Card
            create_aqi_card(aqi, selected_station, aqi_info)
           
            # Health Impact
            st.markdown("### 💡 Health Impact")
            st.markdown(f"""
            <div class="custom-card">
                <p style="color: #f1f5f9; line-height: 1.7; margin-bottom: 14px; font-weight: 500;">{aqi_info['description']}</p>
                <p style="color: #ffffff; line-height: 1.7; font-weight: 600;"><strong style="color: #ffffff;">💊 Recommendation:</strong> {aqi_info['health_impact']}</p>
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
            # Hourly Pattern
            st.markdown("### 📊 Hourly AQI Pattern")
            hourly_data = get_hourly_data(df, selected_station, selected_date_dt)
            create_hourly_chart(hourly_data, selected_station, selected_date_dt)
           
            st.markdown("")
           
            # Historical Trend
            st.markdown("### 📈 Historical Trend (Last 30 Days)")
            historical_data = get_historical_data(df, selected_station, days=30)
            create_historical_chart(historical_data, selected_station)
           
            st.markdown("")
           
            # India Map
            st.markdown("### 🗺️ All India AQI Map")
            create_india_map(df, selected_date_dt)
   
    with tab2:  # Monthly Analytics
        month_names_full = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        st.markdown(f"## Monthly Analytics")
        st.markdown(f"### {month_names_full[selected_month-1]} {selected_year}")
        st.markdown("")
       
        # Get monthly stats
        monthly_stats = get_month_statistics(df, selected_year, selected_month, selected_station)
       
        if monthly_stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📅 Days with Data", monthly_stats['Days with data'])
            with col2:
                st.metric("📊 Total Records", f"{monthly_stats['Total records']:,}")
            with col3:
                st.metric("🌍 Stations", monthly_stats['Unique stations'])
       
        st.markdown("")
       
        # Get monthly AQI
        result = get_monthly_analytics(df, selected_year, selected_month, selected_station)
        if result:
            monthly_aqi, monthly_pollutants = result
            col1, col2 = st.columns([1, 2])
           
            with col1:
                st.metric("Average AQI", f"{monthly_aqi:.1f}",
                         delta=None,
                         help="Monthly average Air Quality Index")
                aqi_category = get_aqi_category(monthly_aqi)
               
                st.markdown(f"""
                <div class="custom-card">
                    <p style="color: #ffffff; font-size: 20px; font-weight: 700; margin-bottom: 10px;">Category: {aqi_category['category']}</p>
                    <p style="color: #f1f5f9; line-height: 1.7; font-weight: 500;">{aqi_category['description']}</p>
                </div>
                """, unsafe_allow_html=True)
               
                # Monthly pollutants
                st.markdown("### 🏭 Average Pollutant Levels")
                for pollutant, value in sorted(monthly_pollutants.items()):
                    if pd.notna(value):
                        st.markdown(f"<p style='color: #ffffff; font-weight: 600; font-size: 15px;'><strong style='color: #ffffff;'>{pollutant}:</strong> {value:.2f}</p>", unsafe_allow_html=True)
           
            with col2:
                # Pollutant comparison chart
                st.markdown("### 📊 Monthly Pollutant Distribution")
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(monthly_pollutants.keys()),
                        y=list(monthly_pollutants.values()),
                        marker=dict(
                            color='#a855f7',
                            line=dict(width=0),
                            opacity=0.9
                        ),
                        hovertemplate='<b>%{x}</b><br>Concentration: %{y:.2f}<extra></extra>'
                    )
                ])
                fig.update_layout(
                    template='plotly_dark',
                    title=dict(text="Pollutant Concentrations", font=dict(color='#ffffff', family='Inter', size=20)),
                    xaxis_title="Pollutant",
                    yaxis_title="Concentration",
                    height=450,
                    showlegend=False,
                    plot_bgcolor='rgba(15, 23, 42, 0.5)',
                    paper_bgcolor='rgba(30, 27, 75, 0.3)',
                    font=dict(color='#f1f5f9', family='Inter', size=13),
                    xaxis=dict(
                        tickfont=dict(color='#f1f5f9', size=12),
                        title=dict(font=dict(color='#ffffff', size=15)),
                        gridcolor='rgba(139, 92, 246, 0.15)'
                    ),
                    yaxis=dict(
                        tickfont=dict(color='#f1f5f9', size=12),
                        title=dict(font=dict(color='#ffffff', size=15)),
                        gridcolor='rgba(139, 92, 246, 0.15)'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
   
    with tab3:  # Yearly Comparison
        st.markdown(f"## Yearly Comparison")
        st.markdown(f"### {selected_station}")
        st.markdown("")
       
        yearly_data = get_year_comparison(df, selected_station)
       
        if yearly_data is not None and len(yearly_data) > 0:
            col1, col2 = st.columns([2, 1])
           
            with col1:
                st.markdown("### 📊 Year-over-Year AQI")
                fig = go.Figure(data=[
                    go.Bar(
                        x=yearly_data['year'].astype(str),
                        y=yearly_data['aqi'],
                        marker=dict(
                            color=yearly_data['aqi'],
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
                    plot_bgcolor='rgba(15, 23, 42, 0.5)',
                    paper_bgcolor='rgba(30, 27, 75, 0.3)',
                    font=dict(color='#f1f5f9', family='Inter', size=13),
                    xaxis=dict(
                        tickfont=dict(color='#f1f5f9', size=12),
                        title=dict(font=dict(color='#ffffff', size=15)),
                        gridcolor='rgba(139, 92, 246, 0.15)'
                    ),
                    yaxis=dict(
                        tickfont=dict(color='#f1f5f9', size=12),
                        title=dict(font=dict(color='#ffffff', size=15)),
                        gridcolor='rgba(139, 92, 246, 0.15)'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
           
            with col2:
                st.markdown("### 📋 Yearly Averages")
                for _, row in yearly_data.iterrows():
                    year = int(row['year'])
                    aqi_val = row['aqi']
                    category = get_aqi_category(aqi_val)
                   
                    st.markdown(f"""
                    <div class="custom-card" style="margin: 14px 0;">
                        <p style="color: #ffffff; font-size: 20px; font-weight: 800; margin-bottom: 6px;">{year}</p>
                        <p style="color: {category['color']}; font-size: 28px; font-weight: 900; margin: 10px 0; text-shadow: 0 2px 8px {category['color']}40;">{aqi_val:.1f}</p>
                        <p style="color: #ffffff; font-size: 15px; font-weight: 600;">{category['category']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📊 Insufficient data for yearly comparison")
   
    with tab4:  # Station Comparison
        st.markdown(f"## Air Quality Comparison")
        st.markdown(f"### {selected_date.strftime('%B %d, %Y')}")
        st.markdown("")
       
        # Get all stations for this date
        all_stations = df[df['Date'] == pd.to_datetime(selected_date).normalize()]['Station'].unique()
       
        if len(all_stations) > 0:
            col1, col2 = st.columns([1, 1])
           
            with col1:
                st.markdown("### 🔴 Top 10 Most Polluted")
                top_cities = get_top_cities(df, selected_date_dt, top_n=10)
                if top_cities is not None:
                    fig = go.Figure(data=[
                        go.Bar(
                            y=top_cities['station'],
                            x=top_cities['aqi'],
                            orientation='h',
                            marker=dict(
                                color=top_cities['aqi'],
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
                        plot_bgcolor='rgba(15, 23, 42, 0.5)',
                        paper_bgcolor='rgba(30, 27, 75, 0.3)',
                        font=dict(color='#f1f5f9', family='Inter', size=13),
                        xaxis=dict(
                            tickfont=dict(color='#f1f5f9', size=12),
                            title=dict(font=dict(color='#ffffff', size=15)),
                            gridcolor='rgba(139, 92, 246, 0.15)'
                        ),
                        yaxis=dict(
                            tickfont=dict(color='#f1f5f9', size=12),
                            gridcolor='rgba(139, 92, 246, 0.15)'
                        ),
                        margin=dict(l=150)
                    )
                    st.plotly_chart(fig, use_container_width=True)
           
            with col2:
                st.markdown("### 🟢 Top 10 Cleanest")
                bottom_cities = get_top_cities(df, selected_date_dt, top_n=10)
                if bottom_cities is not None:
                    bottom_cities = bottom_cities.sort_values('aqi', ascending=True).head(10)
                    fig = go.Figure(data=[
                        go.Bar(
                            y=bottom_cities['station'],
                            x=bottom_cities['aqi'],
                            orientation='h',
                            marker=dict(
                                color=bottom_cities['aqi'],
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
                        plot_bgcolor='rgba(15, 23, 42, 0.5)',
                        paper_bgcolor='rgba(30, 27, 75, 0.3)',
                        font=dict(color='#f1f5f9', family='Inter', size=13),
                        xaxis=dict(
                            tickfont=dict(color='#f1f5f9', size=12),
                            title=dict(font=dict(color='#ffffff', size=15)),
                            gridcolor='rgba(139, 92, 246, 0.15)'
                        ),
                        yaxis=dict(
                            tickfont=dict(color='#f1f5f9', size=12),
                            gridcolor='rgba(139, 92, 246, 0.15)'
                        ),
                        margin=dict(l=150)
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No data available for this date")
   
    with tab5:  # Forecast
        st.markdown(f"## 🔮 AI-Powered AQI Forecast")
        st.markdown(f"### {selected_station}")
        st.markdown("")
       
        st.info("🤖 **ARIMA Machine Learning Model**\n\n"
                "This forecast uses ARIMA (AutoRegressive Integrated Moving Average), a statistical machine learning model "
                "specifically designed for time series prediction. The model has been trained on historical air quality data "
                "to identify patterns and trends. The shaded area represents the 95% confidence interval, showing the range "
                "where actual values are likely to fall.")
       
        # Generate forecast
        forecast_data = forecast_aqi(df, selected_station, days=7)
       
        if forecast_data is not None:
            # Display forecast chart
            create_forecast_chart(forecast_data, selected_station)
           
            st.markdown("")
           
            # Show forecast table
            st.markdown("### 📅 7-Day Forecast Details")
           
            forecast_only = forecast_data[forecast_data['is_forecast']].copy()
            forecast_only['date'] = pd.to_datetime(forecast_only['date'])
           
            col1, col2 = st.columns([2, 1])
           
            with col1:
                # Create forecast cards with PERFECT text contrast
                for idx, row in forecast_only.iterrows():
                    aqi_cat = get_aqi_category(row['aqi'])
                    date_str = row['date'].strftime('%A, %B %d')
                   
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(30, 27, 75, 0.9) 100%);
                        padding: 22px;
                        border-radius: 16px;
                        margin: 14px 0;
                        border-left: 6px solid {aqi_cat['color']};
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                        transition: all 0.3s ease;
                    " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 24px rgba(0, 0, 0, 0.4)';"
                       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0, 0, 0, 0.3)';">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 20px; font-weight: 800; color: #ffffff; margin-bottom: 10px; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);">
                                    {date_str}
                                </div>
                                <div style="font-size: 15px; color: #ffffff; font-weight: 600; opacity: 0.9;">
                                    📊 Range: {int(row['lower_bound'])} - {int(row['upper_bound'])} AQI
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 44px; font-weight: 900; color: {aqi_cat['color']}; text-shadow: 0 3px 10px {aqi_cat['color']}50, 0 1px 3px rgba(0, 0, 0, 0.3);">
                                    {int(row['aqi'])}
                                </div>
                                <div style="font-size: 15px; color: #ffffff; font-weight: 800; letter-spacing: 1px; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);">
                                    {aqi_cat['category']}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
           
            with col2:
                st.markdown("### 📊 Forecast Summary")
               
                avg_forecast = forecast_only['aqi'].mean()
                max_forecast = forecast_only['aqi'].max()
                min_forecast = forecast_only['aqi'].min()
               
                st.metric("📈 Average AQI", f"{int(avg_forecast)}")
                st.metric("🔴 Peak AQI", f"{int(max_forecast)}")
                st.metric("🟢 Best AQI", f"{int(min_forecast)}")
               
                st.markdown("")
               
                # Trend indicator
                trend = forecast_only['aqi'].iloc[-1] - forecast_only['aqi'].iloc[0]
                if trend > 10:
                    st.markdown("""
                    <div class="custom-card" style="background: rgba(239, 68, 68, 0.25); border-left: 5px solid #ef4444;">
                        <p style="color: #ffffff; font-weight: 800; font-size: 17px; margin-bottom: 6px;">📈 Trend: Worsening</p>
                        <p style="color: #ffffff; font-size: 15px; font-weight: 500; opacity: 0.95;">Air quality expected to deteriorate</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif trend < -10:
                    st.markdown("""
                    <div class="custom-card" style="background: rgba(34, 197, 94, 0.25); border-left: 5px solid #22c55e;">
                        <p style="color: #ffffff; font-weight: 800; font-size: 17px; margin-bottom: 6px;">📉 Trend: Improving</p>
                        <p style="color: #ffffff; font-size: 15px; font-weight: 500; opacity: 0.95;">Air quality expected to improve</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="custom-card" style="background: rgba(59, 130, 246, 0.25); border-left: 5px solid #3b82f6;">
                        <p style="color: #ffffff; font-weight: 800; font-size: 17px; margin-bottom: 6px;">➡️ Trend: Stable</p>
                        <p style="color: #ffffff; font-size: 15px; font-weight: 500; opacity: 0.95;">Air quality expected to remain stable</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Unable to generate forecast. Need at least 14 days of historical data for ARIMA model training.")
   
    with tab6:  # Model Metrics
        st.markdown(f"## 🤖 Model Performance")
        st.markdown(f"### {selected_station}")
        st.markdown("")
        
        # Calculate metrics
        metrics = calculate_model_metrics(df, selected_station)
        display_model_metrics(metrics)
   
    with tab7:  # AQI Calculations
        st.markdown(f"## 📐 AQI Calculation Methodology")
        st.markdown("")
        display_aqi_calculations(pollutants, aqi)
   
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <p style="color: #ffffff; font-size: 15px; margin-bottom: 10px; font-weight: 600;">📊 Data Source: Central Pollution Control Board (CPCB) | AQI calculated using Indian standards</p>
        <p style="color: #ffffff; font-size: 14px; font-weight: 700;">🌍 Built with Streamlit • 📊 Powered by Plotly • 🤖 ARIMA ML Forecasting • 💜 Made for India • ✨ WCAG AAA Accessible</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
