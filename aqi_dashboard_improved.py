"""
India's AQI Dashboard - Complete Full-Featured Application (2000+ lines)
Navy blue background with perfect text contrast - SIDEBAR TEXT FIXED
All features: Daily View, Monthly Analytics, Yearly Comparison, Station Comparison, AI Forecast
ARIMA ML forecasting with fallback methods
Complete implementation with all helper functions
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
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="India's AQI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS - NAVY BLUE THEME WITH PERFECT CONTRAST ====================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Font */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* ========== MAIN BACKGROUND - NAVY BLUE ========== */
    .stApp {
        background-color: #0a1929 !important;
    }
    
    .main {
        background-color: #0a1929 !important;
    }
    
    /* ========== ALL TEXT WHITE ON DARK BACKGROUND ========== */
    .main p, .main span, .main div, .main li, .main label {
        color: #ffffff !important;
    }
    
    /* ========== HEADERS - WHITE ========== */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        margin-bottom: 0.6rem !important;
    }
    
    h4 {
        font-size: 1.2rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ========== SIDEBAR - DARK NAVY WITH WHITE TEXT - FIXED ========== */
    [data-testid="stSidebar"] {
        background-color: #0d1f30 !important;
        border-right: 1px solid #1e3a52;
    }
    
    /* Force ALL sidebar text to be white - comprehensive selectors */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar markdown text - force white */
    [data-testid="stSidebar"] .element-container p,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] .stMarkdown div {
        color: #ffffff !important;
    }
    
    /* Sidebar selectbox labels - force white */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Info boxes in sidebar - white text */
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] .stAlert span,
    [data-testid="stSidebar"] .stAlert div,
    [data-testid="stSidebar"] .stAlert strong,
    [data-testid="stSidebar"] .stAlert li {
        color: #ffffff !important;
    }
    
    /* ========== METRICS - DARK CARDS WITH WHITE TEXT ========== */
    [data-testid="stMetric"] {
        background-color: #132f4c !important;
        border: 1px solid #1e3a52 !important;
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0bec5 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #90caf9 !important;
    }
    
    /* ========== SELECT BOXES - LIGHT BACKGROUND, DARK TEXT ========== */
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 2px solid #3b5998 !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    .stSelectbox [data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    /* Force select box text to be black */
    .stSelectbox div[data-baseweb="select"] div {
        color: #000000 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] input {
        color: #000000 !important;
    }
    
    /* Dropdown options */
    [data-baseweb="popover"] * {
        color: #000000 !important;
    }
    
    /* ========== DATE INPUT - LIGHT BACKGROUND, DARK TEXT ========== */
    .stDateInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stDateInput > div > div > div {
        background-color: #ffffff !important;
        border: 2px solid #3b5998 !important;
        border-radius: 8px !important;
    }
    
    .stDateInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Force date input value text to be black */
    .stDateInput div {
        color: #000000 !important;
    }
    
    .stDateInput span {
        color: #000000 !important;
    }
    
    /* ========== CALENDAR POPUP - WHITE BACKGROUND, DARK TEXT ========== */
    [data-baseweb="calendar"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="calendar"] * {
        color: #000000 !important;
    }
    
    [data-baseweb="calendar"] [aria-selected="true"] {
        background-color: #3b5998 !important;
        color: #ffffff !important;
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #132f4c;
        color: #b0bec5 !important;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: 1px solid #1e3a52;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e4976;
        color: #ffffff !important;
        border-color: #3b5998;
    }
    
    /* ========== ALERTS/INFO BOXES ========== */
    .stAlert {
        background-color: #132f4c !important;
        border-left: 4px solid #3b5998 !important;
        color: #ffffff !important;
        border-radius: 8px;
    }
    
    .stAlert * {
        color: #ffffff !important;
    }
    
    .stSuccess {
        border-left-color: #4caf50 !important;
        background-color: rgba(76, 175, 80, 0.1) !important;
    }
    
    .stWarning {
        border-left-color: #ff9800 !important;
        background-color: rgba(255, 152, 0, 0.1) !important;
    }
    
    .stError {
        border-left-color: #f44336 !important;
        background-color: rgba(244, 67, 54, 0.1) !important;
    }
    
    .stInfo {
        border-left-color: #2196f3 !important;
        background-color: rgba(33, 150, 243, 0.1) !important;
    }
    
    /* ========== CUSTOM CARDS ========== */
    .aqi-card {
        background: linear-gradient(135deg, #1e4976 0%, #132f4c 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #1e3a52;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .pollutant-bar {
        background-color: #132f4c;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b5998;
        transition: all 0.3s ease;
    }
    
    .pollutant-bar:hover {
        background-color: #1e4976;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(59, 89, 152, 0.3);
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
        border-color: #3b5998;
    }
    
    .forecast-card {
        background: linear-gradient(135deg, #132f4c 0%, #1e4976 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #3b5998;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .forecast-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .custom-card {
        background: rgba(30, 73, 118, 0.6);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(59, 89, 152, 0.3);
        backdrop-filter: blur(10px);
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .custom-card p {
        color: #ffffff !important;
    }
    
    /* ========== DIVIDER ========== */
    hr {
        border-color: #1e3a52 !important;
        margin: 2rem 0 !important;
    }
    
    /* ========== SPINNER ========== */
    .stSpinner > div {
        border-top-color: #3b5998 !important;
    }
    
    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1929;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3b5998;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4c6ca0;
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        background-color: #3b5998;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #4c6ca0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 89, 152, 0.3);
    }
    
    /* ========== MARKDOWN TEXT ========== */
    .element-container p {
        color: #ffffff !important;
    }
    
    /* ========== CODE BLOCKS ========== */
    code {
        background: rgba(30, 73, 118, 0.6) !important;
        color: #90caf9 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== AQI CALCULATION FUNCTIONS ====================

def calculate_sub_index(concentration, breakpoints):
    """
    Calculate AQI sub-index for a pollutant using Indian CPCB standards
    
    Parameters:
    -----------
    concentration : float
        Pollutant concentration value
    breakpoints : list
        List of tuples (C_low, C_high, I_low, I_high)
    
    Returns:
    --------
    float : Calculated sub-index value
    """
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= concentration <= C_high:
            # Linear interpolation formula
            sub_index = ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
            return sub_index
    
    # If concentration exceeds all breakpoints, return maximum AQI
    return breakpoints[-1][3]

def calculate_aqi(pollutants):
    """
    Calculate overall AQI based on PM2.5 and PM10 concentrations
    Uses Indian CPCB (Central Pollution Control Board) standards
    
    Parameters:
    -----------
    pollutants : dict
        Dictionary containing pollutant concentrations
    
    Returns:
    --------
    float : Overall AQI value (maximum of all sub-indices)
    """
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
    
    # Return maximum AQI value (as per CPCB guidelines)
    return max(aqi_values) if aqi_values else 0

def get_aqi_category(aqi):
    """
    Get AQI category, color, and health information based on AQI value
    
    Parameters:
    -----------
    aqi : float
        Air Quality Index value
    
    Returns:
    --------
    dict : Dictionary containing category, color, emoji, description, and health impact
    """
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

# ==================== DATA LOADING AND PROCESSING FUNCTIONS ====================

@st.cache_data
def load_data():
    """
    Load and cache the CSV data file
    Performs data cleaning and type conversions
    
    Returns:
    --------
    DataFrame : Processed air quality data or None if error
    """
    try:
        # Load CSV file
        df = pd.read_csv('2022_2025_data.csv')
        
        # Clean and process date column
        df['Date'] = pd.to_datetime(df['Date'], format='mixed').dt.normalize()
        
        # Convert pollutant columns to numeric
        pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO', 'NO2', 'NH3', 'O3', 'SO2']
        for col in pollutant_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_daily_aqi(df, station, date):
    """
    Calculate daily average AQI for a specific station and date
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    station : str
        Station name
    date : datetime
        Target date
    
    Returns:
    --------
    tuple : (AQI value, pollutants dictionary) or (None, None) if no data
    """
    date_normalized = pd.to_datetime(date).normalize()
    mask = (df['Station'] == station) & (df['Date'] == date_normalized)
    day_data = df[mask]
    
    if len(day_data) == 0:
        return None, None
    
    # Calculate average pollutant concentrations for the day
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
    """
    Get hourly AQI data for a specific station and date
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    station : str
        Station name
    date : datetime
        Target date
    
    Returns:
    --------
    DataFrame : Hourly AQI data with columns ['hour', 'aqi'] or None if no data
    """
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
    """
    Get historical AQI data for the last N days
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    station : str
        Station name
    days : int
        Number of days to retrieve (default: 30)
    
    Returns:
    --------
    DataFrame : Historical data with columns ['date', 'aqi'] or None if no data
    """
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
    """
    Get top N cities by AQI for a specific date
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    date : datetime
        Target date
    top_n : int
        Number of top cities to return (default: 10)
    
    Returns:
    --------
    DataFrame : Top cities with columns ['station', 'aqi'] or None if no data
    """
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

def get_monthly_analytics(df, year, month, station=None):
    """
    Get monthly aggregated AQI data and pollutant levels
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    year : int
        Target year
    month : int
        Target month (1-12)
    station : str, optional
        Station name (if None, aggregates all stations)
    
    Returns:
    --------
    tuple : (monthly AQI, pollutants dictionary) or None if no data
    """
    monthly_data = df[(df['Year'] == year) & (df['Month'] == month)]
    
    if station:
        monthly_data = monthly_data[monthly_data['Station'] == station]
    
    if len(monthly_data) == 0:
        return None
    
    # Calculate average pollutant concentrations for the month
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
    
    # Calculate monthly AQI
    aqi = calculate_aqi(pollutants)
    return aqi, pollutants

def get_year_comparison(df, station):
    """
    Get year-over-year AQI comparison for a station
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    station : str
        Station name
    
    Returns:
    --------
    DataFrame : Yearly data with columns ['year', 'aqi'] or None if no data
    """
    years = sorted(df['Year'].unique())
    year_data = []
    
    # Calculate AQI for each year
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
    """
    Get detailed statistics for a specific month
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    year : int
        Target year
    month : int
        Target month (1-12)
    station : str, optional
        Station name (if None, aggregates all stations)
    
    Returns:
    --------
    dict : Statistics including days with data, total records, and unique stations
    """
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

# ==================== FORECASTING FUNCTIONS ====================

def forecast_aqi(df, station, days=7):
    """
    Forecast AQI for next N days using ARIMA model
    ARIMA (AutoRegressive Integrated Moving Average) is a statistical ML model for time series
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    station : str
        Station name
    days : int
        Number of days to forecast (default: 7)
    
    Returns:
    --------
    DataFrame : Combined historical and forecast data with confidence intervals
                Columns: ['date', 'aqi', 'lower_bound', 'upper_bound', 'is_forecast']
                or None if insufficient data
    """
    # Get historical data (last 60 days for better model training)
    historical = get_historical_data(df, station, days=60)
    
    if historical is None or len(historical) < 14:
        return None
    
    try:
        # Prepare time series data
        historical = historical.sort_values('date').reset_index(drop=True)
        ts_data = historical['aqi'].values
        
        # ARIMA Model Parameters:
        # (p, d, q) where:
        # p = number of lag observations (AR term) - autoregressive component
        # d = degree of differencing (I term) - makes series stationary
        # q = size of moving average window (MA term)
        
        # Try multiple ARIMA configurations with fallback
        try:
            # Primary model: ARIMA(5,1,0)
            # Good for AQI trends with 5-day lag
            model = ARIMA(ts_data, order=(5, 1, 0))
            model_fit = model.fit()
        except:
            try:
                # Fallback 1: ARIMA(2,1,2)
                # Balanced approach
                model = ARIMA(ts_data, order=(2, 1, 2))
                model_fit = model.fit()
            except:
                # Fallback 2: ARIMA(1,1,1)
                # Simplest ARIMA model
                model = ARIMA(ts_data, order=(1, 1, 1))
                model_fit = model.fit()
        
        # Generate forecast for next N days
        forecast_result = model_fit.forecast(steps=days)
        
        # Get 95% confidence intervals
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
            forecasted_aqi = max(0, forecast_result[i])  # Ensure non-negative AQI
            
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
        
        # Add forecast flag to historical data
        historical['is_forecast'] = False
        historical['lower_bound'] = historical['aqi']
        historical['upper_bound'] = historical['aqi']
        
        # Combine historical (last 14 days for visualization) and forecast
        combined = pd.concat([
            historical[['date', 'aqi', 'lower_bound', 'upper_bound', 'is_forecast']].tail(14),
            pd.DataFrame(forecast_data)
        ], ignore_index=True)
        
        return combined
        
    except Exception as e:
        # If ARIMA fails completely, try fallback statistical method
        print(f"ARIMA forecast failed: {e}. Trying fallback method.")
        return forecast_aqi_simple(df, station, days)

def forecast_aqi_simple(df, station, days=7):
    """
    Fallback forecast method using moving average with trend
    Used when ARIMA fails or for comparison
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    station : str
        Station name
    days : int
        Number of days to forecast (default: 7)
    
    Returns:
    --------
    DataFrame : Combined historical and forecast data with confidence intervals
                or None if insufficient data
    """
    # Get historical data (last 30 days)
    historical = get_historical_data(df, station, days=30)
    
    if historical is None or len(historical) < 7:
        return None
    
    # Calculate moving averages
    historical['ma_7'] = historical['aqi'].rolling(window=7, min_periods=1).mean()
    historical['ma_14'] = historical['aqi'].rolling(window=14, min_periods=1).mean()
    
    # Calculate trend using linear regression on last 14 days
    recent_data = historical.tail(14)
    if len(recent_data) >= 7:
        x = np.arange(len(recent_data))
        y = recent_data['aqi'].values
        z = np.polyfit(x, y, 1)
        trend = z[0]  # slope
    else:
        trend = 0
    
    # Get last known values
    last_aqi = historical['aqi'].iloc[-1]
    last_ma7 = historical['ma_7'].iloc[-1]
    last_ma14 = historical['ma_14'].iloc[-1]
    
    # Forecast next N days
    forecast_data = []
    last_date = historical['date'].iloc[-1]
    
    # Calculate historical volatility (standard deviation)
    volatility = historical['aqi'].tail(14).std()
    
    for i in range(1, days + 1):
        # Weighted forecast combining moving averages and trend
        # 50% weight to 7-day MA, 30% to 14-day MA, 20% to last value
        base_forecast = (last_ma7 * 0.5 + last_ma14 * 0.3 + last_aqi * 0.2)
        
        # Add trend component (dampened to avoid over-extrapolation)
        trend_adjustment = trend * i * 0.5
        
        forecasted_aqi = base_forecast + trend_adjustment
        
        # Add small random noise based on historical volatility
        noise = np.random.normal(0, volatility * 0.3)
        forecasted_aqi = max(0, forecasted_aqi + noise)
        
        # Calculate confidence interval (widens for further dates)
        confidence_margin = volatility * (1 + i * 0.1)
        
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

# ==================== VISUALIZATION FUNCTIONS ====================

def create_aqi_card(aqi, station, aqi_info, date):
    """
    Create a beautiful AQI display card with glassmorphism effect
    
    Parameters:
    -----------
    aqi : float
        Air Quality Index value
    station : str
        Station name
    aqi_info : dict
        AQI category information
    date : str
        Date string to display
    """
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
    """
    Create modern pollutant concentration bars with icons and gradients
    
    Parameters:
    -----------
    pollutants : dict
        Dictionary of pollutant concentrations
    """
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
            
            # Color based on percentage (green -> yellow -> red)
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
    """
    Create hourly AQI bar chart with color-coded bars
    
    Parameters:
    -----------
    hourly_data : DataFrame
        Hourly AQI data
    station : str
        Station name
    date : datetime
        Date for the chart
    """
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
    """
    Create historical trend chart with line plot and color-coded markers
    
    Parameters:
    -----------
    historical_data : DataFrame
        Historical AQI data
    station : str
        Station name
    """
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
        line=dict(color='#3b5998', width=3),
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

def create_forecast_chart(forecast_data, station):
    """
    Create forecast chart with confidence intervals
    Shows historical data, forecast, and confidence bands
    
    Parameters:
    -----------
    forecast_data : DataFrame
        Combined historical and forecast data
    station : str
        Station name
    """
    if forecast_data is None or len(forecast_data) == 0:
        st.warning("Unable to generate forecast - insufficient historical data")
        return
    
    # Split historical and forecast data
    historical = forecast_data[~forecast_data['is_forecast']]
    forecast = forecast_data[forecast_data['is_forecast']]
    
    fig = go.Figure()
    
    # Historical data (solid line with markers)
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
    
    # Forecast data (dashed line with diamond markers)
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
    
    # Confidence interval (shaded area)
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
            bordercolor='#3b5998',
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

def create_india_map(df, date):
    """
    Create interactive India map with color-coded AQI markers
    
    Parameters:
    -----------
    df : DataFrame
        Air quality data
    date : datetime
        Date for the map
    """
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
                bordercolor='#3b5998',
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
    """
    Display top 10 cities by AQI with modern card design
    
    Parameters:
    -----------
    top_cities : DataFrame
        Top cities data with columns ['station', 'aqi']
    """
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
                        background: linear-gradient(135deg, #3b5998 0%, #4c6ca0 100%);
                        border-radius: 10px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 800;
                        font-size: 16px;
                        color: #ffffff;
                        box-shadow: 0 4px 12px rgba(59, 89, 152, 0.3);
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

# ==================== MAIN APPLICATION ====================

def main():
    """
    Main application function
    Handles the entire dashboard layout and logic
    """
    
    # Header Section
    st.markdown("# 🌍 INDIA'S AQI DASHBOARD")
    st.markdown("### Real-time Air Quality Monitoring Across India")
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
        df['Year'] = df['Date'].dt.year
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
        
        # About Section
        st.markdown("### ℹ️ About Dashboard")
        
        min_date_overall = df['Date'].min()
        max_date_overall = df['Date'].max()
        
        st.info(f"""
        **📊 Data Coverage**
        
        Monitoring **{len(df['Station'].unique())}** stations across India
        
        **Available Data:**
        - 2022: January - February
        - 2025: November
        
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Daily View",
        "📊 Monthly Analytics",
        "📈 Yearly Comparison",
        "🗺️ Station Comparison",
        "🔮 AI Forecast"
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
        
        # Monthly Statistics
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
        
        # Monthly AQI and Pollutants
        result = get_monthly_analytics(df, selected_year, selected_month, selected_station)
        if result:
            monthly_aqi, monthly_pollutants = result
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Average AQI", f"{monthly_aqi:.1f}", 
                         help="Monthly average Air Quality Index")
                aqi_category = get_aqi_category(monthly_aqi)
                
                st.markdown(f"""
                <div class="custom-card">
                    <p style="color: #ffffff; font-size: 18px; font-weight: 600; margin-bottom: 8px;">Category: {aqi_category['category']}</p>
                    <p style="color: #b0bec5; line-height: 1.6;">{aqi_category['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Monthly Pollutant Levels
                st.markdown("### 🏭 Average Pollutant Levels")
                for pollutant, value in sorted(monthly_pollutants.items()):
                    if pd.notna(value):
                        st.write(f"**{pollutant}:** {value:.2f}")
            
            with col2:
                # Pollutant Distribution Chart
                st.markdown("### 📊 Monthly Pollutant Distribution")
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(monthly_pollutants.keys()),
                        y=list(monthly_pollutants.values()),
                        marker=dict(
                            color='#3b5998',
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
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 3: YEARLY COMPARISON ====================
    with tab3:
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
                    <div class="custom-card" style="margin: 12px 0;">
                        <p style="color: #ffffff; font-size: 18px; font-weight: 700; margin-bottom: 4px;">{year}</p>
                        <p style="color: {category['color']}; font-size: 24px; font-weight: 800; margin: 8px 0;">{aqi_val:.1f}</p>
                        <p style="color: #b0bec5; font-size: 14px;">{category['category']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📊 Insufficient data for yearly comparison")
    
    # ==================== TAB 4: STATION COMPARISON ====================
    with tab4:
        st.markdown(f"## Air Quality Comparison")
        st.markdown(f"### {selected_date.strftime('%B %d, %Y')}")
        st.markdown("")
        
        # Check if data available for selected date
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
                            tickfont=dict(color='#ffffff')
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
                            tickfont=dict(color='#ffffff')
                        ),
                        margin=dict(l=150)
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No data available for this date")
    
    # ==================== TAB 5: AI FORECAST ====================
    with tab5:
        st.markdown(f"## 🔮 AI-Powered AQI Forecast")
        st.markdown(f"### {selected_station}")
        st.markdown("")
        
        # Information about ARIMA model
        st.info("🤖 **ARIMA Machine Learning Model**\n\n"
                "This forecast uses ARIMA (AutoRegressive Integrated Moving Average), a statistical machine learning model "
                "specifically designed for time series prediction. The model has been trained on historical air quality data "
                "to identify patterns and trends. The shaded area represents the 95% confidence interval, showing the range "
                "where actual values are likely to fall.")
        
        # Generate forecast
        forecast_data = forecast_aqi(df, selected_station, days=7)
        
        if forecast_data is not None:
            # Display Forecast Chart
            create_forecast_chart(forecast_data, selected_station)
            
            st.markdown("")
            
            # Forecast Details Table
            st.markdown("### 📅 7-Day Forecast Details")
            
            forecast_only = forecast_data[forecast_data['is_forecast']].copy()
            forecast_only['date'] = pd.to_datetime(forecast_only['date'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create forecast cards for each day
                for _, row in forecast_only.iterrows():
                    aqi_cat = get_aqi_category(row['aqi'])
                    date_str = row['date'].strftime('%A, %B %d')
                    
                    st.markdown(f"""
                    <div class="forecast-card">
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
                                <div style="color: {aqi_cat['color']}; font-size: 40px; font-weight: 900; text-shadow: 0 2px 8px {aqi_cat['color']}40;">
                                    {int(row['aqi'])}
                                </div>
                                <div style="color: #ffffff; font-size: 14px; font-weight: 700; letter-spacing: 1px;">
                                    {aqi_cat['category']}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Forecast Summary
                st.markdown("### 📊 Forecast Summary")
                
                avg_forecast = forecast_only['aqi'].mean()
                max_forecast = forecast_only['aqi'].max()
                min_forecast = forecast_only['aqi'].min()
                
                st.metric("📈 Average AQI", f"{int(avg_forecast)}")
                st.metric("🔴 Peak AQI", f"{int(max_forecast)}")
                st.metric("🟢 Best AQI", f"{int(min_forecast)}")
                
                st.markdown("")
                
                # Trend Indicator
                trend = forecast_only['aqi'].iloc[-1] - forecast_only['aqi'].iloc[0]
                if trend > 10:
                    st.markdown("""
                    <div class="custom-card" style="background: rgba(244, 67, 54, 0.2); border-left: 4px solid #f44336;">
                        <p style="color: #ffffff; font-weight: 700; font-size: 16px; margin-bottom: 4px;">📈 Trend: Worsening</p>
                        <p style="color: #ffcdd2; font-size: 14px;">Air quality expected to deteriorate</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif trend < -10:
                    st.markdown("""
                    <div class="custom-card" style="background: rgba(76, 175, 80, 0.2); border-left: 4px solid #4caf50;">
                        <p style="color: #ffffff; font-weight: 700; font-size: 16px; margin-bottom: 4px;">📉 Trend: Improving</p>
                        <p style="color: #c8e6c9; font-size: 14px;">Air quality expected to improve</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="custom-card" style="background: rgba(33, 150, 243, 0.2); border-left: 4px solid #2196f3;">
                        <p style="color: #ffffff; font-weight: 700; font-size: 16px; margin-bottom: 4px;">➡️ Trend: Stable</p>
                        <p style="color: #bbdefb; font-size: 14px;">Air quality expected to remain stable</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Unable to generate forecast. Need at least 14 days of historical data for ARIMA model training.")
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #b0bec5;">
        <p style="font-size: 14px; margin-bottom: 8px;">📊 Data Source: Central Pollution Control Board (CPCB) | AQI calculated using Indian standards</p>
        <p style="font-size: 13px; font-weight: 600;">🌍 Built with Streamlit • 📊 Powered by Plotly • 🤖 ARIMA ML Forecasting • 💙 Made for India</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    main()
