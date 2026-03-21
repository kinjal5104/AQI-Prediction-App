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
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.inspection import permutation_importance, partial_dependence
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
    st.warning("SHAP library not available. Install with: pip install shap")

# For Lime explanations
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

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
    
    .shap-card {
        background: linear-gradient(135deg, #d81b60 0%, #c2185b 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 3px solid #ec407a;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(216, 27, 96, 0.3);
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
    
    .feature-importance-bar {
        background: rgba(38, 166, 154, 0.2);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #26a69a;
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
    
    .feature-importance-table {
        background: rgba(38, 166, 154, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .prediction-explanation {
        background: linear-gradient(135deg, #8e24aa 0%, #6a1b9a 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #ab47bc;
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
        df = pd.read_csv('expanded_2022_2025_data_100days.csv')
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

# ==================== XAI FUNCTIONS ====================

def calculate_feature_importance_permutation(model, X_test, y_test, feature_names):
    """Calculate feature importance using permutation importance"""
    try:
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    except Exception as e:
        print(f"Error calculating permutation importance: {e}")
        return None

def calculate_shap_values(model, X_train, X_test, feature_names, model_type='tree'):
    """Calculate SHAP values for model interpretation"""
    if not SHAP_AVAILABLE:
        return None, None, None
    
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.KernelExplainer(model.predict, X_train[:100])
            shap_values = explainer.shap_values(X_test[:100])
        
        # Calculate global feature importance
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        return explainer, shap_values, shap_importance
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return None, None, None

def create_shap_summary_plot(shap_values, X_test, feature_names):
    """Create SHAP summary plot"""
    if not SHAP_AVAILABLE or shap_values is None:
        return None
    
    try:
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create summary plot
        shap.summary_plot(
            shap_values, 
            X_test,
            feature_names=feature_names,
            show=False,
            plot_type="dot",
            max_display=15
        )
        
        # Customize plot for dark theme
        ax.set_facecolor('#0a1929')
        fig.patch.set_facecolor('#0a1929')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        # Set title color
        ax.title.set_color('white')
        
        return fig
    except Exception as e:
        print(f"Error creating SHAP plot: {e}")
        return None

def create_feature_importance_chart(importance_df, title="Feature Importance"):
    """Create feature importance bar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance_df['importance_mean'][:15],
        y=importance_df['feature'][:15],
        orientation='h',
        marker=dict(
            color=importance_df['importance_mean'][:15],
            colorscale='Viridis',
            line=dict(width=0)
        ),
        error_x=dict(
            type='data',
            array=importance_df['importance_std'][:15],
            visible=True
        ),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<br>Std: %{error_x.array:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#ffffff', size=18)
        ),
        template="plotly_dark",
        height=500,
        plot_bgcolor='#0a1929',
        paper_bgcolor='#0a1929',
        font=dict(color='#ffffff'),
        xaxis=dict(
            title="Importance Score",
            gridcolor='#1e3a52',
            title_font=dict(color='#b0bec5'),
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            title="Features",
            gridcolor='#1e3a52',
            title_font=dict(color='#b0bec5'),
            tickfont=dict(color='#ffffff')
        ),
        margin=dict(l=100, r=30, t=60, b=30)
    )
    
    return fig

def explain_individual_prediction(model, features, feature_names, target_value, explanation_type='simple'):
    """Explain individual prediction"""
    try:
        # Get prediction
        prediction = model.predict(features.reshape(1, -1))[0]
        
        if explanation_type == 'simple':
            # Simple feature contribution explanation
            if hasattr(model, 'coef_'):
                # Linear model
                contributions = model.coef_ * features
                most_important_idx = np.argmax(np.abs(contributions))
                explanation = f"""
                **Predicted AQI:** {prediction:.1f} (Actual: {target_value:.1f})
                
                **Most influential feature:** {feature_names[most_important_idx]}
                - Value: {features[most_important_idx]:.2f}
                - Contribution: {contributions[most_important_idx]:.2f}
                - Coefficient: {model.coef_[most_important_idx]:.4f}
                """
            elif hasattr(model, 'feature_importances_'):
                # Tree-based model
                importances = model.feature_importances_
                most_important_idx = np.argmax(importances)
                explanation = f"""
                **Predicted AQI:** {prediction:.1f} (Actual: {target_value:.1f})
                
                **Most important feature:** {feature_names[most_important_idx]}
                - Value: {features[most_important_idx]:.2f}
                - Feature importance: {importances[most_important_idx]:.4f}
                """
            else:
                explanation = f"**Predicted AQI:** {prediction:.1f} (Actual: {target_value:.1f})"
            
            return explanation
        
    except Exception as e:
        print(f"Error explaining prediction: {e}")
        return f"**Prediction:** {prediction:.1f} (Actual: {target_value:.1f})"

def create_partial_dependence_plot(model, X_train, feature_names, feature_idx):
    """Create partial dependence plot for a feature"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate partial dependence
        from sklearn.inspection import PartialDependenceDisplay
        
        PartialDependenceDisplay.from_estimator(
            model, 
            X_train, 
            features=[feature_idx],
            ax=ax,
            line_kw={"color": "#5c6bc0", "linewidth": 3}
        )
        
        # Customize for dark theme
        ax.set_facecolor('#0a1929')
        fig.patch.set_facecolor('#0a1929')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, alpha=0.2, color='white')
        
        # Update title
        feature_name = feature_names[feature_idx]
        ax.set_title(f'Partial Dependence: {feature_name}', color='white', fontsize=14)
        
        return fig
    except Exception as e:
        print(f"Error creating partial dependence plot: {e}")
        return None

def create_model_interpretation_dashboard(model, X_test, y_test, feature_names, model_name, model_type='tree'):
    """Create comprehensive model interpretation dashboard"""
    
    st.markdown(f"## 🧠 {model_name} - Model Interpretation")
    st.markdown("### Understanding how the model makes predictions")
    
    # Create tabs for different interpretation methods
    int_tab1, int_tab2, int_tab3, int_tab4 = st.tabs([
        "📊 Feature Importance",
        "🔍 SHAP Analysis",
        "🎯 Individual Predictions",
        "📈 Model Insights"
    ])
    
    with int_tab1:
        st.markdown("### Feature Importance Analysis")
        
        # Calculate permutation importance
        with st.spinner("Calculating feature importance..."):
            perm_importance = calculate_feature_importance_permutation(model, X_test, y_test, feature_names)
        
        if perm_importance is not None:
            # Display feature importance chart
            fig = create_feature_importance_chart(perm_importance, f"{model_name} - Permutation Importance")
            st.plotly_chart(fig, use_container_width=True, key=f"feature_imp_{model_name}")
            
            # Display top features table
            st.markdown("#### 🏆 Top 10 Most Important Features")
            top_features = perm_importance.head(10)
            
            for idx, row in top_features.iterrows():
                importance_pct = (row['importance_mean'] / perm_importance['importance_mean'].sum()) * 100
                
                st.markdown(f"""
                <div class="feature-importance-bar">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="color: #ffffff; font-weight: 600; font-size: 14px;">{row['feature']}</div>
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="color: #26a69a; font-weight: 700; font-size: 14px;">{row['importance_mean']:.4f}</div>
                            <div style="
                                background: linear-gradient(90deg, #26a69a 0%, #26a69a {importance_pct}%, #1e3a52 {importance_pct}%, #1e3a52 100%);
                                width: 200px;
                                height: 8px;
                                border-radius: 4px;
                            "></div>
                            <div style="color: #b0bec5; font-size: 12px;">{importance_pct:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Built-in feature importance for tree models
        if hasattr(model, 'feature_importances_'):
            st.markdown("#### 🌳 Built-in Feature Importance (Gini Importance)")
            
            builtin_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            for idx, row in builtin_importance.iterrows():
                st.markdown(f"""
                <div class="custom-card" style="padding: 12px; margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #ffffff;">{row['feature']}</span>
                        <span style="color: #4caf50; font-weight: 700;">{row['importance']:.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with int_tab2:
        st.markdown("### SHAP (SHapley Additive exPlanations) Analysis")
        
        if not SHAP_AVAILABLE:
            st.warning("""
            **SHAP library not installed.** 
            
            Install it with: `pip install shap`
            
            SHAP provides game-theoretic optimal feature attributions for any machine learning model.
            """)
        else:
            with st.spinner("Calculating SHAP values..."):
                # Get a subset of data for SHAP (faster computation)
                X_train_sample = X_test[:100]
                shap_explainer, shap_values, shap_importance = calculate_shap_values(
                    model, X_train_sample, X_test[:50], feature_names, model_type
                )
            
            if shap_explainer is not None and shap_values is not None:
                # SHAP Summary Plot
                st.markdown("#### 📊 SHAP Summary Plot")
                st.markdown("""
                This plot shows:
                - **Y-axis:** Features ranked by importance
                - **X-axis:** SHAP value (impact on prediction)
                - **Color:** Feature value (red=high, blue=low)
                """)
                
                shap_fig = create_shap_summary_plot(shap_values, X_test[:50], feature_names)
                if shap_fig:
                    st.pyplot(shap_fig, use_container_width=True)
                
                # SHAP Feature Importance
                st.markdown("#### 🎯 SHAP Feature Importance")
                if shap_importance is not None:
                    top_shap_features = shap_importance.head(10)
                    
                    for idx, row in top_shap_features.iterrows():
                        st.markdown(f"""
                        <div class="custom-card" style="padding: 12px; margin: 8px 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #ffffff; font-weight: 600;">{row['feature']}</span>
                                <span style="color: #d81b60; font-weight: 700;">{row['importance']:.4f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # SHAP Explanation
                st.markdown("#### 📖 SHAP Interpretation Guide")
                st.markdown("""
                <div class="shap-card">
                    <h4>Understanding SHAP Values:</h4>
                    <ul style="color: #f8bbd9;">
                        <li><strong>Positive SHAP value:</strong> Feature increases the predicted AQI</li>
                        <li><strong>Negative SHAP value:</strong> Feature decreases the predicted AQI</li>
                        <li><strong>Magnitude:</strong> How much the feature affects the prediction</li>
                        <li><strong>Color:</strong> Red = high feature value, Blue = low feature value</li>
                    </ul>
                    <p style="color: #f8bbd9; margin-top: 12px;">
                    <strong>Example:</strong> If PM2.5 has a large positive SHAP value and is red, 
                    it means high PM2.5 levels significantly increase the predicted AQI.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    with int_tab3:
        st.markdown("### Individual Prediction Explanations")
        
        # Select a sample to explain
        sample_idx = st.selectbox(
            "Select a sample to explain:",
            range(min(20, len(X_test))),
            format_func=lambda x: f"Sample {x+1}: Actual AQI = {y_test.iloc[x] if hasattr(y_test, 'iloc') else y_test[x]:.1f}"
        )
        
        if sample_idx < len(X_test):
            features = X_test[sample_idx]
            actual_value = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
            
            # Get explanation
            explanation = explain_individual_prediction(
                model, features, feature_names, actual_value, 'simple'
            )
            
            st.markdown(f"""
            <div class="prediction-explanation">
                <h4 style="color: #ffffff; margin-bottom: 16px;">📋 Prediction Explanation</h4>
                <div style="color: #e1bee7; line-height: 1.6;">
                    {explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature values for the sample
            st.markdown("#### 📊 Feature Values for This Sample")
            
            top_features_idx = []
            if hasattr(model, 'feature_importances_'):
                top_features_idx = np.argsort(model.feature_importances_)[-5:][::-1]
            elif perm_importance is not None:
                top_features_idx = perm_importance.head(5).index
            
            for idx in top_features_idx[:5]:
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                    feature_value = features[idx]
                    
                    # Get feature statistics for context
                    feature_mean = X_test[:, idx].mean()
                    feature_std = X_test[:, idx].std()
                    
                    # Calculate z-score
                    z_score = (feature_value - feature_mean) / feature_std if feature_std > 0 else 0
                    
                    # Determine if value is high or low
                    if z_score > 1:
                        status = "🔴 High"
                        color = "#f44336"
                    elif z_score < -1:
                        status = "🟢 Low"
                        color = "#4caf50"
                    else:
                        status = "🟡 Normal"
                        color = "#ffc107"
                    
                    st.markdown(f"""
                    <div class="custom-card" style="padding: 12px; margin: 8px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="color: #ffffff; font-weight: 600; font-size: 14px;">{feature_name}</div>
                                <div style="color: #b0bec5; font-size: 12px;">Value: {feature_value:.2f} (z-score: {z_score:.2f})</div>
                            </div>
                            <div style="color: {color}; font-weight: 700;">{status}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with int_tab4:
        st.markdown("### Model Insights & Recommendations")
        
        # Get top 3 features
        if perm_importance is not None:
            top_3_features = perm_importance.head(3)['feature'].tolist()
            
            st.markdown(f"""
            <div class="xai-explanation">
                <h4 style="color: #ffffff;">🎯 Key Findings</h4>
                <p style="color: #b2dfdb;">
                The model's predictions are primarily driven by:
                </p>
                <ol style="color: #b2dfdb;">
                    <li><strong>{top_3_features[0] if len(top_3_features) > 0 else 'N/A'}</strong> - Most influential factor</li>
                    <li><strong>{top_3_features[1] if len(top_3_features) > 1 else 'N/A'}</strong> - Secondary contributor</li>
                    <li><strong>{top_3_features[2] if len(top_3_features) > 2 else 'N/A'}</strong> - Tertiary factor</li>
                </ol>
                
                <div style="margin-top: 20px;">
                    <h5 style="color: #ffffff;">📈 Actionable Insights:</h5>
                    <ul style="color: #b2dfdb;">
                        <li>Monitor <strong>{top_3_features[0] if len(top_3_features) > 0 else 'key pollutants'}</strong> closely for AQI prediction</li>
                        <li>Historical AQI values (lag features) are crucial for time series prediction</li>
                        <li>Consider external factors like weather data for improved accuracy</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Model-specific insights
        st.markdown("#### 🤖 Model-Specific Characteristics")
        
        if model_name == "Random Forest":
            st.markdown("""
            <div class="custom-card">
                <h5 style="color: #ffffff;">🌳 Random Forest Insights:</h5>
                <ul style="color: #bbdefb;">
                    <li><strong>Ensemble Method:</strong> Combines multiple decision trees</li>
                    <li><strong>Robust to Overfitting:</strong> Uses bagging and feature randomness</li>
                    <li><strong>Feature Importance:</strong> Based on Gini impurity reduction</li>
                    <li><strong>Interpretability:</strong> Can analyze individual tree decisions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif model_name == "Decision Tree":
            st.markdown("""
            <div class="custom-card">
                <h5 style="color: #ffffff;">🌿 Decision Tree Insights:</h5>
                <ul style="color: #c8e6c9;">
                    <li><strong>Rule-Based:</strong> Makes decisions via if-else rules</li>
                    <li><strong>Visualizable:</strong> Can view the entire decision path</li>
                    <li><strong>Splitting Criteria:</strong> Uses information gain or Gini index</li>
                    <li><strong>Prone to Overfitting:</strong> Needs depth limitation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif model_name == "Linear Regression":
            st.markdown("""
            <div class="custom-card">
                <h5 style="color: #ffffff;">📐 Linear Regression Insights:</h5>
                <ul style="color: #ffecb3;">
                    <li><strong>Linear Relationship:</strong> Assumes linear feature effects</li>
                    <li><strong>Coefficient Interpretation:</strong> Direct feature impact quantification</li>
                    <li><strong>Global Explanation:</strong> Same coefficients apply to all predictions</li>
                    <li><strong>Assumptions:</strong> Linearity, independence, normality, homoscedasticity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ==================== ARIMA XAI FUNCTIONS ====================

def explain_arima_model(model_fit, series):
    """Explain ARIMA model components and predictions"""
    
    st.markdown("## 🧠 ARIMA Model Interpretation")
    st.markdown("### Understanding Time Series Components")
    
    # Model summary
    st.markdown("#### 📋 Model Summary")
    
    # Extract key parameters
    params = model_fit.params
    pvalues = model_fit.pvalues
    
    # Create parameter table
    param_df = pd.DataFrame({
        'Parameter': params.index,
        'Value': params.values,
        'P-value': pvalues.values,
        'Significant': pvalues.values < 0.05
    })
    
    # Display parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ar_params = param_df[param_df['Parameter'].str.startswith('ar')]
        if len(ar_params) > 0:
            st.markdown("**AR Coefficients:**")
            for _, row in ar_params.iterrows():
                sig_color = "#4caf50" if row['Significant'] else "#f44336"
                st.markdown(f"""
                <div style="color: {sig_color}; font-weight: 600;">
                    {row['Parameter']}: {row['Value']:.4f} (p={row['P-value']:.4f})
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        ma_params = param_df[param_df['Parameter'].str.startswith('ma')]
        if len(ma_params) > 0:
            st.markdown("**MA Coefficients:**")
            for _, row in ma_params.iterrows():
                sig_color = "#4caf50" if row['Significant'] else "#f44336"
                st.markdown(f"""
                <div style="color: {sig_color}; font-weight: 600;">
                    {row['Parameter']}: {row['Value']:.4f} (p={row['P-value']:.4f})
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        # Model diagnostics
        st.markdown("**Model Diagnostics:**")
        st.markdown(f"AIC: `{model_fit.aic:.2f}`")
        st.markdown(f"BIC: `{model_fit.bic:.2f}`")
        st.markdown(f"HQIC: `{model_fit.hqic:.2f}`")
    
    # Time Series Decomposition
    st.markdown("---")
    st.markdown("#### 📊 Time Series Decomposition")
    
    try:
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(series.dropna(), model='additive', period=7)
        
        # Create decomposition plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        
        # Original series
        ax1.plot(series.index, series.values, color='#5c6bc0', linewidth=2)
        ax1.set_title('Original Series', color='white', fontsize=12)
        ax1.set_facecolor('#0a1929')
        ax1.tick_params(colors='white')
        
        # Trend component
        ax2.plot(series.index[:len(decomposition.trend)], decomposition.trend, 
                color='#4caf50', linewidth=2)
        ax2.set_title('Trend Component', color='white', fontsize=12)
        ax2.set_facecolor('#0a1929')
        ax2.tick_params(colors='white')
        
        # Seasonal component
        ax3.plot(series.index[:len(decomposition.seasonal)], decomposition.seasonal, 
                color='#ff9800', linewidth=2)
        ax3.set_title('Seasonal Component', color='white', fontsize=12)
        ax3.set_facecolor('#0a1929')
        ax3.tick_params(colors='white')
        
        # Residual component
        ax4.plot(series.index[:len(decomposition.resid)], decomposition.resid, 
                color='#f44336', linewidth=2)
        ax4.set_title('Residual Component', color='white', fontsize=12)
        ax4.set_facecolor('#0a1929')
        ax4.tick_params(colors='white')
        
        # Set background color
        fig.patch.set_facecolor('#0a1929')
        
        st.pyplot(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("""
        <div class="xai-explanation">
            <h4 style="color: #ffffff;">📖 Decomposition Interpretation:</h4>
            <ul style="color: #b2dfdb;">
                <li><strong>Trend:</strong> Long-term progression of AQI (increasing/decreasing)</li>
                <li><strong>Seasonal:</strong> Regular pattern repeating each period (weekly/monthly)</li>
                <li><strong>Residual:</strong> Random noise after removing trend and seasonality</li>
                <li><strong>Model Fit:</strong> Small residuals indicate good model fit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.warning(f"Could not perform decomposition: {e}")
    
    # ACF and PACF Analysis
    st.markdown("---")
    st.markdown("#### 📈 Autocorrelation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ACF Plot
        fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
        plot_acf(series.dropna(), ax=ax_acf, lags=20, color='#5c6bc0')
        ax_acf.set_title('Autocorrelation Function (ACF)', color='white', fontsize=12)
        ax_acf.set_facecolor('#0a1929')
        ax_acf.tick_params(colors='white')
        ax_acf.xaxis.label.set_color('white')
        ax_acf.yaxis.label.set_color('white')
        fig_acf.patch.set_facecolor('#0a1929')
        st.pyplot(fig_acf, use_container_width=True)
    
    with col2:
        # PACF Plot
        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
        plot_pacf(series.dropna(), ax=ax_pacf, lags=20, color='#4caf50')
        ax_pacf.set_title('Partial Autocorrelation Function (PACF)', color='white', fontsize=12)
        ax_pacf.set_facecolor('#0a1929')
        ax_pacf.tick_params(colors='white')
        ax_pacf.xaxis.label.set_color('white')
        ax_pacf.yaxis.label.set_color('white')
        fig_pacf.patch.set_facecolor('#0a1929')
        st.pyplot(fig_pacf, use_container_width=True)
    
    # Interpretation of ACF/PACF
    st.markdown("""
    <div class="custom-card">
        <h5 style="color: #ffffff;">🔍 ACF/PACF Interpretation Guide:</h5>
        <ul style="color: #bbdefb;">
            <li><strong>ACF:</strong> Shows correlation with lagged values - helps determine MA order (q)</li>
            <li><strong>PACF:</strong> Shows direct correlation with lagged values - helps determine AR order (p)</li>
            <li><strong>Significant Lags:</strong> Points outside confidence bands (blue area)</li>
            <li><strong>Decay Pattern:</strong> Slow decay suggests differencing needed (determines d)</li>
        </ul>
        <p style="color: #90caf9; margin-top: 10px;">
        <strong>For this model:</strong> Significant spikes at specific lags indicate appropriate AR and MA terms.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Residuals Analysis
    st.markdown("---")
    st.markdown("#### 📊 Model Residuals Analysis")
    
    residuals = model_fit.resid
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals plot
        fig_res, ax_res = plt.subplots(figsize=(10, 4))
        ax_res.plot(residuals.index, residuals.values, color='#f44336', linewidth=1)
        ax_res.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax_res.set_title('Model Residuals Over Time', color='white', fontsize=12)
        ax_res.set_facecolor('#0a1929')
        ax_res.tick_params(colors='white')
        ax_res.xaxis.label.set_color('white')
        ax_res.yaxis.label.set_color('white')
        fig_res.patch.set_facecolor('#0a1929')
        st.pyplot(fig_res, use_container_width=True)
    
    with col2:
        # Residuals histogram
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        ax_hist.hist(residuals.dropna(), bins=30, color='#5c6bc0', alpha=0.7, edgecolor='white')
        ax_hist.axvline(x=residuals.mean(), color='#4caf50', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
        ax_hist.set_title('Residuals Distribution', color='white', fontsize=12)
        ax_hist.set_facecolor('#0a1929')
        ax_hist.tick_params(colors='white')
        ax_hist.xaxis.label.set_color('white')
        ax_hist.yaxis.label.set_color('white')
        ax_hist.legend(facecolor='#0a1929', edgecolor='white', labelcolor='white')
        fig_hist.patch.set_facecolor('#0a1929')
        st.pyplot(fig_hist, use_container_width=True)
    
    # Residuals statistics
    st.markdown("#### 📈 Residuals Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{residuals.mean():.4f}")
    
    with col2:
        st.metric("Std Dev", f"{residuals.std():.4f}")
    
    with col3:
        # Check for normality using Jarque-Bera test approximation
        skewness = residuals.skew()
        kurtosis = residuals.kurtosis()
        st.metric("Skewness", f"{skewness:.4f}")
    
    with col4:
        st.metric("Kurtosis", f"{kurtosis:.4f}")
    
    # Residuals interpretation
    st.markdown("""
    <div class="xai-explanation">
        <h4 style="color: #ffffff;">✅ Residuals Quality Check:</h4>
        <ul style="color: #b2dfdb;">
            <li><strong>Zero Mean:</strong> Residuals should center around zero (good fit)</li>
            <li><strong>Constant Variance:</strong> No patterns in residuals over time</li>
            <li><strong>Normality:</strong> Residuals should be normally distributed (bell curve)</li>
            <li><strong>No Autocorrelation:</strong> Residuals should not correlate with themselves</li>
        </ul>
        <p style="color: #80cbc4; margin-top: 12px;">
        <strong>Diagnosis:</strong> Well-behaved residuals indicate the ARIMA model has captured all predictable patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

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

def create_hourly_chart(hourly_data, station, date, key_suffix=""):
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
    
    st.plotly_chart(fig, use_container_width=True, key=f"hourly_chart_{station}_{date.strftime('%Y%m%d')}{key_suffix}")

def create_historical_chart(historical_data, station, key_suffix=""):
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
    
    st.plotly_chart(fig, use_container_width=True, key=f"historical_chart_{station}{key_suffix}")

def create_india_map(df, date, key_suffix=""):
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
    
    st.plotly_chart(fig, use_container_width=True, key=f"india_map_{date.strftime('%Y%m%d')}{key_suffix}")

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

def create_forecast_chart(forecast_data, station, key_suffix=""):
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
    
    st.plotly_chart(fig, use_container_width=True, key=f"forecast_chart_{station}{key_suffix}")

# ==================== IMPROVED ARIMA WITH HYPERPARAMETER TUNING ====================

def find_best_arima_order(series, max_p=3, max_d=2, max_q=3):
    """Find best ARIMA order using AIC criterion"""
    best_aic = np.inf
    best_order = (1, 1, 1)
    
    # Try different ARIMA orders
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue  # Skip ARIMA(0,d,0)
                
                try:
                    model = ARIMA(series.dropna(), order=(p, d, q))
                    model_fit = model.fit()
                    
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        
                except Exception as e:
                    continue
    
    return best_order, best_aic

def evaluate_arima_model(train_data, test_data, order=(1, 1, 1)):
    """Evaluate ARIMA model performance"""
    try:
        # Fit model on training data
        model = ARIMA(train_data.dropna(), order=order)
        model_fit = model.fit()
        
        # Make predictions
        forecast = model_fit.forecast(steps=len(test_data))
        
        # Calculate metrics
        mae = mean_absolute_error(test_data.values, forecast)
        rmse = np.sqrt(mean_squared_error(test_data.values, forecast))
        r2 = r2_score(test_data.values, forecast)
        mape = np.mean(np.abs((test_data.values - forecast) / (test_data.values + 1e-5))) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'AIC': model_fit.aic,
            'BIC': model_fit.bic,
            'model': model_fit
        }
    except Exception as e:
        print(f"Error evaluating ARIMA model: {e}")
        return None

# ==================== ML FUNCTIONS (FOR COMPARISON ONLY) ====================

def prepare_time_series_ml_data(df, station, lag_days=7):
    """Prepare time series data for ML models with lag features"""
    station_data = df[df['Station'] == station].copy()
    
    if len(station_data) < 100:
        return None, None, None, None, None
    
    # Sort by date
    station_data = station_data.sort_values('Date')
    
    # Create lag features
    for lag in range(1, lag_days + 1):
        station_data[f'AQI_lag_{lag}'] = station_data['AQI'].shift(lag)
    
    # Create rolling statistics
    station_data['AQI_rolling_7'] = station_data['AQI'].rolling(window=7, min_periods=1).mean()
    station_data['AQI_rolling_30'] = station_data['AQI'].rolling(window=30, min_periods=1).mean()
    
    # Add day of week dummies
    station_data['DayOfWeek_sin'] = np.sin(2 * np.pi * station_data['DayOfWeek'] / 7)
    station_data['DayOfWeek_cos'] = np.cos(2 * np.pi * station_data['DayOfWeek'] / 7)
    
    # Add month dummies
    station_data['Month_sin'] = np.sin(2 * np.pi * station_data['Month'] / 12)
    station_data['Month_cos'] = np.cos(2 * np.pi * station_data['Month'] / 12)
    
    # Define features
    base_features = ['PM2.5', 'PM10', 'CO', 'NO', 'NO2', 'NH3', 'O3', 'SO2']
    lag_features = [f'AQI_lag_{i}' for i in range(1, lag_days + 1)]
    rolling_features = ['AQI_rolling_7', 'AQI_rolling_30']
    time_features = ['DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos']
    
    all_features = base_features + lag_features + rolling_features + time_features
    all_features = [f for f in all_features if f in station_data.columns]
    
    # Remove rows with missing values
    station_data = station_data.dropna(subset=all_features + ['AQI'])
    
    if len(station_data) < 50:
        return None, None, None, None, None
    
    X = station_data[all_features]
    y = station_data['AQI']
    
    # Time series split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, all_features

def train_ml_models_for_comparison(X_train, y_train):
    """Train ML models for comparison with ARIMA"""
    models = {}
    
    # Models with different characteristics for XAI
    models['Linear Regression'] = LinearRegression()
    models['Decision Tree'] = DecisionTreeRegressor(max_depth=5, random_state=42)
    models['Random Forest'] = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    models['XGBoost'] = xgb.XGBRegressor(n_estimators=50, max_depth=5, random_state=42)
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    return trained_models

# ==================== MAIN APPLICATION ====================

def main():
    """Main application function"""
    
    # Header Section
    st.markdown("# 🌍 INDIA'S AQI DASHBOARD")
    st.markdown("### 🤖 ARIMA-Powered Air Quality Forecasting with XAI")
    st.markdown("*Superior Time Series Analysis with Explainable AI*")
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
            index=len(available_years) - 1,
            key="year_select"
        )
        
        # Month Selection
        available_months = sorted(df[df['Year'] == selected_year]['Month'].unique())
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        selected_month = st.selectbox(
            "📆 Select Month",
            available_months,
            format_func=lambda x: month_names[x-1],
            index=len(available_months) - 1,
            key="month_select"
        )
        
        # Station Selection
        stations = sorted(df['Station'].unique())
        selected_station = st.selectbox(
            "📍 Select Station",
            stations,
            index=0,
            key="station_select"
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
                max_value=max_date,
                key="date_select"
            )
        else:
            st.error(f"No data available for {selected_station} in {month_names[selected_month-1]} {selected_year}")
            return
        
        st.markdown("---")
        
        # XAI Settings
        st.markdown("### 🔍 Explainable AI Settings")
        enable_xai = st.checkbox("Enable XAI Explanations", value=True, key="enable_xai")
        
        if enable_xai:
            xai_method = st.selectbox(
                "XAI Method",
                ["All Methods", "SHAP", "Feature Importance", "Partial Dependence"],
                key="xai_method"
            )
        
        st.markdown("---")
        
        # ARIMA Settings
        st.markdown("### 🔧 ARIMA Settings")
        use_auto_arima = st.checkbox("Auto-tune ARIMA parameters", value=True, key="auto_arima")
        
        if not use_auto_arima:
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.slider("AR(p)", 0, 3, 1, help="Autoregressive order", key="p_slider")
            with col2:
                d = st.slider("I(d)", 0, 2, 1, help="Differencing order", key="d_slider")
            with col3:
                q = st.slider("MA(q)", 0, 3, 1, help="Moving average order", key="q_slider")
            arima_order = (p, d, q)
        
        st.markdown("---")
        
        # About Section
        st.markdown("### ℹ️ About Dashboard")
        
        min_date_overall = df['Date'].min()
        max_date_overall = df['Date'].max()
        
        st.info(f"""
        **📊 Data Coverage**
        
        Monitoring **{len(df['Station'].unique())}** stations across India
        
        **Period:** {min_date_overall.strftime('%b %d, %Y')} to {max_date_overall.strftime('%b %d, %Y')}
        
        **🤖 Primary Model:**
        - ✅ **ARIMA Time Series** - Best for temporal patterns
        
        **🧠 Explainable AI:**
        - 📊 Feature Importance
        - 🔍 SHAP Values
        - 🎯 Individual Predictions
        - 📈 Model Diagnostics
        
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
    
    # Create Tabs - UPDATED WITH XAI TAB
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📅 Daily View",
        "🔮 ARIMA Forecast",
        "🧠 ARIMA XAI",
        "📊 Analytics",
        "🗺️ Station Map",
        "📈 Model Comparison",
        "🎯 Why ARIMA Wins"
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
            create_hourly_chart(hourly_data, selected_station, selected_date_dt, "_daily")
            
            st.markdown("")
            
            # Historical Trend Chart
            st.markdown("### 📈 Historical Trend (Last 30 Days)")
            historical_data = get_historical_data(df, selected_station, days=30)
            create_historical_chart(historical_data, selected_station, "_daily")
            
            st.markdown("")
            
            # India Map
            st.markdown("### 🗺️ All India AQI Map")
            create_india_map(df, selected_date_dt, "_daily")
    
    # ==================== TAB 2: ARIMA FORECAST ====================
    with tab2:
        st.markdown(f"## 🔮 ARIMA Time Series Forecast")
        st.markdown(f"### {selected_station}")
        st.markdown("")
        
        # Get historical data for forecasting
        daily_aqi_series = df[df['Station'] == selected_station].groupby('Date')['AQI'].mean().sort_index()
        
        if len(daily_aqi_series) >= 30:
            # ARIMA Model Training
            with st.spinner("🔬 Training ARIMA model..."):
                if use_auto_arima:
                    # Auto-tune ARIMA parameters
                    best_order, best_aic = find_best_arima_order(daily_aqi_series)
                    st.success(f"✅ Found optimal ARIMA{best_order} with AIC: {best_aic:.2f}")
                    arima_model = train_arima_model(daily_aqi_series, order=best_order)
                else:
                    # Use user-defined order
                    arima_model = train_arima_model(daily_aqi_series, order=arima_order)
                    best_order = arima_order
            
            if arima_model:
                # Display ARIMA Model Summary
                st.markdown("### 📋 ARIMA Model Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AR Order (p)", f"{best_order[0]}")
                with col2:
                    st.metric("Diff Order (d)", f"{best_order[1]}")
                with col3:
                    st.metric("MA Order (q)", f"{best_order[2]}")
                
                # Generate forecast
                forecast, conf_int = arima_forecast(arima_model, steps=7)
                
                if forecast is not None:
                    # Prepare forecast data
                    last_date = daily_aqi_series.index[-1]
                    
                    # Convert forecast to list/numpy array
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
                    create_forecast_chart(combined_data, selected_station, "_forecast")
                    
                    # Display forecast details
                    st.markdown("### 📋 7-Day Forecast Details")
                    forecast_df = pd.DataFrame(forecast_data)
                    forecast_df['Category'] = forecast_df['aqi'].apply(lambda x: get_aqi_category(x)['category'])
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        for idx, row in forecast_df.iterrows():
                            aqi_cat = get_aqi_category(row['aqi'])
                            date_str = row['date'].strftime('%A, %B %d')
                            confidence_width = row['upper_bound'] - row['lower_bound']
                            
                            st.markdown(f"""
                            <div class="custom-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <div style="color: #ffffff; font-size: 18px; font-weight: 700; margin-bottom: 8px;">
                                            {date_str}
                                        </div>
                                        <div style="color: #b0bec5; font-size: 14px; font-weight: 500;">
                                            📊 95% CI: {int(row['lower_bound'])} - {int(row['upper_bound'])} AQI
                                        </div>
                                        <div style="color: #b0bec5; font-size: 12px; margin-top: 4px;">
                                            Uncertainty: ±{int(confidence_width/2)} points
                                        </div>
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="color: {aqi_cat['color']}; font-size: 40px; font-weight: 900;">
                                            {int(row['aqi'])}
                                        </div>
                                        <div style="color: #ffffff; font-size: 14px; font-weight: 700;">
                                            {aqi_cat['category']} {aqi_cat['emoji']}
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
                        avg_uncertainty = (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean() / 2
                        
                        st.metric("📈 Average AQI", f"{int(avg_forecast)}")
                        st.metric("🔴 Peak AQI", f"{int(max_forecast)}")
                        st.metric("🟢 Best AQI", f"{int(min_forecast)}")
                        st.metric("📊 Avg Uncertainty", f"±{int(avg_uncertainty)}")
                        
                        # ARIMA Model Metrics
                        st.markdown("---")
                        st.markdown("#### 🎯 Model Performance")
                        
                        # Evaluate ARIMA on historical data
                        train_size = int(len(daily_aqi_series) * 0.8)
                        train_data = daily_aqi_series[:train_size]
                        test_data = daily_aqi_series[train_size:]
                        
                        arima_metrics = evaluate_arima_model(train_data, test_data, best_order)
                        
                        if arima_metrics:
                            st.metric("R² Score", f"{arima_metrics['R2']:.3f}")
                            st.metric("MAE", f"{arima_metrics['MAE']:.1f}")
                            st.metric("MAPE", f"{arima_metrics['MAPE']:.1f}%")
                else:
                    st.warning("Failed to generate forecast")
            else:
                st.warning("Failed to train ARIMA model")
        else:
            st.warning(f"Need at least 30 days of historical data for ARIMA forecasting. Currently have {len(daily_aqi_series)} days.")
    
    # ==================== TAB 3: ARIMA XAI ====================
    with tab3:
        if enable_xai:
            st.markdown(f"## 🧠 ARIMA Model Explainability")
            st.markdown(f"### Understanding {selected_station}'s AQI Time Series")
            st.markdown("")
            
            # Get historical data
            daily_aqi_series = df[df['Station'] == selected_station].groupby('Date')['AQI'].mean().sort_index()
            
            if len(daily_aqi_series) >= 60:
                # Train ARIMA model for explanation
                with st.spinner("Training ARIMA model for explanation..."):
                    if use_auto_arima:
                        best_order, _ = find_best_arima_order(daily_aqi_series)
                        arima_model = train_arima_model(daily_aqi_series, order=best_order)
                    else:
                        arima_model = train_arima_model(daily_aqi_series, order=arima_order)
                        best_order = arima_order
                
                if arima_model:
                    # Show ARIMA XAI
                    explain_arima_model(arima_model, daily_aqi_series)
                else:
                    st.warning("Could not train ARIMA model for explanation")
            else:
                st.warning(f"Need at least 60 days of data for ARIMA XAI. Currently have {len(daily_aqi_series)} days.")
        else:
            st.info("🔒 XAI explanations are disabled. Enable them in the sidebar to see model interpretability features.")
    
    # ==================== TAB 4: ANALYTICS ====================
    with tab4:
        month_names_full = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
        st.markdown(f"## Monthly & Yearly Analytics")
        st.markdown(f"### {selected_station}")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly Analysis
            st.markdown(f"#### {month_names_full[selected_month-1]} {selected_year}")
            monthly_data = df[(df['Year'] == selected_year) & (df['Month'] == selected_month) & (df['Station'] == selected_station)]
            
            if len(monthly_data) > 0:
                monthly_pollutants = {
                    'PM2.5': monthly_data['PM2.5'].mean(),
                    'PM10': monthly_data['PM10'].mean()
                }
                monthly_aqi = calculate_aqi(monthly_pollutants)
                monthly_aqi_info = get_aqi_category(monthly_aqi)
                
                st.metric("Monthly Avg AQI", f"{monthly_aqi:.1f}", monthly_aqi_info['category'])
                
                # Monthly Stats
                st.write(f"**Days with data:** {monthly_data['Date'].nunique()}")
                st.write(f"**Total records:** {len(monthly_data)}")
            else:
                st.info("No monthly data available")
        
        with col2:
            # Yearly Analysis
            st.markdown(f"#### Year {selected_year}")
            year_data = df[(df['Year'] == selected_year) & (df['Station'] == selected_station)]
            
            if len(year_data) > 0:
                year_pollutants = {
                    'PM2.5': year_data['PM2.5'].mean(),
                    'PM10': year_data['PM10'].mean()
                }
                year_aqi = calculate_aqi(year_pollutants)
                year_aqi_info = get_aqi_category(year_aqi)
                
                st.metric("Yearly Avg AQI", f"{year_aqi:.1f}", year_aqi_info['category'])
                
                # Yearly Stats
                st.write(f"**Months with data:** {year_data['Month'].nunique()}")
                st.write(f"**Days with data:** {year_data['Date'].nunique()}")
            else:
                st.info("No yearly data available")
        
        # Year-over-Year Comparison
        st.markdown("---")
        st.markdown("### 📊 Year-over-Year Comparison")
        
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
                height=400,
                showlegend=False,
                plot_bgcolor='#0a1929',
                paper_bgcolor='#0a1929',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig, use_container_width=True, key="yearly_comparison_chart")
        else:
            st.info("Insufficient data for yearly comparison")
    
    # ==================== TAB 5: STATION MAP ====================
    with tab5:
        st.markdown(f"## 🗺️ All India AQI Map")
        st.markdown(f"### {selected_date.strftime('%B %d, %Y')}")
        st.markdown("")
        
        create_india_map(df, selected_date_dt, "_mapview")
        
        # Top Cities Comparison
        st.markdown("---")
        st.markdown("### 🏙️ City Comparison")
        
        top_cities = get_top_cities(df, selected_date_dt, top_n=20)
        
        if top_cities is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔴 Most Polluted Cities")
                top_polluted = top_cities.head(10)
                
                for idx, row in top_polluted.iterrows():
                    aqi_info = get_aqi_category(row['aqi'])
                    st.markdown(f"""
                    <div class="city-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <div style="color: #ffffff; font-weight: 600; font-size: 16px;">{row['station']}</div>
                            </div>
                            <div style="
                                background: {aqi_info['color']};
                                color: #ffffff;
                                padding: 6px 12px;
                                border-radius: 6px;
                                font-weight: 700;
                                font-size: 14px;
                            ">{int(row['aqi'])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🟢 Cleanest Cities")
                top_clean = top_cities.sort_values('aqi', ascending=True).head(10)
                
                for idx, row in top_clean.iterrows():
                    aqi_info = get_aqi_category(row['aqi'])
                    st.markdown(f"""
                    <div class="city-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <div style="color: #ffffff; font-weight: 600; font-size: 16px;">{row['station']}</div>
                            </div>
                            <div style="
                                background: {aqi_info['color']};
                                color: #ffffff;
                                padding: 6px 12px;
                                border-radius: 6px;
                                font-weight: 700;
                                font-size: 14px;
                            ">{int(row['aqi'])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No data available for city comparison")
    
    # ==================== TAB 6: MODEL COMPARISON ====================
    with tab6:
        st.markdown(f"## 📊 Model Performance Comparison")
        st.markdown(f"### Why ARIMA Outperforms Traditional ML for Time Series")
        st.markdown("")
        
        # Get data for comparison
        daily_aqi_series = df[df['Station'] == selected_station].groupby('Date')['AQI'].mean().sort_index()
        
        if len(daily_aqi_series) >= 100:
            # ARIMA Evaluation
            with st.spinner("Evaluating models..."):
                # Split data for evaluation
                train_size = int(len(daily_aqi_series) * 0.8)
                train_data = daily_aqi_series[:train_size]
                test_data = daily_aqi_series[train_size:]
                
                # Find best ARIMA
                best_order, best_aic = find_best_arima_order(train_data)
                arima_metrics = evaluate_arima_model(train_data, test_data, best_order)
                
                # ML Evaluation
                ml_data = prepare_time_series_ml_data(df, selected_station)
                
                if ml_data[0] is not None:
                    X_train, X_test, y_train, y_test, feature_names = ml_data
                    ml_models = train_ml_models_for_comparison(X_train, y_train)
                    
                    # Calculate ML metrics
                    ml_metrics = {}
                    for name, model in ml_models.items():
                        y_pred = model.predict(X_test)
                        ml_metrics[name] = {
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'R2': r2_score(y_test, y_pred),
                            'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100
                        }
                
            # Display Comparison
            if arima_metrics:
                st.markdown("### 🎯 Performance Metrics")
                
                # Create comparison table
                comparison_data = []
                
                # ARIMA metrics
                comparison_data.append({
                    'Model': f'ARIMA{best_order}',
                    'R²': arima_metrics['R2'],
                    'MAE': arima_metrics['MAE'],
                    'MAPE': f"{arima_metrics['MAPE']:.1f}%",
                    'AIC': f"{arima_metrics['AIC']:.1f}",
                    'Type': 'Time Series'
                })
                
                # ML metrics
                if 'ml_metrics' in locals():
                    for name, metrics in ml_metrics.items():
                        comparison_data.append({
                            'Model': name,
                            'R²': metrics['R2'],
                            'MAE': metrics['MAE'],
                            'MAPE': f"{metrics['MAPE']:.1f}%",
                            'AIC': 'N/A',
                            'Type': 'Traditional ML'
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('R²', ascending=False)
                
                # Highlight ARIMA
                def highlight_arima(row):
                    if 'ARIMA' in row['Model']:
                        return ['background-color: #1a237e; color: white'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(comparison_df.style.apply(highlight_arima, axis=1).format({
                    'R²': '{:.3f}',
                    'MAE': '{:.1f}'
                }), use_container_width=True)
                
                # Visual Comparison
                st.markdown("### 📈 Visual Comparison")
                
                # Create bar chart
                fig = go.Figure()
                
                # R² Scores
                fig.add_trace(go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df['R²'],
                    name='R² Score',
                    marker=dict(
                        color=['#5c6bc0' if 'ARIMA' in m else '#ff9800' for m in comparison_df['Model']],
                        line=dict(width=0)
                    ),
                    hovertemplate='<b>%{x}</b><br>R²: %{y:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(
                        text="Model Comparison - R² Scores (Higher is Better)",
                        font=dict(color='#ffffff', size=16)
                    ),
                    template="plotly_dark",
                    height=400,
                    plot_bgcolor='#0a1929',
                    paper_bgcolor='#0a1929',
                    font=dict(color='#ffffff'),
                    xaxis=dict(
                        tickfont=dict(color='#ffffff'),
                        title=dict(text="Model", font=dict(color='#b0bec5'))
                    ),
                    yaxis=dict(
                        tickfont=dict(color='#ffffff'),
                        title=dict(text="R² Score", font=dict(color='#b0bec5'))
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True, key="model_comparison_chart")
                
                # XAI for ML Models if enabled
                if enable_xai and 'ml_models' in locals() and ml_models:
                    st.markdown("---")
                    st.markdown("## 🧠 ML Model Explanations")
                    
                    # Select ML model for XAI
                    selected_ml_model = st.selectbox(
                        "Select ML Model for XAI Analysis:",
                        list(ml_models.keys()),
                        key="ml_model_select"
                    )
                    
                    if selected_ml_model in ml_models:
                        model = ml_models[selected_ml_model]
                        
                        # Determine model type for SHAP
                        if selected_ml_model in ['Random Forest', 'Decision Tree', 'XGBoost']:
                            model_type = 'tree'
                        elif selected_ml_model == 'Linear Regression':
                            model_type = 'linear'
                        else:
                            model_type = 'kernel'
                        
                        # Create XAI dashboard for selected model
                        create_model_interpretation_dashboard(
                            model, X_test, y_test, feature_names, 
                            selected_ml_model, model_type
                        )
                
                # Key Insights
                st.markdown("### 💡 Key Insights")
                
                if arima_metrics['R2'] > 0:
                    if arima_metrics['R2'] > 0.7:
                        performance = "**Excellent** 🏆"
                        color = "#4caf50"
                    elif arima_metrics['R2'] > 0.5:
                        performance = "**Good** 👍"
                        color = "#8bc34a"
                    else:
                        performance = "**Moderate** ⚠️"
                        color = "#ffc107"
                    
                    st.markdown(f"""
                    <div class="custom-card">
                        <h4 style="color: {color};">ARIMA Performance: {performance}</h4>
                        <p style="color: #b0bec5;">
                        ARIMA{best_order} explains <strong>{arima_metrics['R2']:.1%}</strong> of AQI variance with 
                        <strong>{arima_metrics['MAE']:.1f}</strong> MAE error.
                        </p>
                        <p style="color: #b0bec5; margin-top: 12px;">
                        <strong>✅ ARIMA Advantages:</strong><br>
                        • Captures temporal dependencies<br>
                        • Provides uncertainty quantification<br>
                        • Specifically designed for time series
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("ARIMA model needs improvement. Consider adding more historical data or trying different parameters.")
            else:
                st.warning("Could not evaluate ARIMA model")
        else:
            st.warning(f"Need at least 100 days of data for model comparison. Currently have {len(daily_aqi_series)} days.")
    
    # ==================== TAB 7: WHY ARIMA WINS ====================
    with tab7:
        st.markdown("## 🎯 Why ARIMA is Superior for AQI Forecasting")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="arima-card">
                <h3>✅ ARIMA Strengths</h3>
                <div style="margin-top: 1.5rem;">
                    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
                        <div style="background: #5c6bc0; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: 700;">1</div>
                        <div>
                            <h4 style="color: #ffffff; margin: 0;">Temporal Awareness</h4>
                            <p style="color: #bbdefb; margin: 4px 0 0 0;">Understands that today's AQI depends on yesterday's</p>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
                        <div style="background: #5c6bc0; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: 700;">2</div>
                        <div>
                            <h4 style="color: #ffffff; margin: 0;">Trend Capture</h4>
                            <p style="color: #bbdefb; margin: 4px 0 0 0;">Models increasing/decreasing pollution trends</p>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
                        <div style="background: #5c6bc0; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: 700;">3</div>
                        <div>
                            <h4 style="color: #ffffff; margin: 0;">Uncertainty Quantification</h4>
                            <p style="color: #bbdefb; margin: 4px 0 0 0;">Provides confidence intervals for predictions</p>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: flex-start;">
                        <div style="background: #5c6bc0; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: 700;">4</div>
                        <div>
                            <h4 style="color: #ffffff; margin: 0;">Seasonality Handling</h4>
                            <p style="color: #bbdefb; margin: 4px 0 0 0;">Captures weekly, monthly, seasonal patterns</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="comparison-card">
                <h3>⚠️ Traditional ML Limitations</h3>
                <div style="margin-top: 1.5rem;">
                    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
                        <div style="background: #ab47bc; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: 700;">1</div>
                        <div>
                            <h4 style="color: #ffffff; margin: 0;">No Time Understanding</h4>
                            <p style="color: #e1bee7; margin: 4px 0 0 0;">Treats Monday and Tuesday as independent events</p>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
                        <div style="background: #ab47bc; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: 700;">2</div>
                        <div>
                            <h4 style="color: #ffffff; margin: 0;">Poor Trend Extrapolation</h4>
                            <p style="color: #e1bee7; margin: 4px 0 0 0;">Cannot predict beyond training data range</p>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
                        <div style="background: #ab47bc; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: 700;">3</div>
                        <div>
                            <h4 style="color: #ffffff; margin: 0;">No Uncertainty Measures</h4>
                            <p style="color: #e1bee7; margin: 4px 0 0 0;">Cannot tell how confident predictions are</p>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: flex-start;">
                        <div style="background: #ab47bc; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: 700;">4</div>
                        <div>
                            <h4 style="color: #ffffff; margin: 0;">Feature Engineering Required</h4>
                            <p style="color: #e1bee7; margin: 4px 0 0 0;">Needs manual creation of lag/rolling features</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical Explanation
        st.markdown("---")
        st.markdown("### 🔬 Technical Explanation: How ARIMA Works")
        
        st.markdown("""
        <div class="custom-card">
            <h4>ARIMA = AutoRegressive + Integrated + Moving Average</h4>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 1.5rem 0;">
                <div class="arima-param">
                    <h5 style="color: #ffffff; margin-bottom: 8px;">AR(p)</h5>
                    <p style="color: #bbdefb; font-size: 14px;">
                    <strong>AutoRegressive</strong><br>
                    Current AQI depends on previous p days' AQI values
                    </p>
                    <p style="color: #7986cb; font-size: 12px; margin-top: 8px;">
                    Formula: AQIₜ = φ₁AQIₜ₋₁ + ... + φₚAQIₜ₋ₚ
                    </p>
                </div>
                
                <div class="arima-param">
                    <h5 style="color: #ffffff; margin-bottom: 8px;">I(d)</h5>
                    <p style="color: #bbdefb; font-size: 14px;">
                    <strong>Integrated</strong><br>
                    Differencing to make time series stationary (remove trends)
                    </p>
                    <p style="color: #7986cb; font-size: 12px; margin-top: 8px;">
                    Formula: ΔAQIₜ = AQIₜ - AQIₜ₋₁
                    </p>
                </div>
                
                <div class="arima-param">
                    <h5 style="color: #ffffff; margin-bottom: 8px;">MA(q)</h5>
                    <p style="color: #bbdefb; font-size: 14px;">
                    <strong>Moving Average</strong><br>
                    Current AQI depends on previous q days' forecast errors
                    </p>
                    <p style="color: #7986cb; font-size: 12px; margin-top: 8px;">
                    Formula: AQIₜ = εₜ + θ₁εₜ₋₁ + ... + θₚεₜ₋ₚ
                    </p>
                </div>
            </div>
            
            <div style="background: rgba(92, 107, 192, 0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <p style="color: #e3f2fd;">
                <strong>🎯 Key Insight for AQI:</strong> Air pollution exhibits strong autocorrelation - 
                today's pollution levels are highly dependent on yesterday's. This temporal dependency is 
                exactly what ARIMA models capture, making it the ideal choice for AQI forecasting.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # XAI Benefits
        st.markdown("### 🧠 Benefits of Explainable AI (XAI) in AQI Forecasting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="xai-explanation">
                <h4>🔍 Transparency</h4>
                <ul style="color: #b2dfdb;">
                    <li>Understand why ARIMA predicts certain AQI values</li>
                    <li>See which time lags are most influential</li>
                    <li>Identify seasonal patterns in pollution</li>
                    <li>Build trust in model predictions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="shap-card">
                <h4>🎯 Actionable Insights</h4>
                <ul style="color: #f8bbd9;">
                    <li>Identify key pollution drivers</li>
                    <li>Understand prediction confidence intervals</li>
                    <li>Detect unusual patterns (anomalies)</li>
                    <li>Make data-driven policy decisions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #b0bec5;">
        <p style="font-size: 14px; margin-bottom: 8px;">📊 Data Source: Central Pollution Control Board (CPCB) | AQI calculated using Indian standards</p>
        <p style="font-size: 13px; font-weight: 600;">
        🌍 Built with Streamlit • 📈 Powered by <strong>ARIMA Time Series Models</strong> • 🧠 Explainable AI • 
        🎯 Confidence-Aware Forecasting • 💙 Made for India
        </p>
        <p style="font-size: 12px; margin-top: 12px; color: #7986cb;">
        ⚡ <strong>Primary Model:</strong> ARIMA | <strong>XAI Methods:</strong> SHAP, Feature Importance, Residual Analysis | 
        <strong>Key Advantage:</strong> Temporal Dependency Capture with Explainability
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    main()