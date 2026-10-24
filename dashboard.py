"""
Fraud Detection Dashboard
Professional Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.feature_engineer import FraudDataFeatureEngineer, CreditCardFeatureEngineer, prepare_features
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.predictor import load_predictor
from src.config import data_config, model_config

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main styling */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e1e 0%, #2d2d2d 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1e1e1e 0%, #2d2d2d 100%);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# ==================== HELPER FUNCTIONS ====================
def get_model_path(model_type: str, ds_name: str) -> Path:
    """
    Get model path supporting both naming conventions:
    - Notebook convention: {dataset_prefix}_{model_type}_model.joblib (e.g., ecommerce_xgboost_model.joblib)
    - Dashboard convention: {model_type}_{ds_name}_model.joblib (e.g., xgboost_fraud_model.joblib)
    """
    models_dir = model_config.model_save_dir
    
    # Map ds_name to notebook prefix
    prefix_map = {'fraud': 'ecommerce', 'creditcard': 'creditcard'}
    notebook_prefix = prefix_map.get(ds_name, ds_name)
    
    # Try notebook naming convention first (from trained notebooks)
    notebook_path = models_dir / f"{notebook_prefix}_{model_type}_model.joblib"
    if notebook_path.exists():
        return notebook_path
    
    # Try dashboard naming convention
    dashboard_path = models_dir / f"{model_type}_{ds_name}_model.joblib"
    if dashboard_path.exists():
        return dashboard_path
    
    # Return notebook path as default (will raise FileNotFoundError if not exists)
    return notebook_path

def get_available_models(ds_name: str) -> list:
    """
    Get available models supporting both naming conventions.
    Returns list of tuples: (display_name, model_type, path)
    """
    models_dir = model_config.model_save_dir
    if not models_dir.exists():
        return []
    
    available = []
    prefix_map = {'fraud': 'ecommerce', 'creditcard': 'creditcard'}
    notebook_prefix = prefix_map.get(ds_name, ds_name)
    
    # Find notebook-style models: {prefix}_{model_type}_model.joblib
    for model_path in models_dir.glob(f"{notebook_prefix}_*_model.joblib"):
        # Extract model type from filename
        filename = model_path.stem  # e.g., 'ecommerce_xgboost_model'
        model_type = filename.replace(f"{notebook_prefix}_", "").replace("_model", "")
        display_name = model_type.replace('_', ' ').title()
        available.append((display_name, model_type, model_path))
    
    # Find dashboard-style models: {model_type}_{ds_name}_model.joblib
    for model_path in models_dir.glob(f"*_{ds_name}_model.joblib"):
        filename = model_path.stem
        model_type = filename.replace(f"_{ds_name}_model", "")
        # Avoid duplicates
        if not any(m[1] == model_type for m in available):
            display_name = model_type.replace('_', ' ').title()
            available.append((display_name, model_type, model_path))
    
    return available

@st.cache_data
def load_data(dataset_type):
    """Load fraud data with IP-to-country mapping (matching notebook pipeline)"""
    loader = DataLoader()
    try:
        if dataset_type == "Fraud E-commerce":
            # Load fraud data
            df = loader.load_fraud_data()
            
            # Load IP to Country mapping
            ip_country = loader.load_ip_to_country()
            
            if not ip_country.empty:
                # IP addresses in Fraud_Data.csv are ALREADY INTEGERS stored as floats
                # Just convert to int64 directly (no need to parse dot-notation)
                df['ip_as_int'] = df['ip_address'].fillna(0).astype('int64')
                
                # Ensure ip_country bounds are int64 for compatible merge
                ip_country['lower_bound_ip_address'] = ip_country['lower_bound_ip_address'].astype('int64')
                ip_country['upper_bound_ip_address'] = ip_country['upper_bound_ip_address'].astype('int64')
                
                # Sort both dataframes for merge_asof
                df_sorted = df.sort_values('ip_as_int')
                ip_country_sorted = ip_country.sort_values('lower_bound_ip_address')
                
                # Use merge_asof for range-based lookup
                df_merged = pd.merge_asof(
                    df_sorted,
                    ip_country_sorted[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
                    left_on='ip_as_int',
                    right_on='lower_bound_ip_address',
                    direction='backward'
                )
                
                # Filter out invalid matches (IP must be <= upper_bound)
                df_merged['ip_country'] = df_merged.apply(
                    lambda row: row['country'] if pd.notna(row['upper_bound_ip_address']) and row['ip_as_int'] <= row['upper_bound_ip_address'] else None,
                    axis=1
                )
                
                # Drop temporary columns and restore original order
                df = df_merged.drop(
                    columns=['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
                ).sort_index()
                
                # Calculate fraud rate by country and create is_high_risk_country feature
                country_fraud_stats = df[df['ip_country'].notna()].groupby('ip_country').agg(
                    total_transactions=('class', 'count'),
                    fraud_count=('class', 'sum')
                ).reset_index()
                country_fraud_stats['fraud_rate'] = country_fraud_stats['fraud_count'] / country_fraud_stats['total_transactions']
                
                # Define high-risk countries as those with fraud rate > 20%
                HIGH_RISK_THRESHOLD = 0.20
                high_risk_countries = set(country_fraud_stats[country_fraud_stats['fraud_rate'] > HIGH_RISK_THRESHOLD]['ip_country'].tolist())
                
                # Create the is_high_risk_country feature
                df['is_high_risk_country'] = df['ip_country'].apply(
                    lambda x: 1 if x in high_risk_countries else 0
                )
            
            return df, 'fraud'
        else:
            df = loader.load_creditcard_data()
            return df, 'creditcard'
    except FileNotFoundError:
        return None, None

@st.cache_data
def get_class_distribution(df, target_col):
    """Get class distribution"""
    return df[target_col].value_counts().sort_index()

def process_single_transaction_for_prediction(sample_row, df, ds_name):
    """
    Process a single transaction through the full pipeline (matching notebook).
    
    Args:
        sample_row: DataFrame with single transaction
        df: Full dataset for reference
        ds_name: Dataset name ('fraud' or 'creditcard')
    
    Returns:
        Processed features ready for model prediction
    """
    if ds_name == 'fraud':
        # Apply feature engineering (matching notebook)
        engineer = FraudDataFeatureEngineer()
        sample_engineered = engineer.engineer_features(sample_row.copy(), fit=False)
        
        # Drop columns exactly like notebook
        columns_to_drop = ['class', 'signup_time', 'purchase_time', 'device_id', 
                         'user_id', 'ip_address', 'browser_source', 'ip_country']
        existing_cols_to_drop = [col for col in columns_to_drop if col in sample_engineered.columns]
        
        X_sample = sample_engineered.drop(columns=existing_cols_to_drop, errors='ignore')
        
    else:
        # Credit card processing
        engineer = CreditCardFeatureEngineer()
        sample_engineered = engineer.engineer_features(sample_row.copy(), fit=False)
        
        columns_to_drop = data_config.creditcard_drop_columns + ['class', 'Class']
        existing_cols_to_drop = [col for col in columns_to_drop if col in sample_engineered.columns]
        
        X_sample = sample_engineered.drop(columns=existing_cols_to_drop, errors='ignore')
    
    return X_sample

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### Fraud Detection System")
    st.markdown("---")
    
    # Dataset selection
    st.markdown("**Select Dataset**")
    dataset_type = st.selectbox(
        "Dataset",
        ["Fraud E-commerce", "Credit Card"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Main navigation
    st.markdown("### Navigation")
    
    main_page = st.radio(
        "Main Page",
        ["Fraud Detector", "🔧 Model Management"],
        label_visibility="collapsed"
    )
    
    # Sub-navigation for Model Management
    if main_page == "🔧 Model Management":
        st.markdown("---")
        st.markdown("**Management Tools**")
        sub_page = st.radio(
            "Tools",
            ["Data Analysis", "Feature Engineering", "Train Models", 
             "Evaluate Models", "Analytics Dashboard"],
            label_visibility="collapsed"
        )
    else:
        sub_page = None
    
    st.markdown("---")
    
    # System status
    st.markdown("### System Status")
    
    models_dir = model_config.model_save_dir
    if models_dir.exists():
        # Count only model files (exclude preprocessors and feature engineers)
        models = [f for f in models_dir.glob("*_model.joblib")]
        st.metric("Available Models", len(models))
    else:
        st.metric("Available Models", 0)
    
    data_available = 0
    if data_config.fraud_data_path.exists():
        data_available += 1
    if data_config.creditcard_data_path.exists():
        data_available += 1
    st.metric("Datasets Loaded", data_available)
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "**Fraud Detection System v2.0**\n\n"
        "ML-powered fraud detection\n\n"
        "© 2024 Adey Innovations Inc."
    )

# ==================== LOAD DATA ====================
df, ds_name = load_data(dataset_type)

# ==================== FRAUD DETECTOR PAGE ====================
if main_page == "Fraud Detector":
    st.markdown('<div class="main-header"><h1>Fraud Detector</h1><p>Real-time fraud prediction powered by machine learning</p></div>', unsafe_allow_html=True)
    
    # Check for trained models (supports both naming conventions)
    available_models = get_available_models(ds_name)
    
    if not available_models:
        st.warning("No trained models found. Train a model first in Model Management or run the notebook pipelines.")
    else:
        # Model selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            model_options = [(m[0], m[1]) for m in available_models]  # (display_name, model_type)
            selected_display = st.selectbox(
                "Select Model",
                [m[0] for m in model_options],
            )
            # Get the actual model_type from the selected display name
            selected_model = next(m[1] for m in model_options if m[0] == selected_display)
        
        with col2:
            threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        
        # Prediction tabs
        tab1, tab2, tab3 = st.tabs(["Single Transaction", "Batch Prediction", "Upload CSV"])
        
        # Single Transaction
        with tab1:
            st.markdown("### Enter Transaction Details")
            
            if ds_name == 'fraud':
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    purchase_value = st.number_input("Purchase Value ($)", min_value=0.0, value=150.0, step=10.0)
                    age = st.number_input("👤 Customer Age", min_value=18, max_value=120, value=35)
                
                with col2:
                    hour_of_day = st.number_input("Hour of Day", min_value=0, max_value=23, value=14)
                    day_of_week = st.selectbox(
                        "Day of Week",
                        options=[0, 1, 2, 3, 4, 5, 6],
                        format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x],
                        index=2
                    )
                
                with col3:
                    time_since_signup = st.number_input("Hours Since Signup", min_value=0.0, value=48.5, step=1.0)
                
                st.markdown("")
                
                if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
                    try:
                        # Create timestamps
                        purchase_time = datetime.now().replace(hour=hour_of_day, minute=0, second=0, microsecond=0)
                        days_to_add = (day_of_week - purchase_time.weekday()) % 7
                        purchase_time = purchase_time + timedelta(days=days_to_add)
                        signup_time = purchase_time - timedelta(hours=time_since_signup)
                        
                        # Get sample row structure from the loaded data
                        sample_row = df.iloc[0:1].copy()
                        sample_row['purchase_value'] = purchase_value
                        sample_row['age'] = age
                        sample_row['signup_time'] = signup_time.strftime('%Y-%m-%d %H:%M:%S')
                        sample_row['purchase_time'] = purchase_time.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Process through full pipeline (matching notebook)
                        X_sample = process_single_transaction_for_prediction(sample_row, df, ds_name)
                        
                        # Load processed data to get the preprocessor
                        if ds_name in st.session_state.processed_data:
                            df_processed = st.session_state.processed_data[ds_name]
                        else:
                            loader = DataLoader()
                            df_processed = loader.load_processed_data(ds_name)
                        
                        # Fit preprocessor on training data
                        y = df_processed[data_config.target_column]
                        X = df_processed.drop(columns=[data_config.target_column])
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=data_config.test_size,
                            random_state=data_config.random_state, stratify=y
                        )
                        
                        preprocessor = DataPreprocessor(use_smote=False)
                        X_train_proc, _ = preprocessor.fit_transform(X_train, y_train)
                        
                        # Transform sample (ensure correct columns)
                        missing_cols = set(X_train.columns) - set(X_sample.columns)
                        for col in missing_cols:
                            X_sample[col] = 0
                        
                        X_sample = X_sample[X_train.columns]
                        X_sample_proc = preprocessor.transform(X_sample)
                        
                        # Load model and predict
                        predictor = load_predictor(selected_model, ds_name)
                        result = predictor.predict(X_sample_proc, return_proba=True, threshold=threshold)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### Analysis Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            risk_color = "🔴" if result['predictions'][0] == 1 else "🟢"
                            risk_text = "FRAUD" if result['predictions'][0] == 1 else "LEGITIMATE"
                            st.metric("Prediction", f"{risk_color} {risk_text}")
                        
                        with col2:
                            prob = result['fraud_probability'][0] * 100
                            st.metric("Fraud Probability", f"{prob:.1f}%")
                        
                        with col3:
                            confidence = max(result['fraud_probability'][0], 1 - result['fraud_probability'][0]) * 100
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=result['fraud_probability'][0] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Fraud Risk Score"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': '#00CC96'},
                                    {'range': [30, 60], 'color': '#FFA15A'},
                                    {'range': [60, 80], 'color': '#FF6692'},
                                    {'range': [80, 100], 'color': '#EF553B'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': threshold * 100
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        with st.expander("Error Details"):
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.info("Credit card predictions require V1-V28 features. Use CSV upload.")
        
        # Batch Prediction
        with tab2:
            if df is not None:
                st.markdown("### Batch Prediction")
                
                num_samples = st.number_input(
                    "Number of samples",
                    min_value=1,
                    max_value=min(1000, len(df)),
                    value=min(10, len(df))
                )
                
                if st.button("Predict Batch", type="primary", use_container_width=True):
                    try:
                        # Sample data
                        sample_data = df.sample(n=num_samples, random_state=42)
                        
                        with st.spinner(f"Processing {num_samples} samples..."):
                            # Process through full pipeline (matching notebook)
                            X_batch = process_single_transaction_for_prediction(sample_data, df, ds_name)
                            
                            # Load processed data and fit preprocessor
                            if ds_name in st.session_state.processed_data:
                                df_processed = st.session_state.processed_data[ds_name]
                            else:
                                loader = DataLoader()
                                df_processed = loader.load_processed_data(ds_name)
                            
                            y = df_processed[data_config.target_column]
                            X = df_processed.drop(columns=[data_config.target_column])
                            
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=data_config.test_size,
                                random_state=data_config.random_state, stratify=y
                            )
                            
                            preprocessor = DataPreprocessor(use_smote=False)
                            X_train_proc, _ = preprocessor.fit_transform(X_train, y_train)
                            
                            # Ensure correct columns
                            missing_cols = set(X_train.columns) - set(X_batch.columns)
                            for col in missing_cols:
                                X_batch[col] = 0
                            
                            X_batch = X_batch[X_train.columns]
                            X_batch_proc = preprocessor.transform(X_batch)
                            
                            # Predict
                            predictor = load_predictor(selected_model, ds_name)
                            result = predictor.predict(X_batch_proc, return_proba=True, threshold=threshold)
                            
                            predictions = pd.DataFrame({
                                'is_fraud': result['predictions'],
                                'fraud_probability': result['fraud_probability'],
                                'confidence': [max(p, 1-p) for p in result['fraud_probability']],
                                'risk_level': [predictor._get_risk_level(p) for p in result['fraud_probability']]
                            })
                        
                        fraud_count = predictions['is_fraud'].sum()
                        fraud_pct = (fraud_count / len(predictions)) * 100
                        
                        st.markdown("### Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", f"{len(predictions):,}")
                        with col2:
                            st.metric("Fraud Detected", f"{fraud_count:,}", delta=f"{fraud_pct:.1f}%", delta_color="inverse")
                        with col3:
                            st.metric("Normal", f"{len(predictions) - fraud_count:,}")
                        
                        results_df = pd.concat([sample_data.reset_index(drop=True), predictions], axis=1)
                        st.dataframe(results_df, use_container_width=True, height=400)
                        
                        st.session_state.predictions = results_df
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            f"predictions_{ds_name}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # CSV Upload
        with tab3:
            st.markdown("### Upload CSV")
            
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file:
                try:
                    upload_df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(upload_df):,} records")
                    
                    with st.expander("Preview", expanded=True):
                        st.dataframe(upload_df.head(20), use_container_width=True)
                    
                    if st.button("Predict All", type="primary", use_container_width=True):
                        try:
                            predictor = load_predictor(selected_model, ds_name)
                            
                            with st.spinner(f"Processing {len(upload_df):,} records..."):
                                predictions = predictor.predict_batch(
                                    upload_df, threshold=threshold, include_details=True
                                )
                            
                            fraud_count = predictions['is_fraud'].sum()
                            
                            st.success(f"Complete! Found {fraud_count:,} fraud cases")
                            
                            results_df = pd.concat([upload_df.reset_index(drop=True), predictions], axis=1)
                            st.dataframe(results_df.head(100), use_container_width=True, height=400)
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "Download Complete Results",
                                csv,
                                f"predictions_{uploaded_file.name}",
                                "text/csv",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

# ==================== MODEL MANAGEMENT PAGES ====================
elif main_page == "🔧 Model Management":
    
    # Data Analysis
    if sub_page == "Data Analysis":
        st.markdown('<div class="main-header"><h1>Data Analysis</h1><p>Explore datasets</p></div>', unsafe_allow_html=True)
        
        if df is None:
            st.error("Data not found")
        else:
            st.success(f"Loaded {len(df):,} records")
            
            target_col = 'class' if 'class' in df.columns else 'Class'
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Records", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                fraud_rate = df[target_col].mean() * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            with col4:
                st.metric("Fraud Cases", f"{df[target_col].sum():,}")
            
            st.markdown("### Class Distribution")
            class_dist = get_class_distribution(df, target_col)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    values=class_dist.values,
                    names=['Normal', 'Fraud'],
                    color_discrete_sequence=['#00CC96', '#EF553B']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(go.Bar(
                    x=['Normal', 'Fraud'],
                    y=class_dist.values,
                    marker_color=['#00CC96', '#EF553B'],
                    text=class_dist.values,
                    textposition='auto'
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Data Preview")
            st.dataframe(df.head(100), use_container_width=True, height=400)
            
            with st.expander("Statistical Summary"):
                st.dataframe(df.describe(), use_container_width=True)
    
    # Feature Engineering
    elif sub_page == "Feature Engineering":
        st.markdown('<div class="main-header"><h1>🔧 Feature Engineering</h1><p>Create features (matching notebook pipeline)</p></div>', unsafe_allow_html=True)
        
        if df is None:
            st.error("Data not found")
        else:
            st.info(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Show IP country merge status for fraud dataset
            if ds_name == 'fraud' and 'ip_country' in df.columns:
                country_count = df['ip_country'].notna().sum()
                high_risk_count = df['is_high_risk_country'].sum() if 'is_high_risk_country' in df.columns else 0
                st.success(f"✓ IP-to-Country mapping: {country_count:,} records mapped")
                if high_risk_count > 0:
                    st.success(f"✓ High-risk country feature: {high_risk_count:,} flagged transactions")
            
            if st.button("Engineer Features", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        # Feature engineering (same as notebook)
                        if ds_name == 'fraud':
                            engineer = FraudDataFeatureEngineer()
                        else:
                            engineer = CreditCardFeatureEngineer()
                        
                        df_engineered = engineer.engineer_features(df.copy(), fit=True)
                        st.success(f"✓ Feature engineering complete: {df_engineered.shape[0]:,} × {df_engineered.shape[1]}")
                        
                        new_features = [col for col in df_engineered.columns if col not in df.columns]
                        st.success(f"✓ Created {len(new_features)} new features")
                        
                        # Prepare for training (drop columns exactly like notebook)
                        target_col = 'class' if 'class' in df_engineered.columns else 'Class'
                        
                        if ds_name == 'fraud':
                            # Match notebook's column dropping logic
                            columns_to_drop = ['class', 'signup_time', 'purchase_time', 'device_id', 
                                             'user_id', 'ip_address', 'browser_source', 'ip_country']
                        else:
                            columns_to_drop = data_config.creditcard_drop_columns + [target_col]
                        
                        # Keep only columns that exist
                        existing_cols_to_drop = [col for col in columns_to_drop if col in df_engineered.columns]
                        
                        # Create processed data with target column preserved separately
                        X = df_engineered.drop(columns=existing_cols_to_drop)
                        y = df_engineered[target_col]
                        
                        # Recombine for storage
                        df_processed = X.copy()
                        df_processed[data_config.target_column] = y.values
                        
                        st.info(f"✓ Dropped columns: {', '.join(existing_cols_to_drop)}")
                        st.info(f"✓ Final feature count: {X.shape[1]}")
                        
                        # Store in session state
                        st.session_state.processed_data[ds_name] = df_processed
                        
                        # Save processed data
                        loader = DataLoader()
                        loader.save_processed_data(df_processed, ds_name)
                        
                        st.success("✓ Processed data saved!")
                        
                        with st.expander("Preview Processed Data"):
                            st.dataframe(df_processed.head(20), use_container_width=True)
                        
                        with st.expander("Feature Summary"):
                            st.write(f"**Total features:** {X.shape[1]}")
                            st.write(f"**Feature dtypes:**")
                            dtype_counts = X.dtypes.value_counts()
                            for dtype, count in dtype_counts.items():
                                st.write(f"- {dtype}: {count} features")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        with st.expander("Error Details"):
                            import traceback
                            st.code(traceback.format_exc())
    
    # Train Models
    elif sub_page == "Train Models":
        st.markdown('<div class="main-header"><h1>Train Models</h1><p>Train ML models</p></div>', unsafe_allow_html=True)
        
        if df is None:
            st.error("Data not found")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_models = st.multiselect(
                    "Select Models",
                    ["logistic_regression", "random_forest", "xgboost"],
                    default=["random_forest"],
                    format_func=lambda x: x.replace('_', ' ').title()
                )
            
            with col2:
                use_smote = st.checkbox("Use SMOTE", value=True)
                perform_cv = st.checkbox("Cross-Validation", value=True)
            
            if st.button("Start Training", type="primary", use_container_width=True):
                if not selected_models:
                    st.error("Select at least one model")
                else:
                    try:
                        if ds_name in st.session_state.processed_data:
                            df_processed = st.session_state.processed_data[ds_name]
                        else:
                            with st.spinner("Loading..."):
                                loader = DataLoader()
                                df_processed = loader.load_processed_data(ds_name)
                        
                        y = df_processed[data_config.target_column]
                        X = df_processed.drop(columns=[data_config.target_column])
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=data_config.test_size,
                            random_state=data_config.random_state, stratify=y
                        )
                        
                        st.info(f"Training: {len(X_train):,} | Testing: {len(X_test):,}")
                        
                        with st.spinner("Preprocessing..."):
                            preprocessor = DataPreprocessor(use_smote=use_smote)
                            X_train_proc, y_train_proc = preprocessor.fit_transform(X_train, y_train)
                            X_test_proc = preprocessor.transform(X_test)
                        
                        if use_smote:
                            st.success(f"SMOTE: {len(X_train_proc):,} samples")
                        
                        results = {}
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, model_type in enumerate(selected_models):
                            status_text.text(f"Training {model_type.replace('_', ' ').title()}...")
                            
                            trainer = ModelTrainer(model_type=model_type)
                            metrics = trainer.train(X_train_proc, y_train_proc, validate=perform_cv)
                            
                            evaluator = ModelEvaluator()
                            y_pred_proba = trainer.predict_proba(X_test_proc)
                            y_pred = trainer.predict(X_test_proc)
                            
                            eval_metrics = evaluator.evaluate(y_test, y_pred_proba, y_pred)
                            
                            results[model_type] = {
                                'trainer': trainer,
                                'metrics': eval_metrics,
                                'train_metrics': metrics
                            }
                            
                            trainer.save_model(dataset_name=ds_name)
                            progress_bar.progress((idx + 1) / len(selected_models))
                        
                        status_text.text("Complete!")
                        
                        st.session_state.trained_models[ds_name] = results
                        st.session_state.test_data = (X_test_proc, y_test)
                        
                        st.success("All models trained!")
                        
                        st.markdown("### Performance")
                        
                        comparison_data = []
                        for model_name, result in results.items():
                            metrics = result['metrics']
                            comparison_data.append({
                                'Model': model_name.replace('_', ' ').title(),
                                'Accuracy': f"{metrics['accuracy']:.4f}",
                                'Precision': f"{metrics['precision']:.4f}",
                                'Recall': f"{metrics['recall']:.4f}",
                                'F1': f"{metrics['f1_score']:.4f}",
                                'ROC AUC': f"{metrics['roc_auc']:.4f}"
                            })
                        
                        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                        
                        # Radar chart
                        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                        fig = go.Figure()
                        
                        for model_name, result in results.items():
                            metrics = result['metrics']
                            values = [metrics[m] for m in metrics_to_plot]
                            
                            fig.add_trace(go.Scatterpolar(
                                r=values,
                                theta=[m.replace('_', ' ').title() for m in metrics_to_plot],
                                fill='toself',
                                name=model_name.replace('_', ' ').title()
                            ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=True,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except FileNotFoundError:
                        st.error("Data not found. Run Feature Engineering first!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Evaluate Models
    elif sub_page == "Evaluate Models":
        st.markdown('<div class="main-header"><h1>Evaluate Models</h1><p>Analyze performance</p></div>', unsafe_allow_html=True)
        
        if ds_name not in st.session_state.trained_models:
            st.warning("No models found. Train models first!")
        else:
            results = st.session_state.trained_models[ds_name]
            X_test_proc, y_test = st.session_state.test_data
            
            selected_model = st.selectbox(
                "Select Model",
                list(results.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if selected_model:
                result = results[selected_model]
                trainer = result['trainer']
                metrics = result['metrics']
                
                st.markdown("### Performance Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1", f"{metrics['f1_score']:.4f}")
                with col5:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
                
                # Confusion Matrix
                st.markdown("### Confusion Matrix")
                cm = np.array(metrics['confusion_matrix'])
                
                fig = go.Figure(go.Heatmap(
                    z=cm,
                    x=['Normal', 'Fraud'],
                    y=['Normal', 'Fraud'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 20}
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve
                st.markdown("### ROC Curve")
                y_pred_proba = trainer.predict_proba(X_test_proc)
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'ROC (AUC = {metrics["roc_auc"]:.4f})',
                    line=dict(color='#636EFA', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    name='Random', line=dict(color='gray', width=2, dash='dash')
                ))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                if selected_model in ['random_forest', 'xgboost']:
                    st.markdown("### Feature Importance")
                    importance_df = trainer.get_feature_importance()
                    
                    if importance_df is not None:
                        top_n = st.slider("Features to display", 5, 30, 15)
                        top_features = importance_df.head(top_n)
                        
                        fig = go.Figure(go.Bar(
                            x=top_features['importance'],
                            y=top_features['feature'],
                            orientation='h',
                            marker_color='#636EFA'
                        ))
                        fig.update_layout(
                            height=600,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Analytics Dashboard
    elif sub_page == "Analytics Dashboard":
        st.markdown('<div class="main-header"><h1>Analytics Dashboard</h1><p>Data insights</p></div>', unsafe_allow_html=True)
        
        if df is None:
            st.error("No data loaded")
        else:
            target_col = 'class' if 'class' in df.columns else 'Class'
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Transactions", f"{len(df):,}")
            with col2:
                fraud_count = df[target_col].sum()
                st.metric("Fraud", f"{fraud_count:,}")
            with col3:
                fraud_rate = (fraud_count / len(df)) * 100
                st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
            with col4:
                st.metric("Normal", f"{len(df) - fraud_count:,}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                class_dist = df[target_col].value_counts()
                fig = px.pie(
                    values=class_dist.values,
                    names=['Normal', 'Fraud'],
                    color_discrete_sequence=['#00CC96', '#EF553B']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(go.Bar(
                    x=['Normal', 'Fraud'],
                    y=class_dist.values,
                    marker_color=['#00CC96', '#EF553B'],
                    text=class_dist.values,
                    textposition='auto'
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            if ds_name == 'fraud' and 'purchase_value' in df.columns:
                st.markdown("### Purchase Value Analysis")
                fig = px.box(
                    df, x=target_col, y='purchase_value',
                    color=target_col,
                    color_discrete_sequence=['#00CC96', '#EF553B']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.predictions is not None:
                st.markdown("---")
                st.markdown("### Recent Predictions")
                
                pred_df = st.session_state.predictions
                
                col1, col2 = st.columns(2)
                with col1:
                    risk_dist = pred_df['risk_level'].value_counts()
                    fig = px.pie(values=risk_dist.values, names=risk_dist.index)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        pred_df, x='fraud_probability', nbins=50
                    )
                    st.plotly_chart(fig, use_container_width=True)
