import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from io import BytesIO
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Customer AI Manager", layout="wide")
st.title("📊 E-Commerce Customer Manager + AI Predictor")
st.markdown("Upload your CSVs, edit rows, add new customers, and let AI predict satisfaction!")

# ---------- LOAD MODEL (Cached) ----------
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('xgboost_satisfaction_pipeline.pkl')
        encoder = joblib.load('label_encoder.pkl')
        return pipeline, encoder
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'xgboost_satisfaction_pipeline.pkl' and 'label_encoder.pkl' are in the same directory as this app.")
        return None, None

model, label_encoder = load_model()

# ---------- INITIALIZE SESSION STATE ----------
if 'df' not in st.session_state:
    st.session_state.df = None

if 'predicted_cols' not in st.session_state:
    st.session_state.predicted_cols = []

# ---------- SIDEBAR: UPLOAD ----------
with st.sidebar:
    st.header("📁 Data Source")
    
    uploaded_files = st.file_uploader(
        "Upload CSV(s) to Merge",
        type=['csv'],
        accept_multiple_files=True,
        key="uploader"
    )
    
    if uploaded_files:
        if st.button("🔄 Merge & Load Uploaded Files"):
            try:
                dfs = []
                for file in uploaded_files:
                    df_temp = pd.read_csv(file)
                    dfs.append(df_temp)
                merged_df = pd.concat(dfs, ignore_index=True)
                
                # Ensure Customer ID exists
                if 'Customer ID' not in merged_df.columns:
                    merged_df.insert(0, 'Customer ID', range(1, len(merged_df) + 1))
                
                st.session_state.df = merged_df
                st.session_state.predicted_cols = []  # Clear old predictions
                st.success(f"✅ Merged {len(merged_df)} rows from {len(uploaded_files)} files!")
            except Exception as e:
                st.error(f"Error merging files: {e}")

    st.divider()
    st.caption("💡 **Instructions:**")
    st.caption("1. Upload CSV(s)")
    st.caption("2. Edit cells directly in the table below")
    st.caption("3. Click '+' at the bottom to add rows")
    st.caption("4. Click 'Re-run Predictions' to analyze changes")
    st.caption("5. Download the final merged CSV")

# ---------- MAIN CONTENT ----------
if st.session_state.df is not None:
    # --- Make a copy to display in editor ---
    # We need to drop prediction columns if they exist so they aren't edited (they are auto-generated)
    df_display = st.session_state.df.copy()
    display_cols = [col for col in df_display.columns if col not in ['Predicted Satisfaction', 'Confidence', 'Recommended Action']]
    df_display = df_display[display_cols]
    
    # --- DATA EDITOR (Excel-like experience) ---
    st.subheader("✏️ Edit Your Data (Add / Delete / Modify Rows)")
    st.caption("Click the '+' button at the bottom of the table to add a new customer. Click the 'x' on a row to delete it.")
    
    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",  # Enables adding and deleting rows
        use_container_width=True,
        key="data_editor"
    )
    
    # --- Update session state with edits ---
    st.session_state.df = edited_df
    
    # --- ACTION BUTTONS ROW ---
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🔮 Re-run Predictions", use_container_width=True):
            if model is not None:
                with st.spinner("Analyzing customers..."):
                    df_pred = st.session_state.df.copy()
                    
                    # Preprocess for prediction
                    if 'Discount Applied' in df_pred.columns:
                        df_pred['Discount Applied'] = df_pred['Discount Applied'].apply(
                            lambda x: 1 if x in [True, 'TRUE', 'True', 'true', 1, '1'] else 0
                        )
                    
                    # Prepare X
                    features = ['Gender', 'Age', 'City', 'Membership Type', 'Total Spend', 
                                'Items Purchased', 'Average Rating', 'Discount Applied', 'Days Since Last Purchase']
                    X_pred = df_pred[features].copy()
                    
                    # Predict
                    pred_encoded = model.predict(X_pred)
                    probabilities = model.predict_proba(X_pred)
                    
                    # Decode
                    pred_labels = label_encoder.inverse_transform(pred_encoded)
                    
                    # Assign back
                    st.session_state.df['Predicted Satisfaction'] = pred_labels
                    st.session_state.df['Confidence'] = probabilities.max(axis=1).round(2)
                    
                    # Actions
                    def get_action(row):
                        if row['Predicted Satisfaction'] == 'Unsatisfied':
                            return "🚨 Send 15% discount + support call"
                        elif row['Predicted Satisfaction'] == 'Neutral':
                            return "⚠️ Send 'We miss you' email + small perk"
                        else:
                            return "✅ Upsell premium bundle (loyal)"
                    
                    st.session_state.df['Recommended Action'] = st.session_state.df.apply(get_action, axis=1)
                    st.session_state.predicted_cols = ['Predicted Satisfaction', 'Confidence', 'Recommended Action']
                    st.success("✅ Predictions updated for all rows!")
            else:
                st.error("Model not loaded. Please check your .pkl files.")

    with col2:
        # Reset button to clear all data
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.df = None
            st.session_state.predicted_cols = []
            st.rerun()

    with col3:
        # Refresh editor (rerun app to reflect state)
        if st.button("🔄 Refresh View", use_container_width=True):
            st.rerun()

    # --- DASHBOARD METRICS (After Prediction) ---
    if 'Predicted Satisfaction' in st.session_state.df.columns:
        st.divider()
        st.subheader("📈 Real-time Dashboard")
        
        df_viz = st.session_state.df
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Customers", len(df_viz))
        with col_m2:
            st.metric("🟢 Satisfied", len(df_viz[df_viz['Predicted Satisfaction'] == 'Satisfied']))
        with col_m3:
            st.metric("🟡 Neutral", len(df_viz[df_viz['Predicted Satisfaction'] == 'Neutral']))
        with col_m4:
            st.metric("🔴 Churn Risk (Unsatisfied)", len(df_viz[df_viz['Predicted Satisfaction'] == 'Unsatisfied']))
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df_viz, names='Predicted Satisfaction', title='Satisfaction Breakdown', 
                         color='Predicted Satisfaction', 
                         color_discrete_map={'Satisfied':'#2ecc71', 'Neutral':'#f1c40f', 'Unsatisfied':'#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            # Show high risk customers
            risk_df = df_viz[df_viz['Predicted Satisfaction'] == 'Unsatisfied'][['Customer ID', 'City', 'Total Spend', 'Days Since Last Purchase']]
            st.write("🔴 **Top Churn Risks (Unsatisfied Customers)**")
            st.dataframe(risk_df.head(10), use_container_width=True)

    # --- FULL DATAFRAME VIEW (with predictions) ---
    st.divider()
    st.subheader("📋 Complete Customer List (With AI Insights)")
    
    # Styling for the final display
    if 'Predicted Satisfaction' in st.session_state.df.columns:
        def color_satisfaction(val):
            if val == 'Satisfied':
                return 'background-color: #d4edda'
            elif val == 'Neutral':
                return 'background-color: #fff3cd'
            elif val == 'Unsatisfied':
                return 'background-color: #f8d7da'
            return ''
        
        styled_df = st.session_state.df.style.applymap(color_satisfaction, subset=['Predicted Satisfaction'])
        st.dataframe(styled_df, use_container_width=True, height=500)
    else:
        st.dataframe(st.session_state.df, use_container_width=True, height=500)

    # --- DOWNLOAD SECTION ---
    st.divider()
    st.subheader("💾 Export Merged & Edited Data")
    
    # Prepare download
    output = BytesIO()
    st.session_state.df.to_csv(output, index=False)
    output.seek(0)
    
    st.download_button(
        label="⬇️ Download Final Merged CSV (Including Predictions)",
        data=output,
        file_name="merged_customer_data_with_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("👈 Please upload one or more CSV files in the sidebar to get started.")
    st.image("https://i.imgur.com/4r3JXj1.png", width=200)  # Optional placeholder
