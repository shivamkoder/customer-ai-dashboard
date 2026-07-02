import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

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
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

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
                
                if 'Customer ID' not in merged_df.columns:
                    merged_df.insert(0, 'Customer ID', range(1, len(merged_df) + 1))
                
                st.session_state.df = merged_df
                st.session_state.predicted_cols = []
                st.session_state.feature_importance = None
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
    df_display = st.session_state.df.copy()
    display_cols = [col for col in df_display.columns if col not in ['Predicted Satisfaction', 'Confidence', 'Recommended Action']]
    df_display = df_display[display_cols]
    
    st.subheader("✏️ Edit Your Data (Add / Delete / Modify Rows)")
    st.caption("Click the '+' button at the bottom of the table to add a new customer. Click the 'x' on a row to delete it.")
    
    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
    
    st.session_state.df = edited_df
    
    # ---------- ACTION BUTTONS ----------
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔮 Re-run Predictions", use_container_width=True):
            if model is not None:
                with st.spinner("Analyzing customers..."):
                    try:
                        df_pred = st.session_state.df.copy()
                        
                        # --- Define the exact feature columns expected by the model ---
                        feature_cols = [
                            'Gender', 'Age', 'City', 'Membership Type', 'Total Spend',
                            'Items Purchased', 'Average Rating', 'Discount Applied', 'Days Since Last Purchase'
                        ]
                        
                        # --- Check if all required columns exist ---
                        missing_cols = [col for col in feature_cols if col not in df_pred.columns]
                        if missing_cols:
                            st.error(f"❌ Missing required columns: {missing_cols}. Please upload a CSV that includes all these columns.")
                            st.stop()
                        
                        # --- Extract only the required columns ---
                        X_pred = df_pred[feature_cols].copy()
                        
                        # --- Clean 'Discount Applied' ---
                        X_pred['Discount Applied'] = X_pred['Discount Applied'].apply(
                            lambda x: 1 if x in [True, 'TRUE', 'True', 'true', 1, '1', 'Yes', 'yes', 'Y', 'y'] else 0
                        )
                        
                        # --- Ensure numeric columns are numeric ---
                        numeric_cols = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
                        for col in numeric_cols:
                            X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')
                        
                        # --- Predict ---
                        pred_encoded = model.predict(X_pred)
                        probabilities = model.predict_proba(X_pred)
                        pred_labels = label_encoder.inverse_transform(pred_encoded)
                        
                        # --- Store results ---
                        st.session_state.df['Predicted Satisfaction'] = pred_labels
                        st.session_state.df['Confidence'] = probabilities.max(axis=1).round(2)
                        
                        def get_action(row):
                            if row['Predicted Satisfaction'] == 'Unsatisfied':
                                return "🚨 Send 15% discount + support call"
                            elif row['Predicted Satisfaction'] == 'Neutral':
                                return "⚠️ Send 'We miss you' email + small perk"
                            else:
                                return "✅ Upsell premium bundle (loyal)"
                        
                        st.session_state.df['Recommended Action'] = st.session_state.df.apply(get_action, axis=1)
                        st.session_state.predicted_cols = ['Predicted Satisfaction', 'Confidence', 'Recommended Action']
                        
                        # --- Extract feature importance ---
                        try:
                            xgb_model = model.named_steps['classifier']
                            preprocessor = model.named_steps['preprocessor']
                            num_feat_names = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
                            cat_feat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['Gender', 'City', 'Membership Type'])
                            bin_feat_names = ['Discount Applied']
                            all_feat_names = list(num_feat_names) + list(cat_feat_names) + bin_feat_names
                            importances = xgb_model.feature_importances_
                            st.session_state.feature_importance = sorted(
                                zip(all_feat_names, importances),
                                key=lambda x: x[1],
                                reverse=True
                            )
                        except Exception as e:
                            st.warning(f"Could not extract feature importance: {e}")
                        
                        st.success("✅ Predictions updated for all rows!")
                        
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        st.info("Please check that your data types are correct (e.g., Age, Spend are numbers) and that all required columns are present.")
            else:
                st.error("Model not loaded. Please ensure the .pkl files are in the app directory.")

    with col2:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.df = None
            st.session_state.predicted_cols = []
            st.session_state.feature_importance = None
            st.rerun()

    # ---------- DISPLAY DASHBOARD & INSIGHTS (IF PREDICTIONS EXIST) ----------
    if 'Predicted Satisfaction' in st.session_state.df.columns:
        df_viz = st.session_state.df
        
        # ---- 1. METRICS ----
        st.divider()
        st.subheader("📈 Real-time Dashboard")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Customers", len(df_viz))
        with col_m2:
            st.metric("🟢 Satisfied", len(df_viz[df_viz['Predicted Satisfaction'] == 'Satisfied']))
        with col_m3:
            st.metric("🟡 Neutral", len(df_viz[df_viz['Predicted Satisfaction'] == 'Neutral']))
        with col_m4:
            st.metric("🔴 Churn Risk (Unsatisfied)", len(df_viz[df_viz['Predicted Satisfaction'] == 'Unsatisfied']))
        
        # ---- 2. CHARTS ----
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df_viz, names='Predicted Satisfaction', title='Satisfaction Breakdown', 
                         color='Predicted Satisfaction', 
                         color_discrete_map={'Satisfied':'#2ecc71', 'Neutral':'#f1c40f', 'Unsatisfied':'#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            risk_df = df_viz[df_viz['Predicted Satisfaction'] == 'Unsatisfied'][['Customer ID', 'City', 'Total Spend', 'Days Since Last Purchase']]
            st.write("🔴 **Top Churn Risks (Unsatisfied Customers)**")
            st.dataframe(risk_df.head(10), use_container_width=True)

        # ------------------------------------------------------------
        # KEY BUSINESS INSIGHTS SECTION
        # ------------------------------------------------------------
        st.divider()
        st.subheader("📊 Key Business Insights (What Drives Satisfaction?)")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Top 5 Drivers", "👤 Customer Profiles", "🌍 City & Membership Risk", "📉 Discount Impact"])
        
        # --- TAB 1: FEATURE IMPORTANCE ---
        with tab1:
            if st.session_state.feature_importance:
                top_10 = st.session_state.feature_importance[:10]
                names = [item[0] for item in top_10]
                values = [item[1] for item in top_10]
                
                # Clean up feature names for display
                display_names = []
                for name in names:
                    if '_' in name and name.split('_')[0] in ['Gender', 'City', 'Membership Type']:
                        parts = name.split('_')
                        display_names.append(f"{parts[0]}: {parts[1]}")
                    else:
                        display_names.append(name)
                
                fig_imp = go.Figure(go.Bar(
                    x=values,
                    y=display_names,
                    orientation='h',
                    marker=dict(
                        color=['#e74c3c' if i < 5 else '#3498db' for i in range(len(values))]
                    ),
                    text=[f"{v*100:.1f}%" for v in values],
                    textposition='outside'
                ))
                fig_imp.update_layout(
                    title="<b>Top 10 Factors Influencing Customer Satisfaction</b>",
                    xaxis_title="Importance Score (Higher = More Impact)",
                    yaxis_title="Feature",
                    height=450,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
                if len(st.session_state.feature_importance) >= 5:
                    top1 = st.session_state.feature_importance[0][0]
                    top2 = st.session_state.feature_importance[1][0]
                    st.success(f"""
                    💡 **Executive Summary**: 
                    1. **{top1}** is the #1 predictor of satisfaction. 
                    2. **{top2}** is the second most important.
                    Focus your retention efforts on improving **{top1}** and **{top2}** before sending discounts.
                    """)
            else:
                st.info("Re-run predictions to see the top drivers of satisfaction.")

        # --- TAB 2: CUSTOMER PROFILES ---
        with tab2:
            st.subheader("👤 What does the 'Average' Customer Look Like?")
            
            profile_cols = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
            profile_df = df_viz.groupby('Predicted Satisfaction')[profile_cols].mean().reset_index()
            st.dataframe(
                profile_df.style.background_gradient(cmap='RdYlGn', subset=['Average Rating', 'Total Spend']),
                use_container_width=True
            )
            
            fig_profile = px.bar(
                profile_df, 
                x='Predicted Satisfaction', 
                y=['Total Spend', 'Average Rating'], 
                barmode='group',
                title="Spending & Rating by Satisfaction Level",
                labels={'value': 'Average Value', 'variable': 'Metric'},
                color_discrete_map={'Total Spend': '#3498db', 'Average Rating': '#2ecc71'}
            )
            st.plotly_chart(fig_profile, use_container_width=True)

        # --- TAB 3: CITY & MEMBERSHIP RISK ---
        with tab3:
            st.subheader("🌍 Which Cities / Memberships are Most At-Risk?")
            
            cross_tab = pd.crosstab(
                [df_viz['City'], df_viz['Membership Type']], 
                df_viz['Predicted Satisfaction'],
                normalize='index'
            ) * 100
            cross_tab_reset = cross_tab.reset_index()
            cross_tab_melted = cross_tab_reset.melt(id_vars=['City', 'Membership Type'], 
                                                     var_name='Satisfaction', 
                                                     value_name='Percentage')
            risk_heat = cross_tab_melted[cross_tab_melted['Satisfaction'] == 'Unsatisfied']
            
            if not risk_heat.empty:
                fig_heat = px.bar(
                    risk_heat,
                    x='City',
                    y='Percentage',
                    color='Membership Type',
                    title="% of Unsatisfied Customers by City & Membership",
                    labels={'Percentage': '% Unsatisfied'},
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                st.dataframe(risk_heat.sort_values('Percentage', ascending=False), use_container_width=True)
            else:
                st.info("Not enough data to display city/membership breakdown.")

        # --- TAB 4: DISCOUNT IMPACT ---
        with tab4:
            st.subheader("📉 Do Discounts Actually Improve Satisfaction?")
            
            df_viz['Discount Applied'] = df_viz['Discount Applied'].astype(int)
            discount_impact = pd.crosstab(df_viz['Discount Applied'], df_viz['Predicted Satisfaction'], normalize='index') * 100
            discount_impact = discount_impact.reset_index()
            discount_impact['Discount Applied'] = discount_impact['Discount Applied'].map({0: 'No Discount', 1: 'Discount Given'})
            discount_melted = discount_impact.melt(id_vars=['Discount Applied'], 
                                                   var_name='Satisfaction', 
                                                   value_name='Percentage')
            
            fig_discount = px.bar(
                discount_melted,
                x='Discount Applied',
                y='Percentage',
                color='Satisfaction',
                title="Satisfaction Distribution: Discount vs No Discount",
                barmode='group',
                color_discrete_map={'Satisfied':'#2ecc71', 'Neutral':'#f1c40f', 'Unsatisfied':'#e74c3c'},
                labels={'Percentage': '% of Customers'}
            )
            st.plotly_chart(fig_discount, use_container_width=True)
            
            st.warning("""
            ⚠️ **Insight**: Customers who received discounts are often **MORE unsatisfied** than those who didn't. 
            This confirms what our model found: **discounts are reactive (given to unhappy customers)** and do not drive loyalty. 
            Action: Shift budget from broad discounts to fixing product quality / support for your top drivers!
            """)

        # --- CONTINUE WITH FULL DATAFRAME ---
        st.divider()
        st.subheader("📋 Complete Customer List (With AI Insights)")
        
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

        # --- DOWNLOAD ---
        st.divider()
        st.subheader("💾 Export Merged & Edited Data")
        
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
        # If data is loaded but not yet predicted
        st.info("Click 'Re-run Predictions' to analyze your data.")

else:
    st.info("👈 Please upload one or more CSV files in the sidebar to get started.")
