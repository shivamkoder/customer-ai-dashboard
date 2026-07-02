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
                
                # --- Extract only the required columns (this avoids any extra columns like Customer ID) ---
                X_pred = df_pred[feature_cols].copy()
                
                # --- Clean 'Discount Applied' (convert to 0/1) ---
                X_pred['Discount Applied'] = X_pred['Discount Applied'].apply(
                    lambda x: 1 if x in [True, 'TRUE', 'True', 'true', 1, '1', 'Yes', 'yes', 'Y', 'y'] else 0
                )
                
                # --- Ensure numeric columns are actually numeric (coerce errors to NaN, which pipeline's imputer will handle) ---
                numeric_cols = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
                for col in numeric_cols:
                    X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')
                
                # --- The pipeline's imputer will handle any remaining NaNs, so we can proceed ---
                pred_encoded = model.predict(X_pred)
                probabilities = model.predict_proba(X_pred)
                pred_labels = label_encoder.inverse_transform(pred_encoded)
                
                # --- Store results back ---
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
                
                # --- Extract feature importance (if not already stored) ---
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
