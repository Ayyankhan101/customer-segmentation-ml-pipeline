import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

def plot_confusion_matrix(cm, labels, title, ax):
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)))
    ax.set_xticklabels(labels), ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

def plot_feature_importance(model, feature_names, title):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='#11998e')
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
    
    return fig

def plot_model_comparison(results_df):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(results_df))
    width = 0.2
    colors = ['#667eea', '#764ba2', '#11998e', '#38ef7d']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, results_df[metric], width, label=metric, color=colors[i])
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    return fig

def plot_feature_importance(model, feature_names, title):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='#11998e')
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
    
    return fig

def plot_model_comparison(results_df):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(results_df))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#667eea', '#764ba2', '#11998e', '#38ef7d']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, results_df[metric], width, label=metric, color=colors[i])
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    return fig

def plot_metric_comparison(before_df, after_df, metric):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = before_df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, before_df[metric], width, label='Before Tuning', color='#e74c3c')
    bars2 = ax.bar(x + width/2, after_df[metric], width, label='After Tuning', color='#27ae60')
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Before vs After Tuning')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.set_ylim(0, min(1.1, max(before_df[metric].max(), after_df[metric].max()) * 1.2))
    
    return fig

def main():
    st.sidebar.title("📊 ML Pipeline")
    
    df = load_data()
    X, y, label_encoder, feature_names = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    trained_models = train_models(X_train_scaled, y_train)
    predictions = {name: model.predict(X_test_scaled) for name, model in trained_models.items()}
    
    results = []
    for name, pred in predictions.items():
        results.append(evaluate_classification(y_test, pred, name))
    results_df = pd.DataFrame(results)
    
    page = st.sidebar.radio("Navigate", [
        "Home", "Data Overview", "Model Building", "Evaluation", 
        "Hyperparameter Tuning", "Comparison", "Prediction API"
    ])
    
    segment_labels = label_encoder.classes_
    
    if page == "Home":
        st.markdown('<div class="page-header"><h1>📊 ML Classification Pipeline</h1></div>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        best_idx = results_df['F1 Score'].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_f1 = results_df.loc[best_idx, 'F1 Score']
        
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Samples</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{len(feature_names)}</div>
                <div class="metric-label">Features</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{len(segment_labels)}</div>
                <div class="metric-label">Classes</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{best_f1:.2%}</div>
                <div class="metric-label">Best F1 Score</div></div>""", unsafe_allow_html=True)
        
        st.markdown("### 📋 Pipeline Summary")
        
        st.markdown(f"""
        <div class="insight-box">
        <b>Problem Type:</b> Multi-class Classification<br>
        <b>Target Variable:</b> segment (Standard, Inactive, Basic, Premium)<br>
        <b>Features:</b> annual_spend, visits_per_month, items_per_order, support_tickets<br>
        <b>Train/Test Split:</b> 80/20 (stratified)<br>
        <b>Models Trained:</b> Logistic Regression, Decision Tree, Random Forest
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🏆 Best Model")
        st.success(f"**{best_model}** achieves the highest F1 Score of **{best_f1:.2%}**")
        
        st.markdown("### 📈 Quick Performance Overview")
        st.dataframe(results_df.set_index('Model').style.background_gradient(
            subset=['Accuracy', 'F1 Score'], cmap='Greens'
        ), use_container_width=True)
    
    elif page == "Data Overview":
        st.markdown('<div class="page-header"><h1>📈 Data Overview & EDA</h1></div>', 
                   unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Dataset", "Distributions", "Correlation"])
        
        with tab1:
            st.subheader("Raw Dataset Sample")
            st.dataframe(df.head(10))
            
            st.subheader("Cleaned Dataset Statistics")
            st.dataframe(X.describe(), use_container_width=True)
            
            st.subheader("Class Distribution")
            class_dist = df['segment'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#667eea', '#764ba2', '#11998e', '#38ef7d']
            ax.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', 
                  colors=colors, explode=[0.02]*len(class_dist))
            ax.set_title("Segment Distribution")
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Feature Distributions")
            for col in X.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(X[col], bins=30, color='#11998e', edgecolor='white', alpha=0.7)
                ax.axvline(X[col].mean(), color='red', linestyle='--', label=f'Mean: {X[col].mean():.2f}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.set_title(f"{col} Distribution")
                ax.legend()
                st.pyplot(fig)
            
            st.subheader("Segment-wise Feature Boxplots")
            dfviz = X.copy()
            dfviz['segment'] = label_encoder.inverse_transform(y)
            
            for col in X.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                dfviz.boxplot(column=col, by='segment', ax=ax)
                ax.set_xlabel('Segment')
                ax.set_ylabel(col)
                ax.set_title(f"{col} by Segment")
                plt.suptitle('')
                st.pyplot(fig)
        
        with tab3:
            st.subheader("Correlation Heatmap")
            corr = X.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr.values, interpolation='nearest', cmap='RdYlGn')
            ax.set(xticks=np.arange(len(corr.columns)), yticks=np.arange(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45), ax.set_yticklabels(corr.columns)
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center',
                           color='white' if abs(corr.values[i, j]) > 0.5 else 'black')
            ax.set_title("Feature Correlation Matrix")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            
            st.subheader("Key Insights")
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [(i, j, upper.loc[i, j]) for i in upper.index 
                       for j in upper.columns if abs(upper.loc[i, j]) > 0.3]
            
            if high_corr:
                for f1, f2, corr_val in high_corr:
                    st.info(f"**{f1} ↔ {f2}**: {corr_val:.3f}")
            else:
                st.info("No strong correlations (>0.3) found between features.")
    
    elif page == "Model Building":
        st.markdown('<div class="page-header"><h1>🤖 Model Building</h1></div>', 
                   unsafe_allow_html=True)
        
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        
        st.info("""
        **Models Implemented:**
        1. **Logistic Regression** - Linear classifier, good for linearly separable data
        2. **Decision Tree** - Non-parametric, handles non-linear relationships
        3. **Random Forest** - Ensemble method, reduces overfitting
        """)
        
        st.subheader("Model Coefficients / Importances")
        
        for name, model in trained_models.items():
            st.markdown(f"**{name}**")
            
            if hasattr(model, 'feature_importances_'):
                imp = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.dataframe(imp, hide_index=True)
            else:
                imp = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': model.coef_[0]
                }).sort_values('Coefficient', ascending=False)
                st.dataframe(imp, hide_index=True)
            
            st.markdown("---")
    
    elif page == "Evaluation":
        st.markdown('<div class="page-header"><h1>📊 Model Evaluation</h1></div>', 
                   unsafe_allow_html=True)
        
        st.subheader("Performance Metrics Summary")
        st.dataframe(results_df.set_index('Model').style.background_gradient(
            subset=['Accuracy', 'F1 Score'], cmap='Greens'
        ), use_container_width=True)
        
        for name, pred in predictions.items():
            st.markdown(f"### {name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, pred):.2%}")
            
            with col2:
                st.metric("F1 Score (weighted)", f"{f1_score(y_test, pred, average='weighted'):.2%}")
            
            cm = confusion_matrix(y_test, pred)
            fig = plot_confusion_matrix(cm, segment_labels, f"{name} Confusion Matrix")
            st.pyplot(fig)
            
            st.code(classification_report(y_test, pred, target_names=segment_labels))
            st.markdown("---")
        
        st.subheader("Feature Importance (All Models)")
        for name, model in trained_models.items():
            fig = plot_feature_importance(model, feature_names, f"{name} Feature Importance")
            st.pyplot(fig)
    
    elif page == "Hyperparameter Tuning":
        st.markdown('<div class="page-header"><h1>⚙️ Hyperparameter Tuning</h1></div>', 
                   unsafe_allow_html=True)
        
        st.info("Applying GridSearchCV on Random Forest (best performer)")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        with st.spinner("Running GridSearchCV (this may take a moment)..."):
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1_weighted', 
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        st.success(f"Best Parameters: {best_params}")
        st.success(f"Best CV Score: {best_score:.2%}")
        
        rf_tuned = grid_search.best_estimator_
        y_pred_tuned = rf_tuned.predict(X_test_scaled)
        
        tuned_metrics = evaluate_classification(y_test, y_pred_tuned, "Random Forest (Tuned)")
        
        st.subheader("Random Forest: Before vs After Tuning")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{results_df.loc[2, 'Accuracy']:.2%}", 
                     f"{(tuned_metrics['Accuracy'] - results_df.loc[2, 'Accuracy']):.2%}")
        with col2:
            st.metric("Precision", f"{results_df.loc[2, 'Precision']:.2%}",
                     f"{(tuned_metrics['Precision'] - results_df.loc[2, 'Precision']):.2%}")
        with col3:
            st.metric("Recall", f"{results_df.loc[2, 'Recall']:.2%}",
                     f"{(tuned_metrics['Recall'] - results_df.loc[2, 'Recall']):.2%}")
        with col4:
            st.metric("F1 Score", f"{results_df.loc[2, 'F1 Score']:.2%}",
                     f"{(tuned_metrics['F1 Score'] - results_df.loc[2, 'F1 Score']):.2%}")
        
        before_vs_after = pd.DataFrame([
            {**results_df.loc[2].to_dict()},
            {**tuned_metrics}
        ])
        
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            fig = plot_metric_comparison(
                pd.DataFrame([results_df.loc[2].to_dict()]),
                pd.DataFrame([tuned_metrics]),
                metric
            )
            st.pyplot(fig)
        
        cm_tuned = confusion_matrix(y_test, y_pred_tuned)
        fig = plot_confusion_matrix(cm_tuned, segment_labels, "Tuned Random Forest Confusion Matrix")
        st.pyplot(fig)
    
    elif page == "Comparison":
        st.markdown('<div class="page-header"><h1>📊 Model Comparison</h1></div>', 
                   unsafe_allow_html=True)
        
        st.subheader("Performance Metrics Comparison")
        
        fig = plot_model_comparison(results_df)
        st.pyplot(fig)
        
        st.subheader("Detailed Comparison Table")
        st.dataframe(results_df.set_index('Model').style.background_gradient(
            subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], cmap='Greens'
        ), use_container_width=True)
        
        st.subheader("Analysis")
        
        best_idx = results_df['F1 Score'].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_accuracy = results_df.loc[best_idx, 'Accuracy']
        worst_idx = results_df['F1 Score'].idxmin()
        worst_model = results_df.loc[worst_idx, 'Model']
        
        for idx, row in results_df.iterrows():
            status = []
            if row['Accuracy'] > 0.9:
                status.append("High accuracy")
            if row['F1 Score'] < 0.7:
                status.append("May underfit")
            
            st.markdown(f"**{row['Model']}**: {', '.join(status) if status else 'Balanced'}")
        
        st.markdown(f"""
        <div class="insight-box">
        <b>Best Model:</b> {best_model} (F1: {results_df.loc[best_idx, 'F1 Score']:.2%})<br>
        <b>Insight:</b> Random Forest typically outperforms due to ensemble averaging,
        which reduces variance and handles non-linear relationships better than 
        Logistic Regression.
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Prediction API":
        st.markdown('<div class="page-header"><h1>🔮 Prediction API</h1></div>', 
                   unsafe_allow_html=True)
        
        st.info("FastAPI Endpoint: POST /predict")
        
        st.subheader("Test Prediction")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            annual_spend = st.number_input("Annual Spend ($)", min_value=0.0, value=1500.0, step=100.0)
        with col2:
            visits = st.number_input("Visits/Month", min_value=0, value=5, step=1)
        with col3:
            items = st.number_input("Items/Order", min_value=1, value=3, step=1)
        with col4:
            tickets = st.number_input("Support Tickets", min_value=0, value=1, step=1)
        
        if st.button("Predict Segment"):
            input_data = np.array([[annual_spend, visits, items, tickets]])
            input_scaled = scaler.transform(input_data)
            
            rf_pred = trained_models['Random Forest'].predict(input_scaled)[0]
            dt_pred = trained_models['Decision Tree'].predict(input_scaled)[0]
            lr_pred = trained_models['Logistic Regression'].predict(input_scaled)[0]
            
            st.success(f"**Random Forest**: {label_encoder.inverse_transform([rf_pred])[0]}")
            st.success(f"**Decision Tree**: {label_encoder.inverse_transform([dt_pred])[0]}")
            st.success(f"**Logistic Regression**: {label_encoder.inverse_transform([lr_pred])[0]}")
        
        st.markdown("---")
        
        st.subheader("API Code")
        
        api_code = '''from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

class CustomerData(BaseModel):
    annual_spend: float
    visits_per_month: int
    items_per_order: int
    support_tickets: int

@app.post("/predict")
def predict(data: CustomerData):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Load and fit model (or load from pickle in production)
    
    features = [[data.annual_spend, data.visits_per_month, 
                data.items_per_order, data.support_tickets]]
    
    prediction = model.predict(features)
    segments = ["Basic", "Inactive", "Premium", "Standard"]
    
    return {"prediction": segments[prediction[0]]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        st.code(api_code, language="python")
        
        st.markdown("""
        **To run API:**
        ```bash
        pip install fastapi uvicorn
        python -m uvicorn app:app --reload --port 8000
        ```
        
        **Test with curl:**
        ```bash
        curl -X POST http://localhost:8000/predict \\
             -H "Content-Type: application/json" \\
             -d '{"annual_spend": 1500, "visits_per_month": 5, 
                 "items_per_order": 3, "support_tickets": 1}'
        ```
        """)

if __name__ == "__main__":
    main()