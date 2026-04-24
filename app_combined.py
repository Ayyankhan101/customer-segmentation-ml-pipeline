import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG & THEME ---
CHART_COLORS = ['#667eea', '#764ba2', '#11998e', '#38ef7d', '#d29922']
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- STYLING ---
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main background and text */
    [data-testid="stAppViewContainer"] {
        background-color: #0f1419;
        color: #e7e9ea;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Header styling */
    .page-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Metric Card styling */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 2rem;
    }
    .m-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    .m-rev { background: linear-gradient(135deg, #238636, #2ea043); }
    .m-cust { background: linear-gradient(135deg, #1f6feb, #388bfd); }
    .m-nps { background: linear-gradient(135deg, #8957e5, #a371f7); }
    .m-churn { background: linear-gradient(135deg, #d29922, #e3b341); }
    
    .m-val { font-size: 2.2rem; margin-bottom: 0.2rem; }
    .m-lab { font-size: 0.9rem; opacity: 0.9; }

    /* Insight Box */
    .insight-box {
        background: #161b22;
        border-left: 4px solid #238636;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0 6px 6px 0;
        border: 1px solid #30363d;
        border-left-width: 5px;
    }
    
    /* Tables */
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    /* Sidebar adjustments */
    .css-17l243g { color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# --- PLOTTING UTILS ---
def set_plt_style():
    plt.rcParams.update({
        'figure.facecolor': '#161b22',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#8b949e',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'text.color': '#e7e9ea',
        'grid.color': '#30363d',
        'font.family': 'sans-serif'
    })

def plot_confusion_matrix_styled(cm, labels, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, pad=20, color='#58a6ff', fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

# --- DATA PROCESSING ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('customer_segments.csv')
    df_clean = df.dropna(subset=['segment'])
    feature_names = ['annual_spend', 'visits_per_month', 'items_per_order', 'support_tickets']
    X = df_clean[feature_names].copy()
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_names, index=X.index)
    le = LabelEncoder()
    y = le.fit_transform(df_clean['segment'])
    return df, X_imputed, y, le, feature_names

@st.cache_resource
def train_pipeline_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def main():
    apply_custom_css()
    set_plt_style()
    
    # Sidebar
    st.sidebar.markdown("<h2 style='color:#58a6ff;'>📊 ML Pipeline</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio("Navigate", [
        "Executive Dashboard", "Data Exploration", "Model Training", "Evaluation", 
        "Hyperparameter Tuning", "Prediction Tool"
    ])
    
    try:
        df, X, y, le, features = load_and_prep_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = train_pipeline_models(X_train_scaled, y_train)
        results = []
        for name, model in models.items():
            pred = model.predict(X_test_scaled)
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, pred),
                'Precision': precision_score(y_test, pred, average='weighted'),
                'Recall': recall_score(y_test, pred, average='weighted'),
                'F1 Score': f1_score(y_test, pred, average='weighted')
            })
        results_df = pd.DataFrame(results)

        if page == "Executive Dashboard":
            st.markdown('<div class="page-header"><h1>Executive Dashboard</h1></div>', unsafe_allow_html=True)
            
            # KPI Cards
            total_rev = df['annual_spend'].sum()
            avg_nps = 72 # Mock data to match HTML
            churn = 3.2 # Mock data to match HTML
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="m-card m-rev">
                    <div class="m-val">${total_rev/1e6:.2f}M</div>
                    <div class="m-lab">Total Revenue ▲ 12.5%</div>
                </div>
                <div class="m-card m-cust">
                    <div class="m-val">{len(df):,}</div>
                    <div class="m-lab">Total Customers ▲ 8.2%</div>
                </div>
                <div class="m-card m-nps">
                    <div class="m-val">{avg_nps}</div>
                    <div class="m-lab">NPS Score ▲ 5pts</div>
                </div>
                <div class="m-card m-churn">
                    <div class="m-val">{churn}%</div>
                    <div class="m-lab">Churn Rate ▼ 0.8%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Revenue Trend")
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                rev_vals = [120, 145, 168, 172, 189, 205, 218, 225, 238, 252, 268, 290]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(months, rev_vals, marker='o', color='#667eea', linewidth=3, markersize=8)
                ax.fill_between(months, rev_vals, alpha=0.2, color='#667eea')
                ax.set_ylabel("Revenue ($K)")
                st.pyplot(fig)
                
            with col2:
                st.subheader("Revenue by Segment")
                seg_rev = df.groupby('segment')['annual_spend'].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.pie(seg_rev, labels=seg_rev.index, autopct='%1.1f%%', colors=CHART_COLORS, startangle=140, wedgeprops={'edgecolor': '#161b22'})
                st.pyplot(fig)

            st.markdown("### Model Performance Overview")
            st.table(results_df.style.format({
                'Accuracy': '{:.1%}', 'Precision': '{:.1%}', 'Recall': '{:.1%}', 'F1 Score': '{:.1%}'
            }))

        elif page == "Data Exploration":
            st.markdown('<div class="page-header"><h1>Data Overview & EDA</h1></div>', unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["📊 Distributions", "🔗 Correlations"])
            
            with tab1:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("Segment Distribution")
                    seg_counts = df['segment'].value_counts()
                    fig, ax = plt.subplots()
                    ax.bar(seg_counts.index, seg_counts.values, color=CHART_COLORS)
                    st.pyplot(fig)
                with col2:
                    st.subheader("Dataset Sample")
                    st.dataframe(df.head(10), height=300)
                
                st.subheader("Feature Distributions")
                cols = st.columns(len(features))
                for i, feat in enumerate(features):
                    with cols[i]:
                        fig, ax = plt.subplots()
                        sns.histplot(df[feat], kde=True, color=CHART_COLORS[i % 4], ax=ax)
                        ax.set_title(feat)
                        st.pyplot(fig)

            with tab2:
                st.subheader("Feature Correlation Matrix")
                corr = X.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, ax=ax)
                st.pyplot(fig)

        elif page == "Model Training":
            st.markdown('<div class="page-header"><h1>Model Building</h1></div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-box">
                <b>Training Configuration:</b><br>
                - Total Samples: {len(X)}<br>
                - Features: {', '.join(features)}<br>
                - Split: 80% Train / 20% Test (Stratified)
            </div>
            """, unsafe_allow_html=True)
            
            for name, model in models.items():
                with st.expander(f"Details: {name}"):
                    if hasattr(model, 'feature_importances_'):
                        imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                        st.bar_chart(imp.set_index('Feature'))
                    else:
                        st.write("Linear coefficients used for this model.")

        elif page == "Evaluation":
            st.markdown('<div class="page-header"><h1>Model Evaluation</h1></div>', unsafe_allow_html=True)
            
            selected_model = st.selectbox("Select Model to Evaluate", list(models.keys()))
            model = models[selected_model]
            pred = model.predict(X_test_scaled)
            
            col1, col2 = st.columns(2)
            with col1:
                cm = confusion_matrix(y_test, pred)
                st.pyplot(plot_confusion_matrix_styled(cm, le.classes_, f"{selected_model} Confusion Matrix"))
            
            with col2:
                st.subheader("Classification Report")
                st.code(classification_report(y_test, pred, target_names=le.classes_))
                
                st.markdown("### Key Metrics")
                m1, m2 = st.columns(2)
                m1.metric("Accuracy", f"{accuracy_score(y_test, pred):.1%}")
                m2.metric("F1 Score", f"{f1_score(y_test, pred, average='weighted'):.1%}")

        elif page == "Hyperparameter Tuning":
            st.markdown('<div class="page-header"><h1>Hyperparameter Tuning</h1></div>', unsafe_allow_html=True)
            st.info("Optimization process for Random Forest Classifier (Best Performer)")
            
            if st.button("Run GridSearchCV"):
                with st.spinner("Searching for optimal parameters..."):
                    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, None], 'min_samples_split': [2, 5]}
                    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
                    gs.fit(X_train_scaled, y_train)
                    st.success(f"Optimal Parameters: {gs.best_params_}")
                    st.metric("Best Cross-Val Score", f"{gs.best_score_:.2%}")

        elif page == "Prediction Tool":
            st.markdown('<div class="page-header"><h1>Prediction Tool</h1></div>', unsafe_allow_html=True)
            
            with st.form("pred_form"):
                c1, c2, c3, c4 = st.columns(4)
                f1 = c1.number_input("Annual Spend ($)", value=1500.0)
                f2 = c2.number_input("Visits/Month", value=5)
                f3 = c3.number_input("Items/Order", value=3)
                f4 = c4.number_input("Support Tickets", value=1)
                
                submitted = st.form_submit_button("Predict Segment", use_container_width=True)
                
                if submitted:
                    input_data = scaler.transform([[f1, f2, f3, f4]])
                    res = models['Random Forest'].predict(input_data)[0]
                    prob = models['Random Forest'].predict_proba(input_data)[0]
                    
                    st.markdown(f"### Predicted Segment: <span style='color:#38ef7d'>{le.inverse_transform([res])[0]}</span>", unsafe_allow_html=True)
                    
                    prob_df = pd.DataFrame({'Segment': le.classes_, 'Probability': prob})
                    st.bar_chart(prob_df.set_index('Segment'))

    except Exception as e:
        st.error(f"Failed to load data: {e}")

if __name__ == "__main__":
    main()
