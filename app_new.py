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

def load_data():
    df = pd.read_csv('customer_segments.csv')
    return df

def prepare_data(df):
    df_clean = df.dropna(subset=['segment'])
    feature_names = ['annual_spend', 'visits_per_month', 'items_per_order', 'support_tickets']
    X = df_clean[feature_names].copy()
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_names, index=X.index)
    le = LabelEncoder()
    y = le.fit_transform(df_clean['segment'])
    return X_imputed, y, le, feature_names

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def evaluate_classification(y_true, y_pred, model_name):
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
