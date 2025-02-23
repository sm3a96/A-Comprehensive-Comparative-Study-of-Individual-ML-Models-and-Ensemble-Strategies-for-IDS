import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder
import time

# ---------------------------- Helper Functions ----------------------------
def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }


def plot_confusion_matrix(y_true, y_pred, classes, title):
    """Plot and display a confusion matrix."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def data_preprocessing():
    # ---------------------------- Data Preprocessing ----------------------------
    df = pd.read_csv('/home/ibibers/IDS Project/IDS_Datasets/Combined_datasets/Simargelpreprocessed_dataset_with_original_labels.csv')

    # Remove duplicates and unnecessary columns
    df = df.drop_duplicates()
    # df.rename(columns=lambda x: x.lstrip(), inplace=True)
    
    # Label encoding for the target variable
    le = LabelEncoder()
    df['ALERT'] = le.fit_transform(df['ALERT'])

    dropped_cols = ['ALERT']
    X = df.drop(columns=dropped_cols, axis=1)
    y = df['ALERT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ---------------------------- Feature Selection ----------------------------
    # Top features selected using Information Gain
    IGtop_5_features = ['IPV4_SRC_ADDR', 'TCP_FLAGS', 'IPV4_DST_ADDR', 'IN_BYTES', 'FLOW_ID'] 
    IGtop_10_features = IGtop_5_features + [ 'TOTAL_FLOWS_EXP', 'TCP_WIN_MAX_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'FLOW_DURATION_MILLISECONDS']
              
    # Top features selected using K-best
    Kbest_top_5_features = ['TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN', 'TCP_WIN_SCALE_IN', 'IPV4_DST_ADDR']
    Kbest_top_10_features = Kbest_top_5_features + ['PROTOCOL', 'TOTAL_FLOWS_EXP', 'FLOW_ID', 'ANALYSIS_TIMESTAMP', 'FIRST_SWITCHED']

    # Subset data based on feature selection
    datasets = {
        "All Features": (X_train, X_test , y_train, y_test),
        "IG Top 5 Features": (X_train[IGtop_5_features], X_test[IGtop_5_features], y_train, y_test),
        "IG Top 10 Features": (X_train[IGtop_10_features], X_test[IGtop_10_features],  y_train, y_test),
        "KBest Top 5 Features": (X_train[Kbest_top_5_features], X_test[Kbest_top_5_features], y_train, y_test),
        "KBest Top 10 Features": (X_train[Kbest_top_10_features], X_test[Kbest_top_10_features], y_train, y_test),
    }
    return datasets , le



