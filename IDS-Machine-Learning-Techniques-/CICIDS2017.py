import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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
        
    # file_names = ['/home/ibibers@ads.iu.edu/Intrusion_Detection_System_IDS/IDS_Datasets/CICIDS2017_Dataset/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    #                '/home/ibibers@ads.iu.edu/Intrusion_Detection_System_IDS/IDS_Datasets/CICIDS2017_Dataset/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    #                  '/home/ibibers@ads.iu.edu/Intrusion_Detection_System_IDS/IDS_Datasets/CICIDS2017_Dataset/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    #                  '/home/ibibers@ads.iu.edu/Intrusion_Detection_System_IDS/IDS_Datasets/CICIDS2017_Dataset/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv',
    #                  '/home/ibibers@ads.iu.edu/Intrusion_Detection_System_IDS/IDS_Datasets/CICIDS2017_Dataset/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    #                  '/home/ibibers@ads.iu.edu/Intrusion_Detection_System_IDS/IDS_Datasets/CICIDS2017_Dataset/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    #                  '/home/ibibers@ads.iu.edu/Intrusion_Detection_System_IDS/IDS_Datasets/CICIDS2017_Dataset/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv',
    #                  '/home/ibibers@ads.iu.edu/Intrusion_Detection_System_IDS/IDS_Datasets/CICIDS2017_Dataset/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv']

    # dataframes = []

    # for file_name in file_names:
    #     df = pd.read_csv(file_name)
    #     dataframes.append(df)

    # # Concatenate all DataFrames into a single DataFrame
    # combined_df = pd.concat(dataframes, ignore_index=True)
    # # Save the combined DataFrame to a CSV file
    # combined_df.to_csv('combined_dataset.csv', index=False)



    df = pd.read_csv('/home/ibibers@ads.iu.edu/IDS_Datasets/Combined_datasets/CICIDS2017_combined_dataset.csv')

    # Remove duplicates and unnecessary columns
    df = df.drop_duplicates()
    df.rename(columns=lambda x: x.lstrip(), inplace=True)
    df['Flow Bytes/s'] = df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean())

    # Label encoding for the target variable
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])

    dropped_cols = ['Label', 'Flow Packets/s', 'Flow Bytes/s']
    X = df.drop(columns=dropped_cols, axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ---------------------------- Feature Selection ----------------------------
    # Top features selected using Information Gain
    IGtop_5_features = ['Average Packet Size', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'Total Length of Bwd Packets']
    IGtop_10_features = IGtop_5_features + ['Subflow Bwd Bytes', 'Avg Bwd Segment Size', 'Bwd Packet Length Mean', 'Total Length of Fwd Packets', 'Subflow Fwd Bytes']

    # Top features selected using K-best
    Kbest_top_5_features = ['Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow IAT Max', 'Fwd IAT Std']
    Kbest_top_10_features = Kbest_top_5_features + ['Fwd IAT Max', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance']

    # Subset data based on feature selection
    datasets = {
        "All Features": (X_train, X_test , y_train, y_test),
        "IG Top 5 Features": (X_train[IGtop_5_features], X_test[IGtop_5_features], y_train, y_test),
        "IG Top 10 Features": (X_train[IGtop_10_features], X_test[IGtop_10_features],  y_train, y_test),
        "KBest Top 5 Features": (X_train[Kbest_top_5_features], X_test[Kbest_top_5_features], y_train, y_test),
        "KBest Top 10 Features": (X_train[Kbest_top_10_features], X_test[Kbest_top_10_features], y_train, y_test),
    }
    return datasets , le

