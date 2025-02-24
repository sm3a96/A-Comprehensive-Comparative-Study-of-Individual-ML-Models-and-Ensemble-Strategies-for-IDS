# A Comprehensive Comparative Study of Individual ML Models and Ensemble Strategies for Network Intrusion Detection Systems

## Overview  

With the increasing sophistication of cyber threats, Intrusion Detection Systems (IDS) play a critical role in safeguarding networks from attacks. This study explores **up to 14 individual and ensemble-based machine learning models**, rigorously evaluating their effectiveness in intrusion detection.  

We propose a **flexible ensemble learning framework**, tested on two benchmark datasets—**RoEduNet-SIMARGL2021** and **CICIDS-2017**—to analyze the impact of different modeling strategies.  

Our results reveal that **ensemble methods consistently outperform individual models**, with Decision Trees and Random Forest achieving **perfect classification (F1 = 1.0)** on RoEduNet-SIMARGL2021, while blending and bagging methods excel on CICIDS-2017 (F1 > 0.996). Additionally, **feature selection using Information Gain (IG) significantly reduced training time by up to 94% without sacrificing accuracy**.  
![image](https://github.com/user-attachments/assets/792beea2-34ba-48bf-9273-dcef8b093ab4)

---

## 🚀 Key Contributions  

- ✅ **Comprehensive Evaluation** of 14 ML models, including Decision Trees, Logistic Regression, k-Nearest Neighbors, and Neural Networks.  
- ✅ **Comparison of Simple & Advanced Ensemble Methods**, such as Bagging, Boosting, Stacking, and Blending.  
- ✅ **Feature Selection Insights** using IG and K-best techniques to optimize performance and reduce computation costs.  
- ✅ **Performance Benchmarking** across multiple datasets with metrics including Accuracy, Precision, Recall, F1-score, and Runtime.  
- ✅ **Publicly Available Framework** to facilitate reproducibility and future research in network intrusion detection.  

---

## 📊 Experimental Setup  

### 📂 Datasets  
- **CICIDS-2017** (Canadian Institute for Cybersecurity)  
- **RoEduNet-SIMARGL2021** (EU-supported intrusion dataset)  

### 🛠️ ML Techniques Used  
- **Individual Models:**  
  - Decision Trees (DT)  
  - k-Nearest Neighbors (k-NN)  
  - Multi-Layer Perceptron (MLP)  
  - Logistic Regression (LR)  

- **Ensemble Methods:**  
  - Random Forest (RF)  
  - eXtreme Gradient Boosting (XGBoost)  
  - Adaptive Boosting (AdaBoost)  
  - Blending, Stacking, Bagging  

- **Feature Selection:**  
  - **Information Gain (IG)**  
  - **K-Best (ANOVA F-score)**  

### 🖥️ Computational Resources  
- **High-performance computing (HPC) cluster** with **NVIDIA A100 GPUs**  
- **TensorFlow & Scikit-learn** for model implementation  

---

## 📈 Results Summary  

📌 **Ensemble methods consistently outperformed standalone models.**  
📌 **Feature selection drastically reduced computational time while maintaining accuracy.**  
📌 **Random Forest and XGBoost provided an optimal balance of speed and accuracy.**  
