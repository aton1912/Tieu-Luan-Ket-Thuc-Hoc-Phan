import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Thuật toán Học máy",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #555; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #f0f4ff; border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center;
        border: 1px solid #c5d5fb;
    }
    .algo-badge {
        display: inline-block; background: #1a73e8;
        color: white; border-radius: 20px;
        padding: 0.2rem 0.9rem; font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #1a237e; border-left: 4px solid #1a73e8;
        padding-left: 0.6rem; margin: 1.2rem 0 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title"> Thuật Toán Học Máy & Ứng Dụng</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tiểu luận Nhập môn Khoa học Dữ liệu – Demo tương tác các thuật toán ML phổ biến</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown("## 🧭 Chọn thuật toán")
    algo = st.selectbox("Thuật toán", [
        "1. K-Nearest Neighbors (KNN)",
        "2. Decision Tree",
        "3. Random Forest",
        "4. Logistic Regression",
        "5. Linear Regression",
        "6. K-Means Clustering",
    ])
    st.markdown("---")
    st.markdown("### 📚 Thông tin")
    st.info("Chọn từng thuật toán để xem lý thuyết, điều chỉnh tham số và xem kết quả trực quan.")

def show_metrics(acc, report):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2.2rem;font-weight:800;color:#1a73e8">{acc*100:.1f}%</div>
            <div style="color:#555;font-size:0.9rem">Độ chính xác (Accuracy)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.text(report)

# 1. KNN
if "KNN" in algo:
    st.markdown('<div class="algo-badge">Supervised Learning · Classification</div>', unsafe_allow_html=True)
    st.markdown("## 1️⃣ K-Nearest Neighbors (KNN)")

    with st.expander("📖 Lý thuyết KNN", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
**KNN** phân loại một điểm dữ liệu mới bằng cách tìm **K điểm gần nhất** trong tập huấn luyện và bầu chọn theo đa số.
""")
        with col2:
            np.random.seed(42)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            X_demo = np.random.randn(30, 2)
            y_demo = (X_demo[:, 0] + X_demo[:, 1] > 0).astype(int)
            for cls, c in enumerate(['#e53935', '#1e88e5']):
                ax.scatter(X_demo[y_demo == cls, 0], X_demo[y_demo == cls, 1], c=c, s=50, label=f"Lớp {cls}")
            ax.legend(fontsize=7)
            st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        k = st.slider("Số lượng hàng xóm K", 1, 20, 5)
        dataset_knn = st.selectbox("Dataset", ["Iris", "Breast Cancer"])
        test_size = st.slider("Tỷ lệ test (%)", 10, 40, 20)

    if dataset_knn == "Iris":
        data = load_iris()
        target_names = ["setosa", "versicolor", "virginica"]
    else:
        data = load_breast_cancer()
        target_names = ["malignant", "benign"]

    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    with col2:
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        show_metrics(acc, report)

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title("Confusion Matrix")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        ks = list(range(1, 21))
        accs = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=ki).fit(X_train_s, y_train).predict(X_test_s)) for ki in ks]
        fig2, ax = plt.subplots(figsize=(4, 3))
        ax.plot(ks, accs, 'o-', color='#1a73e8', linewidth=2)
        ax.set_title("Accuracy theo K")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# 2. DECISION TREE
elif "Decision Tree" in algo:
    st.markdown('<div class="algo-badge">Supervised Learning · Classification</div>', unsafe_allow_html=True)
    st.markdown("## 2️⃣ Decision Tree (Cây quyết định)")

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        max_depth = st.slider("Độ sâu tối đa", 1, 10, 4)
        criterion = st.selectbox("Tiêu chí phân chia", ["gini", "entropy"])

    data = load_iris()
    target_names = ["setosa", "versicolor", "virginica"]
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with col2:
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        show_metrics(acc, report)

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        fig2, ax = plt.subplots(figsize=(4, 3))
        ax.barh(range(len(model.feature_importances_)), model.feature_importances_, color='#1a73e8')
        ax.set_yticks(range(len(model.feature_importances_)))
        ax.set_yticklabels(feature_names)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown('<div class="section-header">🌳 Cây quyết định</div>', unsafe_allow_html=True)
    fig3, ax = plt.subplots(figsize=(14, 5))
    plot_tree(model, feature_names=feature_names, class_names=target_names, filled=True, rounded=True, fontsize=8, ax=ax)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

# 3. RANDOM FOREST
elif "Random Forest" in algo:
    st.markdown('<div class="algo-badge">Ensemble Learning · Classification</div>', unsafe_allow_html=True)
    st.markdown("## 3️⃣ Random Forest")

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        n_estimators = st.slider("Số cây", 10, 200, 100, 10)
        max_depth_rf = st.slider("Độ sâu tối đa", 1, 20, 8)
        dataset_rf = st.selectbox("Dataset", ["Iris", "Breast Cancer"])

    if dataset_rf == "Iris":
        data = load_iris()
        target_names = ["setosa", "versicolor", "virginica"]
        feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    else:
        data = load_breast_cancer()
        target_names = ["malignant", "benign"]
        feature_names = list(data.feature_names)
        
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_rf, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with col2:
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        show_metrics(acc, report)

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        importances = model.feature_importances_
        fig2, ax = plt.subplots(figsize=(4, 3))
        idx = np.argsort(importances)
        ax.barh(range(len(idx)), importances[idx], color='#43a047')
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([str(feature_names[i])[:15] for i in idx], fontsize=7)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# 4. LOGISTIC REGRESSION
elif "Logistic" in algo:
    st.markdown('<div class="algo-badge">Supervised Learning · Classification</div>', unsafe_allow_html=True)
    st.markdown("## 4️⃣ Logistic Regression")

    col1, col2 = st.columns([1, 2])
    with col1:
        C_lr = st.slider("Tham số C (L2)", 0.01, 10.0, 1.0)

    data = load_breast_cancer()
    target_names = ["malignant", "benign"]
    
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(C=C_lr, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    with col2:
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        show_metrics(acc, report)

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        y_prob = model.predict_proba(X_test_s)[:, 1]
        fig2, ax = plt.subplots(figsize=(4, 3))
        ax.hist(y_prob[y_test == 0], bins=15, alpha=0.5, color='red', label='Malignant')
        ax.hist(y_prob[y_test == 1], bins=15, alpha=0.5, color='blue', label='Benign')
        ax.legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# 5. LINEAR REGRESSION
elif "Linear Regression" in algo:
    st.markdown('<div class="algo-badge">Supervised Learning · Regression</div>', unsafe_allow_html=True)
    st.markdown("## 5️⃣ Linear Regression")

    col1, col2 = st.columns([1, 2])
    with col1:
        n_samples = st.slider("Số mẫu", 50, 500, 150)
        noise = st.slider("Độ nhiễu", 0, 100, 30)

    X, y = make_regression(n_samples=n_samples, n_features=2, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with col2:
        c1, c2 = st.columns(2)
        c1.metric("R² Score", f"{r2:.4f}")
        c2.metric("MSE", f"{mse:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Thực tế"); ax.set_ylabel("Dự đoán")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        fig2, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(y_pred, y_test - y_pred, alpha=0.6, color='purple')
        ax.axhline(0, color='red', linestyle='--')
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# 6. K-MEANS
elif "K-Means" in algo:
    st.markdown('<div class="algo-badge">Unsupervised Learning · Clustering</div>', unsafe_allow_html=True)
    st.markdown("## 6️⃣ K-Means Clustering")

    col1, col2 = st.columns([1, 2])
    with col1:
        n_clusters = st.slider("Số cụm K", 2, 8, 4)

    X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.8, random_state=42)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    with col2:
        st.metric("Inertia (WCSS)", f"{model.inertia_:.1f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=30, alpha=0.7)
        ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='black', marker='X', s=150)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        inertias = [KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X).inertia_ for ki in range(1, 9)]
        fig2, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(range(1, 9), inertias, 'o-', color='#1a73e8')
        plt.tight_layout(); st.pyplot(fig2); plt.close()

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding: 1rem 0">
    📊 Tiểu luận Nhập môn Khoa học Dữ liệu
        -Trần Ngọc Bảo Toàn
            -24T1110010
</div>
""", unsafe_allow_html=True)
