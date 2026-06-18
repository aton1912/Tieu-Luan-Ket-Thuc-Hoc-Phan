import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import (
    load_iris, load_breast_cancer, make_moons, make_blobs, make_regression
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Thuật toán Học máy – Demo",
    page_icon="🤖",
    layout="wide",
)

# ─────────── CSS tùy chỉnh ───────────
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

st.markdown('<div class="main-title">🤖 Thuật Toán Học Máy & Ứng Dụng</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tiểu luận Nhập môn Khoa học Dữ liệu – Demo tương tác các thuật toán ML phổ biến</div>', unsafe_allow_html=True)

# ─────────── Sidebar ───────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown("## 🧭 Chọn thuật toán")
    algo = st.selectbox("Thuật toán", [
        "1. K-Nearest Neighbors (KNN)",
        "2. Decision Tree",
        "3. Random Forest",
        "4. Support Vector Machine (SVM)",
        "5. Logistic Regression",
        "6. Linear Regression",
        "7. K-Means Clustering",
    ])
    st.markdown("---")
    st.markdown("### 📚 Thông tin")
    st.info("Chọn từng thuật toán để xem lý thuyết, điều chỉnh tham số và xem kết quả trực quan.")

# ═══════════════════════════════════════════════
# HÀM TIỆN ÍCH
# ═══════════════════════════════════════════════

def plot_confusion(y_test, y_pred, labels=None):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig

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

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    h = 0.02
    x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
    y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                         edgecolors='k', s=40, linewidths=0.5)
    ax.set_title(title); ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════
#  KNN
# ═══════════════════════════════════════════════
if "KNN" in algo:
    st.markdown('<div class="algo-badge">Supervised Learning · Classification</div>', unsafe_allow_html=True)
    st.markdown("## 1️⃣ K-Nearest Neighbors (KNN)")

    with st.expander("📖 Lý thuyết KNN", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
**KNN** phân loại một điểm dữ liệu mới bằng cách tìm **K điểm gần nhất** trong tập huấn luyện và bầu chọn theo đa số.

**Công thức khoảng cách Euclidean:**
$$d(x, y) = \\sqrt{\\sum_{i=1}^{n}(x_i - y_i)^2}$$

**Ưu điểm:**
- Đơn giản, dễ hiểu
- Không cần giai đoạn huấn luyện

**Nhược điểm:**
- Chậm với dữ liệu lớn
- Nhạy cảm với nhiễu và chiều dữ liệu cao

**Ứng dụng:** Hệ thống gợi ý phim, nhận dạng chữ viết tay, chẩn đoán bệnh.
""")
        with col2:
            # Minh họa KNN
            np.random.seed(42)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            X_demo = np.random.randn(30, 2)
            y_demo = (X_demo[:, 0] + X_demo[:, 1] > 0).astype(int)
            colors = ['#e53935', '#1e88e5']
            for cls, c in enumerate(colors):
                mask = y_demo == cls
                ax.scatter(X_demo[mask, 0], X_demo[mask, 1], c=c, s=50, label=f"Lớp {cls}")
            new_pt = np.array([[0.2, 0.1]])
            ax.scatter(*new_pt[0], c='gold', s=200, marker='*', zorder=5, label='Điểm mới')
            dists = np.sqrt(((X_demo - new_pt)**2).sum(axis=1))
            knn_idx = np.argsort(dists)[:3]
            for idx in knn_idx:
                ax.plot([new_pt[0,0], X_demo[idx,0]], [new_pt[0,1], X_demo[idx,1]],
                        'k--', alpha=0.5, linewidth=1)
            ax.legend(fontsize=7); ax.set_title("Minh họa KNN (K=3)", fontsize=9)
            st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        k = st.slider("Số lượng hàng xóm K", 1, 20, 5)
        dataset_knn = st.selectbox("Dataset", ["Iris", "Breast Cancer"])
        test_size = st.slider("Tỷ lệ test (%)", 10, 40, 20)

    if dataset_knn == "Iris":
        data = load_iris()
    else:
        data = load_breast_cancer()
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
        report = classification_report(y_test, y_pred, target_names=data.target_names, zero_division=0)
        show_metrics(acc, report)

    col1, col2 = st.columns(2)
    with col1:
        fig = plot_confusion(y_test, y_pred, list(data.target_names))
        st.pyplot(fig); plt.close()
    with col2:
        # K vs Accuracy curve
        ks = list(range(1, 21))
        accs = []
        for ki in ks:
            m = KNeighborsClassifier(n_neighbors=ki)
            m.fit(X_train_s, y_train)
            accs.append(accuracy_score(y_test, m.predict(X_test_s)))
        fig2, ax = plt.subplots(figsize=(4, 3))
        ax.plot(ks, accs, 'o-', color='#1a73e8', linewidth=2)
        ax.axvline(k, color='red', linestyle='--', alpha=0.7, label=f'K={k}')
        ax.set_xlabel("K"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy theo K")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig2); plt.close()


# ═══════════════════════════════════════════════
#  DECISION TREE
# ═══════════════════════════════════════════════
elif "Decision Tree" in algo:
    st.markdown('<div class="algo-badge">Supervised Learning · Classification / Regression</div>', unsafe_allow_html=True)
    st.markdown("## 2️⃣ Decision Tree (Cây quyết định)")

    with st.expander("📖 Lý thuyết", expanded=True):
        st.markdown("""
**Decision Tree** xây dựng cây phân nhánh dựa trên các câu hỏi về thuộc tính dữ liệu.

**Tiêu chí phân chia (Gini Impurity):**
$$Gini = 1 - \\sum_{i=1}^{C} p_i^2$$

**Information Gain (Entropy):**
$$IG = H(parent) - \\sum_i \\frac{n_i}{n} H(child_i)$$

**Ưu điểm:** Dễ giải thích, không cần chuẩn hóa dữ liệu  
**Nhược điểm:** Dễ overfit nếu không giới hạn độ sâu  
**Ứng dụng:** Chẩn đoán y tế, phân tích rủi ro tín dụng, phát hiện gian lận.
""")

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        max_depth = st.slider("Độ sâu tối đa", 1, 10, 4)
        criterion = st.selectbox("Tiêu chí phân chia", ["gini", "entropy"])
        min_samples = st.slider("Min samples split", 2, 20, 5)

    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                   min_samples_split=min_samples, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with col2:
        report = classification_report(y_test, y_pred, target_names=data.target_names, zero_division=0)
        show_metrics(acc, report)

    col1, col2 = st.columns(2)
    with col1:
        fig = plot_confusion(y_test, y_pred, list(data.target_names))
        st.pyplot(fig); plt.close()
    with col2:
        # Feature importance
        importances = model.feature_importances_
        feat_names = data.feature_names
        fig2, ax = plt.subplots(figsize=(4, 3))
        idx = np.argsort(importances)
        ax.barh(range(len(idx)), importances[idx], color='#1a73e8', alpha=0.8)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feat_names[i] for i in idx], fontsize=8)
        ax.set_title("Feature Importance")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown('<div class="section-header">🌳 Cây quyết định</div>', unsafe_allow_html=True)
    fig3, ax = plt.subplots(figsize=(14, 5))
    plot_tree(model, 
        if dataset_name == "Iris":
            feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
        elif dataset_name == "Breast Cancer":
            feature_names = list(load_breast_cancer().feature_names)
        else:
            feature_names = [f"Feature {i+1}" for i in range(X.shape[1])], class_names=data.target_names,
              filled=True, rounded=True, fontsize=8, ax=ax)
    plt.tight_layout(); st.pyplot(fig3); plt.close()


# ═══════════════════════════════════════════════
#  RANDOM FOREST
# ═══════════════════════════════════════════════
elif "Random Forest" in algo:
    st.markdown('<div class="algo-badge">Ensemble Learning · Classification / Regression</div>', unsafe_allow_html=True)
    st.markdown("## 3️⃣ Random Forest")

    with st.expander("📖 Lý thuyết", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
**Random Forest** là tập hợp nhiều cây quyết định, mỗi cây được huấn luyện trên một mẫu bootstrap khác nhau với tập thuộc tính ngẫu nhiên.

**Dự đoán cuối cùng:**
$$\\hat{y} = \\text{majority vote}\\{h_1(x), h_2(x), ..., h_T(x)\\}$$

- **Bagging**: Mỗi cây dùng mẫu bootstrap khác nhau  
- **Feature randomness**: Mỗi lần tách chỉ xét $\\sqrt{p}$ thuộc tính

**Ưu điểm:** Chính xác cao, chống overfit  
**Ứng dụng:** Dự đoán giá cổ phiếu, phân loại ảnh, phát hiện gian lận.
""")
        with col2:
            # Minh họa ensemble
            fig, ax = plt.subplots(figsize=(3.5, 3))
            trees = ["Tree 1\n→ A", "Tree 2\n→ B", "Tree 3\n→ A", "Tree 4\n→ A"]
            colors = ['#bbdefb', '#bbdefb', '#bbdefb', '#bbdefb']
            for i, (t, c) in enumerate(zip(trees, colors)):
                ax.add_patch(plt.FancyBboxPatch((i*0.9, 0.6), 0.75, 0.4,
                             boxstyle="round,pad=0.05", facecolor=c, edgecolor='#1565c0', linewidth=1.5))
                ax.text(i*0.9+0.375, 0.8, t, ha='center', va='center', fontsize=7, color='#0d47a1')
            ax.add_patch(plt.FancyBboxPatch((1.35, 0.1), 1.0, 0.35,
                         boxstyle="round,pad=0.05", facecolor='#1a73e8', edgecolor='#0d47a1'))
            ax.text(1.85, 0.275, "Vote → A", ha='center', va='center', fontsize=9,
                    color='white', fontweight='bold')
            ax.set_xlim(-0.2, 4); ax.set_ylim(0, 1.3)
            ax.axis('off'); ax.set_title("Ensemble Voting", fontsize=9)
            st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        n_estimators = st.slider("Số cây", 10, 200, 100, 10)
        max_depth_rf = st.slider("Độ sâu tối đa", 1, 20, 8)
        dataset_rf = st.selectbox("Dataset", ["Iris", "Breast Cancer"])

    if dataset_rf == "Iris":
        data = load_iris()
    else:
        data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_rf, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with col2:
        report = classification_report(y_test, y_pred, target_names=data.target_names, zero_division=0)
        show_metrics(acc, report)

    col1, col2 = st.columns(2)
    with col1:
        fig = plot_confusion(y_test, y_pred, list(data.target_names))
        st.pyplot(fig); plt.close()
    with col2:
        importances = model.feature_importances_
        fig2, ax = plt.subplots(figsize=(4, 3))
        idx = np.argsort(importances)
        ax.barh(range(len(idx)), importances[idx], color='#43a047', alpha=0.85)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([data.feature_names[i][:20] for i in idx], fontsize=7)
        ax.set_title("Feature Importance (RF)")
        plt.tight_layout(); st.pyplot(fig2); plt.close()


# ═══════════════════════════════════════════════
#  SVM
# ═══════════════════════════════════════════════
elif "SVM" in algo:
    st.markdown('<div class="algo-badge">Supervised Learning · Classification</div>', unsafe_allow_html=True)
    st.markdown("## 4️⃣ Support Vector Machine (SVM)")

    with st.expander("📖 Lý thuyết", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
**SVM** tìm một siêu phẳng (hyperplane) tối ưu tách biệt các lớp dữ liệu với **lề (margin) rộng nhất** có thể.

**Hàm tối ưu hóa:**
$$\\min_{w,b} \\frac{1}{2}||w||^2 \\quad \\text{s.t. } y_i(w^Tx_i + b) \\geq 1$$

**Kernel trick** (phi tuyến):
- **Linear**: $K(x,y) = x^Ty$
- **RBF (Gaussian)**: $K(x,y) = e^{-\\gamma||x-y||^2}$
- **Polynomial**: $K(x,y) = (x^Ty + c)^d$

**Ứng dụng:** Nhận dạng khuôn mặt, phân loại văn bản, phát hiện email spam.
""")
        with col2:
            # Minh họa margin
            np.random.seed(0)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            X_pos = np.random.randn(15, 2) + np.array([2, 2])
            X_neg = np.random.randn(15, 2) + np.array([-2, -2])
            ax.scatter(X_pos[:, 0], X_pos[:, 1], c='#e53935', s=40, label='Class +1')
            ax.scatter(X_neg[:, 0], X_neg[:, 1], c='#1e88e5', s=40, label='Class -1')
            x_line = np.linspace(-5, 6, 100)
            ax.plot(x_line, -x_line, 'k-', linewidth=2, label='Hyperplane')
            ax.fill_between(x_line, -x_line-1, -x_line+1, alpha=0.15, color='gray', label='Margin')
            ax.set_xlim(-5, 6); ax.set_ylim(-5, 6)
            ax.legend(fontsize=7); ax.set_title("SVM – Hyperplane & Margin", fontsize=9)
            st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        C = st.slider("Tham số C (Regularization)", 0.01, 10.0, 1.0)
        gamma = st.selectbox("Gamma", ["scale", "auto"])

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    with col2:
        report = classification_report(y_test, y_pred, target_names=data.target_names, zero_division=0)
        show_metrics(acc, report)

    col1, col2 = st.columns(2)
    with col1:
        fig = plot_confusion(y_test, y_pred, list(data.target_names))
        st.pyplot(fig); plt.close()
    with col2:
        # Decision boundary on 2 features
        X2 = X[:, :2]
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)
        sc2 = StandardScaler()
        X2_train_s = sc2.fit_transform(X2_train)
        X2_test_s = sc2.transform(X2_test)
        m2 = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        m2.fit(X2_train_s, y2_train)
        fig2 = plot_decision_boundary(m2, X2_train_s, y2_train, f"SVM – {kernel} kernel (2 features)")
        st.pyplot(fig2); plt.close()


# ═══════════════════════════════════════════════
#  LOGISTIC REGRESSION
# ═══════════════════════════════════════════════
elif "Logistic" in algo:
    st.markdown('<div class="algo-badge">Supervised Learning · Binary/Multi-class Classification</div>', unsafe_allow_html=True)
    st.markdown("## 5️⃣ Logistic Regression")

    with st.expander("📖 Lý thuyết", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
**Logistic Regression** dùng hàm **Sigmoid** để dự đoán xác suất thuộc một lớp.

**Hàm Sigmoid:**
$$\\sigma(z) = \\frac{1}{1+e^{-z}}, \\quad z = w^Tx + b$$

**Hàm mất mát (Cross-Entropy):**
$$L = -\\frac{1}{n}\\sum[y_i \\log(\\hat{p}_i) + (1-y_i)\\log(1-\\hat{p}_i)]$$

**Ưu điểm:** Nhanh, giải thích được, đưa ra xác suất  
**Ứng dụng:** Phát hiện email spam, dự báo bệnh, tín dụng.
""")
        with col2:
            z = np.linspace(-6, 6, 200)
            sig = 1 / (1 + np.exp(-z))
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(z, sig, color='#1a73e8', linewidth=2.5)
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='p=0.5 (ngưỡng)')
            ax.fill_between(z, 0, sig, where=(sig > 0.5), alpha=0.15, color='#e53935')
            ax.fill_between(z, 0, sig, where=(sig <= 0.5), alpha=0.15, color='#1e88e5')
            ax.set_xlabel("z = w·x + b"); ax.set_ylabel("σ(z)")
            ax.set_title("Hàm Sigmoid"); ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        C_lr = st.slider("Tham số C (Regularization)", 0.01, 10.0, 1.0)
        solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"])
        max_iter = st.slider("Max iterations", 100, 1000, 300, 50)

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(C=C_lr, solver=solver, max_iter=max_iter, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    with col2:
        report = classification_report(y_test, y_pred, target_names=data.target_names, zero_division=0)
        show_metrics(acc, report)

    col1, col2 = st.columns(2)
    with col1:
        fig = plot_confusion(y_test, y_pred, list(data.target_names))
        st.pyplot(fig); plt.close()
    with col2:
        fig2, ax = plt.subplots(figsize=(4, 3))
        ax.hist(y_prob[y_test == 0], bins=20, alpha=0.6, color='#e53935', label='Malignant')
        ax.hist(y_prob[y_test == 1], bins=20, alpha=0.6, color='#1e88e5', label='Benign')
        ax.axvline(0.5, color='black', linestyle='--')
        ax.set_xlabel("Xác suất dự đoán"); ax.set_ylabel("Số lượng")
        ax.set_title("Phân phối xác suất"); ax.legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()


# ═══════════════════════════════════════════════
#  LINEAR REGRESSION
# ═══════════════════════════════════════════════
elif "Linear Regression" in algo:
    st.markdown('<div class="algo-badge">Supervised Learning · Regression</div>', unsafe_allow_html=True)
    st.markdown("## 6️⃣ Linear Regression")

    with st.expander("📖 Lý thuyết", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
**Linear Regression** mô hình hóa quan hệ tuyến tính giữa biến đầu vào X và đầu ra y.

**Mô hình:**
$$\\hat{y} = w_0 + w_1x_1 + ... + w_px_p = \\mathbf{w}^T\\mathbf{x}$$

**Hàm mất mát (MSE):**
$$MSE = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$

**Nghiệm dạng đóng (Normal Equation):**
$$\\mathbf{w}^* = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{y}$$

**Ứng dụng:** Dự báo giá nhà, ước lượng doanh thu, phân tích xu hướng.
""")
        with col2:
            np.random.seed(42)
            x_demo = np.linspace(0, 10, 40)
            y_demo = 2.5 * x_demo + np.random.randn(40) * 2 + 1
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(x_demo, y_demo, color='#1e88e5', s=30, alpha=0.7)
            m, b = np.polyfit(x_demo, y_demo, 1)
            ax.plot(x_demo, m*x_demo+b, color='#e53935', linewidth=2.5, label=f'y={m:.2f}x+{b:.2f}')
            for xi, yi in zip(x_demo[::4], y_demo[::4]):
                yi_pred = m*xi+b
                ax.plot([xi, xi], [yi, yi_pred], 'gray', alpha=0.4, linewidth=0.8)
            ax.set_title("Linear Regression"); ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        n_samples = st.slider("Số mẫu dữ liệu", 50, 500, 150)
        noise = st.slider("Mức độ nhiễu", 0, 100, 30)
        n_features = st.slider("Số features", 1, 10, 3)
        test_pct = st.slider("Tỷ lệ test (%)", 10, 40, 20)

    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    with col2:
        c1, c2, c3 = st.columns(3)
        for col, label, val in zip([c1, c2, c3],
                                   ["R² Score", "RMSE", "MSE"],
                                   [f"{r2:.4f}", f"{rmse:.2f}", f"{mse:.2f}"]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.7rem;font-weight:800;color:#1a73e8">{val}</div>
                    <div style="color:#555;font-size:0.8rem">{label}</div>
                </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(y_test, y_pred, alpha=0.6, color='#1a73e8', s=30)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect fit')
        ax.set_xlabel("Thực tế"); ax.set_ylabel("Dự đoán")
        ax.set_title("Predicted vs Actual"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        residuals = y_test - y_pred
        fig2, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(y_pred, residuals, alpha=0.6, color='#43a047', s=30)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        plt.tight_layout(); st.pyplot(fig2); plt.close()


# ═══════════════════════════════════════════════
#  K-MEANS
# ═══════════════════════════════════════════════
elif "K-Means" in algo:
    st.markdown('<div class="algo-badge">Unsupervised Learning · Clustering</div>', unsafe_allow_html=True)
    st.markdown("## 7️⃣ K-Means Clustering")

    with st.expander("📖 Lý thuyết", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
**K-Means** phân nhóm dữ liệu thành K cụm bằng cách tối thiểu hóa tổng khoảng cách nội cụm.

**Hàm mục tiêu (WCSS):**
$$J = \\sum_{k=1}^{K}\\sum_{x \\in C_k} ||x - \\mu_k||^2$$

**Thuật toán:**
1. Khởi tạo K centroid ngẫu nhiên
2. Gán mỗi điểm vào cụm gần nhất
3. Cập nhật centroid = trung bình cụm
4. Lặp đến khi hội tụ

**Elbow Method**: Chọn K tại điểm "gập khủy tay" của đường WCSS.

**Ứng dụng:** Phân khúc khách hàng, nén ảnh, phát hiện dị thường.
""")
        with col2:
            np.random.seed(10)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            X_demo, _ = make_blobs(n_samples=80, centers=3, cluster_std=0.8, random_state=10)
            km_demo = KMeans(n_clusters=3, random_state=0, n_init=10)
            labels_demo = km_demo.fit_predict(X_demo)
            colors_demo = ['#e53935', '#1e88e5', '#43a047']
            for k in range(3):
                mask = labels_demo == k
                ax.scatter(X_demo[mask, 0], X_demo[mask, 1], c=colors_demo[k], s=35, alpha=0.7)
            ax.scatter(km_demo.cluster_centers_[:, 0], km_demo.cluster_centers_[:, 1],
                      c='gold', marker='*', s=200, zorder=5, edgecolors='k', label='Centroids')
            ax.legend(fontsize=7); ax.set_title("K-Means (K=3)", fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">⚙️ Thử nghiệm</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        n_clusters = st.slider("Số cụm K", 2, 10, 4)
        n_samples_km = st.slider("Số mẫu", 100, 500, 250)
        n_centers = st.slider("Số cụm thực sự", 2, 8, 4)
        cluster_std = st.slider("Độ phân tán", 0.3, 2.0, 0.8)

    X, y_true = make_blobs(n_samples=n_samples_km, centers=n_centers,
                           cluster_std=cluster_std, random_state=42)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    inertia = model.inertia_

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.9rem;font-weight:800;color:#1a73e8">{inertia:.1f}</div>
            <div style="color:#555">Inertia (WCSS)</div>
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        palette = plt.cm.tab10.colors
        for k in range(n_clusters):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c=[palette[k % 10]], s=30, alpha=0.7, label=f'Cụm {k+1}')
        ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                  c='black', marker='X', s=200, zorder=5, label='Centroids')
        ax.legend(fontsize=7, ncol=2); ax.set_title("Kết quả phân cụm K-Means")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        # Elbow method
        inertias = []
        ks = list(range(1, min(12, n_samples_km//10+2)))
        for ki in ks:
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
        fig2, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(ks, inertias, 'o-', color='#1a73e8', linewidth=2)
        ax.axvline(n_clusters, color='red', linestyle='--', alpha=0.8, label=f'K={n_clusters}')
        ax.set_xlabel("K"); ax.set_ylabel("WCSS (Inertia)")
        ax.set_title("Elbow Method"); ax.legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# ─────────── Footer ───────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding: 1rem 0">
    📊 Tiểu luận Nhập môn Khoa học Dữ liệu &nbsp;|&nbsp; 
    Đề tài: <b>Tìm hiểu về một số thuật toán học máy và ứng dụng trong thực tế</b>
</div>
""", unsafe_allow_html=True)
