# Thuật Toán Học Máy & Ứng Dụng

> **Tiểu luận Nhập môn Khoa học Dữ liệu**  
> Đề tài: *Tìm hiểu về một số thuật toán học máy và ứng dụng trong thực tế*

## 🌐 Demo trực tuyến

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tieu-luan-ket-thuc-hoc-phan-lzswz9lnpnyyupxtmkzwra.streamlit.app/)

---

## Nội dung

Ứng dụng Streamlit tương tác trình bày **7 thuật toán học máy** phổ biến:

| # | Thuật toán | Loại | Dataset |
|---|-----------|------|---------|
| 1 | K-Nearest Neighbors (KNN) | Supervised – Classification | Iris, Breast Cancer |
| 2 | Decision Tree | Supervised – Classification | Iris |
| 3 | Random Forest | Ensemble – Classification | Iris, Breast Cancer |
| 4 | Support Vector Machine (SVM) | Supervised – Classification | Breast Cancer |
| 5 | Logistic Regression | Supervised – Classification | Breast Cancer |
| 6 | Linear Regression | Supervised – Regression | Synthetic |
| 7 | K-Means Clustering | Unsupervised – Clustering | Synthetic |

---

## Chạy ứng dụng

### Cách 1: Cài đặt local

```bash
# Clone repository
git clone https://github.com/yourusername/ml-algorithms-demo.git
cd ml-algorithms-demo

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Cài dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
```

Mở trình duyệt và truy cập: `http://localhost:8501`

### Cách 2: Deploy lên Streamlit Cloud

1. Fork repo này
2. Vào [share.streamlit.io](https://share.streamlit.io)
3. Kết nối GitHub và chọn repo
4. Chọn `app.py` làm file chính
5. Deploy!

---

## 📁 Cấu trúc project

```
ml-algorithms-demo/
├── app.py              # File chính Streamlit
├── requirements.txt    # Dependencies
└── README.md           # Hướng dẫn
```

---

## 🔧 Tính năng

- **Lý thuyết**: Giải thích trực quan với công thức toán học LaTeX
- **Điều chỉnh tham số**: Slider tương tác để thay đổi hyperparameter
- **Kết quả trực quan**: Confusion matrix, decision boundary, feature importance
- **Nhiều dataset**: Iris, Breast Cancer, dữ liệu tổng hợp
- **So sánh**: Đường cong K vs Accuracy, Elbow method

---

## 📖 Thuật toán được trình bày

### Supervised Learning
- **KNN**: Phân loại dựa trên K điểm gần nhất  
- **Decision Tree**: Cây quyết định với Gini/Entropy  
- **Random Forest**: Ensemble nhiều cây quyết định  
- **SVM**: Siêu phẳng tối ưu với kernel trick  
- **Logistic Regression**: Phân loại dựa trên xác suất Sigmoid  
- **Linear Regression**: Hồi quy tuyến tính MSE  

### Unsupervised Learning
- **K-Means**: Phân cụm dữ liệu không có nhãn  

---

## 🛠 Công nghệ sử dụng

- **Streamlit** – Web framework cho Data Science
- **scikit-learn** – Thư viện ML chính
- **matplotlib / seaborn** – Trực quan hóa dữ liệu
- **NumPy / Pandas** – Xử lý dữ liệu

---

## 👤 Tác giả

*Sinh viên Nhập môn Khoa học Dữ liệu*  
*Năm học 2024–2025*
