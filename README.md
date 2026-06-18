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
| 4 | Logistic Regression | Supervised – Classification | Breast Cancer |
| 5 | Linear Regression | Supervised – Regression | Synthetic |
| 6 | K-Means Clustering | Unsupervised – Clustering | Synthetic |

---

## 📁 Cấu trúc project

```
Báo Cáo Tiểu Luận Kết Thúc Học Phần/
├── app.py              # File chính Streamlit
├── requirements.txt    # Dependencies
└── README.md           # Hướng dẫn
```

---

## Tính năng

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
- **Logistic Regression**: Phân loại dựa trên xác suất Sigmoid  
- **Linear Regression**: Hồi quy tuyến tính MSE  

### Unsupervised Learning
- **K-Means**: Phân cụm dữ liệu không có nhãn  

---

## Công nghệ sử dụng

- **Streamlit** – Web framework cho Data Science
- **scikit-learn** – Thư viện ML chính
- **matplotlib / seaborn** – Trực quan hóa dữ liệu
- **NumPy / Pandas** – Xử lý dữ liệu

---

## Tác giả

*Trần Ngọc Bảo Toàn*  
*MSV: 24T1110010"
*Năm học 2024–2025*
