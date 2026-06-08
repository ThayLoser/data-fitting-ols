<div align="center">

*Đọc bằng ngôn ngữ khác: [English](README.md)*

# Hồi quy tuyến tính bằng Bình phương tối thiểu (OLS)

[![Phiên bản Python](https://img.shields.io/badge/Python->=3.8-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Giấy phép](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

Dự án hồi quy xây dựng từ đầu (from scratch), nổi bật với tính chặt chẽ về toán học, hệ thống xử lý dữ liệu và phân tích thống kê chuyên sâu.

[Tổng quan dự án](#tổng-quan-dự-án) • [Tính năng](#tính-năng) • [Bắt đầu](#bắt-đầu) • [Cấu trúc kho lưu trữ](#cấu-trúc-kho-lưu-trữ)

</div>

## Thông tin dự án

- **Môn học**: MTH00051 - Toán Ứng Dụng và Thống Kê (Applied Mathematics and Statistics)
- **Trường**: Ho Chi Minh City University of Science (HCMUS)
- **Tác giả**:
  - Nguyễn Anh Thái
  - Nguyễn Đình Tuấn
  - Nguyễn Huỳnh Gia Bảo
  - Vòng Sau Hậu
  - Lương Nhật Tân

## Tổng quan dự án

Dự án này cung cấp một bản cài đặt toàn diện về hồi quy Bình phương tối thiểu (Ordinary Least Squares - OLS) hoàn toàn từ đầu, sử dụng các nguyên lý đại số tuyến tính cốt lõi. Được thiết kế cho môn Toán ứng dụng và Thống kê, kho lưu trữ (repository) này được chia thành hai phần chính: một phần cài đặt nền tảng chặt chẽ về toán học và một phần ứng dụng thực tế dự đoán điểm IMDB của phim thông qua quy trình tiền xử lý dữ liệu nâng cao.

> [!NOTE]  
> Dự án tập trung vào việc hiểu rõ cách thức hoạt động bên trong của các mô hình hồi quy bằng cách hạn chế tối đa sự phụ thuộc vào các thư viện học máy bậc cao cho các tính toán cốt lõi.

## Tính năng

- **Cài đặt thuật toán từ đầu**: Hồi quy OLS, Ridge và Lasso cơ bản được xây dựng bằng cách sử dụng các phép toán ma trận gốc.
- **Suy luận thống kê (Statistical Inference)**: Tính toán sai số chuẩn (standard errors), giá trị thống kê t (t-statistics), giá trị p (p-values) và khoảng tin cậy 95% trực tiếp từ ma trận Hat.
- **Chẩn đoán nâng cao**: Phân tích phần dư (residual analysis) chuyên sâu, kiểm chứng chéo (cross-validation) và mô phỏng Monte Carlo để kiểm chứng các tính chất của ước lượng và độ mạnh (robustness) của mô hình.
- **Pipeline dữ liệu tối ưu**: Một class tiền xử lý tự động xử lý điền khuyết KNN, Winsorization, biến đổi Box-Cox và mã hóa biến phân loại mà không gây rò rỉ dữ liệu (data leakage).
- **So sánh mô hình**: Đánh giá hiệu suất các bản cài đặt tự viết so với các thư viện tiêu chuẩn như `scikit-learn` và `statsmodels`.

## Công nghệ sử dụng (Technology Stack)

- **Xử lý dữ liệu lõi**: Python (NumPy, pandas, SciPy)
- **Học máy & Thống kê**: scikit-learn, statsmodels
- **Trực quan hóa dữ liệu**: Matplotlib, Seaborn
- **Môi trường phát triển**: Jupyter Notebook

## Tập dữ liệu (Dataset)

Dự án này sử dụng **IMDB 5000 Movie Dataset** cho phần ứng dụng thực tế ở Phần 2. Bạn có thể tải tập dữ liệu này từ Kaggle để tự chạy thử pipeline:

- [IMDB 5000 Movie Dataset trên Kaggle](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)

## Bắt đầu

Bạn cần cài đặt Python trên máy tính cá nhân để chạy dự án này.

### Sử dụng môi trường cục bộ (Local environment)

1. **Clone repository** về máy (nếu chưa thực hiện).

2. **Cài đặt các thư viện phụ thuộc** bằng file `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Khám phá các notebook**:
   - Truy cập vào thư mục `part1/` và mở file `part1_notebook.ipynb` để xem các bản cài đặt từ đầu, chứng minh toán học và mô phỏng Monte Carlo.
   - Truy cập vào thư mục `part2/` và mở file `part2_notebook.ipynb` để xem chi tiết pipeline dữ liệu được áp dụng cho tập dữ liệu phim IMDB.

> [!TIP]  
> Hãy sử dụng JupyterLab hoặc Visual Studio Code kèm extension Python để có trải nghiệm làm việc mượt mà nhất với các file `.ipynb`.

## Cấu trúc kho lưu trữ

Toàn bộ mã nguồn tuân thủ nghiêm ngặt cấu trúc dự án được yêu cầu:

- `part1/`: Chứa các bản cài đặt toán học cốt lõi (`ols.py`, `metrics.py`, `ridge_lasso.py`, v.v.) và các kiểm chứng lý thuyết.
- `part2/`: Bao gồm class `DataPipeline` (`data_pipeline.py`) và các script phục vụ ứng dụng thực tế (`model_comparison.py`, `summary_utils.py`).
- `report/`: Lưu trữ báo cáo dự án tổng hợp cuối cùng (`report.pdf`), trình bày chi tiết phương pháp, chứng minh toán học và kết quả thực nghiệm.
