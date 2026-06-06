from typing import Dict
from ols import ols_fit
from utils import matmul

def RSS(y: list[float], y_hat: list[float]) -> float:
    """RSS = ||y - y_hat||^2."""
    rss = 0.0
    for yi, yhi in zip(y, y_hat):
        val_yi = yi[0] if isinstance(yi, list) else yi
        val_yhi = yhi[0] if isinstance(yhi, list) else yhi
        diff = val_yi - val_yhi
        rss += diff * diff
    return rss

def TSS(y: list[float]) -> float:
    """TSS = ||y - y_bar||^2."""
    y_vals = [yi[0] if isinstance(yi, list) else yi for yi in y]
    y_bar = sum(y_vals) / len(y_vals)
    tss = 0.0
    for yi in y_vals:
        diff = yi - y_bar
        tss += diff * diff
    return tss

def R_squared(y: list[float], y_hat: list[float]) -> float:
    """R2 = 1 - RSS/TSS."""
    rss = RSS(y, y_hat)
    tss = TSS(y)
    if tss == 0:
        raise ValueError("TSS = 0 (y không có biến thiên), R2 không xác định.")

    r_square = 1.0 - rss / tss
    if r_square < 0.0 or r_square > 1.0:
        raise ValueError(f"r_square ({r_square}) không hợp lệ.")
    return r_square

def adjusted_R_squared(r2: float, n: int, p: int) -> float:
    """Adjusted R2 = 1 - (n-1)/(n-p-1) * (1-R2)."""
    return 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)

def F_statistic(rss: float, tss: float, n: int, p: int) -> float:
    """F = (ESS/p) / (RSS/(n-p-1))."""
    if rss == 0:
        raise ValueError("RSS = 0, F-statistic không xác định.")
    if n <= p + 1:
        raise ValueError("Cần n > p + 1 để F-statistic có bậc tự do dương.")

    f_stat = ((tss - rss) / p) / (rss / (n - p - 1))
    return f_stat

def model_metrics(y: list[float], y_hat: list[float], p: int) -> Dict[str, float]:
    """
    Tính các chỉ số đánh giá mô hình OLS: RSS, TSS, R2, Adjusted R2, F-statistic.
    """
    if len(y) != len(y_hat):
        raise ValueError("y và y_hat phải có cùng số phần tử.")
    if not isinstance(p, int) or p < 1:
        raise ValueError("p phải là số nguyên dương.")

    n = len(y)
    if n <= p + 1:
        raise ValueError("Cần n > p + 1 để F-statistic có bậc tự do dương.")

    rss = RSS(y, y_hat)
    tss = TSS(y)

    r2 = 1.0 - rss / tss
    adj_r2 = adjusted_R_squared(r2, n, p)
    f_stat = F_statistic(rss, tss, n, p)

    return {
        "RSS": rss,
        "TSS": tss,
        "R2": r2,
        "Adjusted_R2": adj_r2,
        "F_statistic": f_stat,
    }

def vif(X: list[list[float]]) -> list[float]:
    p = len(X[0])
    vifs = []

    for j in range(1, p):
        """
        Ta chỉ xét tính đa cộng tuyến giữa X_j với các X còn lại, 
        cột đầu toàn 1 dành cho hằng số B_0 nên không xét đa cộng tuyến với nó.
        """
        X_j = [row[j] for row in X]
        X_j_2D = [[val] for val in X_j]

        X_minus_j = [[row[k] for k in range(p) if k != j] for row in X]

        beta_2D, _ = ols_fit(X_minus_j, X_j_2D)
        beta = [b[0] for b in beta_2D]

        X_j_hat = matmul(X_minus_j, beta)  # xấp xỉ của X_j bằng các X còn lại

        data = model_metrics(X_j, X_j_hat, p - 2)  # bỏ đi X_j và cột đầu

        vif_j = 1.0 / (1.0 - data["R2"]) if data["R2"] < 1.0 else float("inf")
        vifs.append(vif_j)

    return vifs
