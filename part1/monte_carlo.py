import math
import random
from typing import Tuple, Dict, Optional
from utils import transpose, matmul, get_inverse
from ols import ols_fit

def generate_X_and_y(n: int, k: int, seed: Optional[int] = None) -> Tuple[list, list, list, list]:
    """
    Tạo dữ liệu giả lập X (ma trận thiết kế có sẵn intercept) và y (vector phụ thuộc).
    """
    if seed is not None:
        random.seed(seed)
    
    X = [[1.0] + [random.gauss(0, 1) for _ in range(k - 1)] for _ in range(n)]
    beta_true = [float(i) for i in range(k)]  # [0.0, 1.0, 2.0, ...]
    
    sigma2 = 1.0  
    epsilon = [random.gauss(0, math.sqrt(sigma2)) for _ in range(n)]
    
    y = [sum(X[i][j] * beta_true[j] for j in range(k)) + epsilon[i] for i in range(n)]
    
    return X, y, beta_true, epsilon


def suboptimal_unbiased_estimator(X, y) -> list:
    """
    Ước lượng tuyến tính KHÔNG CHỆCH nhưng KHÔNG TỐI ƯU (Suboptimal Weighted Least Squares).
    Công thức: β̂_alt = (X' W X)^(-1) X' W y
    """
    n = len(X)
    k = len(X[0])
    
    # Gán trọng số cố định tuần hoàn dựa theo chỉ số dòng (đảm bảo tính độc lập với nhiễu)
    X_weighted = []
    y_weighted = []
    for i in range(n):
        w = 1.0 + (i % 4) * 0.4  # Trọng số cố định: 1.0, 1.4, 1.8, 2.2
        X_weighted.append([X[i][j] * w for j in range(k)])
        y_weighted.append(y[i] * w)
        
    XT = transpose(X_weighted)
    XTX = matmul(XT, X_weighted)
    XTX_inv = get_inverse(XTX)
    XTy = matmul(XT, y_weighted)
    beta = matmul(XTX_inv, XTy)
    
    return beta


def run_monte_carlo(n_simulations: int, n: int, k: int, 
                   alternative_estimator_func=None) -> Dict:
    """
    Chạy Mô phỏng Monte Carlo để kiểm chứng Định lý Gauss-Markov:
    1. E(β̂_OLS) ≈ β_true (Tính không chệch)
    2. Var(β̂_OLS) < Var(β̂_alternative) (OLS có phương sai nhỏ nhất trong các ông không chệch)
    """
    beta_hat_ols_list = []
    beta_hat_alt_list = []
    beta_true = None
    
    for sim in range(n_simulations):
        X, y, beta_true, epsilon = generate_X_and_y(n, k, seed=sim)
        
        # 1. Ước lượng bằng OLS
        beta_hat_ols, _ = ols_fit(X, y)
        beta_hat_ols_list.append(beta_hat_ols)
        
        # 2. Ước lượng bằng phương pháp thay thế không tối ưu
        if alternative_estimator_func is not None:
            beta_hat_alt = alternative_estimator_func(X, y)
            beta_hat_alt_list.append(beta_hat_alt)
        else:
            beta_hat_alt = suboptimal_unbiased_estimator(X, y)
            beta_hat_alt_list.append(beta_hat_alt)
    
    # Tính Kỳ vọng thực nghiệm
    mean_beta_ols = [sum(b[j] for b in beta_hat_ols_list) / n_simulations for j in range(k)]
    mean_beta_alt = [sum(b[j] for b in beta_hat_alt_list) / n_simulations for j in range(k)]
    
    assert beta_true is not None, "beta_true phải được khởi tạo trong vòng lặp"
    
    # Tính Độ chệch (Bias = E(β̂) - β_true)
    bias_ols = [mean_beta_ols[j] - beta_true[j] for j in range(k)]
    bias_alt = [mean_beta_alt[j] - beta_true[j] for j in range(k)]
    
    # Tính Phương sai (Variance)
    var_ols = [
        sum((beta_hat_ols_list[i][j] - mean_beta_ols[j]) ** 2 
            for i in range(n_simulations)) / n_simulations
        for j in range(k)
    ]
    var_alt = [
        sum((beta_hat_alt_list[i][j] - mean_beta_alt[j]) ** 2 
            for i in range(n_simulations)) / n_simulations
        for j in range(k)
    ]
    
    return {
        'beta_hat_ols': beta_hat_ols_list,
        'beta_hat_alt': beta_hat_alt_list,
        'mean_beta_ols': mean_beta_ols,
        'mean_beta_alt': mean_beta_alt,
        'beta_true': beta_true,
        'bias_ols': bias_ols,
        'bias_alt': bias_alt,
        'var_ols': var_ols,
        'var_alt': var_alt,
    }
