import math
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from part1utils.ols import ols_fit
from part1utils.utils import transpose as ols_transpose, get_inverse as invert, matmul as multiply
from part1utils.ridge_lasso import ridge_fit, plot_ridge_trace
from part1utils.metrics import model_metrics, adjusted_R_squared, F_statistic, vif as VIF
from part1utils.cross_validation import kfold_cv
EPSILON = 1e-9

def dataframe_to_matrix(df):
    """Chuyển DataFrame thành ma trận list of lists."""
    return [list(map(float, row)) for row in df.values.tolist()]

def series_to_matrix(series):
    """Chuyển Series hoặc mảng 1 cột thành ma trận N x 1 (Dùng cho OLS)."""
    if hasattr(series, 'values'):
        values = list(series.values)
    else:
        values = list(series)
    return [[float(val)] for val in values]

def series_to_vector(series):
    """Chuyển Series hoặc mảng thành mảng 1 chiều list thuần túy (Dùng cho Ridge)."""
    if hasattr(series, 'values'):
        return list(map(float, series.values))
    return list(map(float, series))

def add_constant(df):
    """Thêm cột hằng số intercept vào đầu DataFrame."""
    return pd.concat([pd.Series([1] * len(df), name='const'), df.reset_index(drop=True)], axis=1)

def calculate_log_likelihood(rss, n):
    """Tính log-likelihood."""
    if rss <= 0: return float('nan')
    return -0.5 * n * (math.log(2 * math.pi) + 1 + math.log(rss / n))

def approximate_pvalue_from_t(t, df):
    """Xấp xỉ p-value của t-statistic qua hàm lỗi toán học math.erf."""
    if df <= 0:
        return float('nan')
    t_abs = abs(t)
    z = t_abs * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + (t_abs ** 2) / (2.0 * df))
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

def format_coefficients(beta_hat, inv_X_T_X, sigma2_hat, feature_names, df_resid):
    """Tạo bảng DataFrame thống kê chi tiết các tham số học được."""
    var_cov = [[cell * sigma2_hat for cell in row] for row in inv_X_T_X]
    std_err = [math.sqrt(max(0, var_cov[i][i])) for i in range(len(var_cov))]
    
    t_stats = [beta_hat[i][0] / std_err[i] if abs(std_err[i]) > EPSILON else float('inf')
               for i in range(len(beta_hat))]
    
    p_values = [approximate_pvalue_from_t(t, df_resid) for t in t_stats]
    lower_bound = [beta_hat[i][0] - 1.96 * std_err[i] for i in range(len(beta_hat))]
    upper_bound = [beta_hat[i][0] + 1.96 * std_err[i] for i in range(len(beta_hat))]
    return pd.DataFrame({
        'Variable': ['const'] + feature_names,
        'coef': [b[0] for b in beta_hat],
        'std err': std_err,
        't': t_stats,
        'P>|t|': p_values,
        '[0.025': lower_bound, 
        '0.975]': upper_bound
    })


def run_ols_model(train_encoded, y_train):
    """Thực thi mô hình OLS và xuất bảng thống kê chỉ số chi tiết."""
    X_train_df = train_encoded.drop(columns=['imdb_score'], errors='ignore')
    X_train_const = add_constant(X_train_df)
    X_train = dataframe_to_matrix(X_train_const)
    y_train_matrix = series_to_matrix(y_train)

    X_T = ols_transpose(X_train)
    X_T_X = multiply(X_T, X_train)
    inv_X_T_X = invert(X_T_X)
    X_T_y = multiply(X_T, y_train_matrix)
    beta_hat = multiply(inv_X_T_X, X_T_y)

    n = len(y_train_matrix)
    p = len(X_train[0]) - 1
    
    y_hat = multiply(X_train, beta_hat)
    
    y_train_flat = [row[0] for row in y_train_matrix]
    y_hat_flat = [row[0] for row in y_hat]

    metrics = model_metrics(y_train_flat, y_hat_flat, p)
    
    sigma2_hat = metrics["RSS"] / (n - p - 1) if (n - p - 1) > 0 else float('nan')
    llf = calculate_log_likelihood(metrics["RSS"], n)
    k_params = p + 1
    aic = -2 * llf + 2 * k_params
    bic = -2 * llf + k_params * math.log(n)
    vif_values = VIF(X_train)
    vif_df = pd.DataFrame({
        'Biến số độc lập': list(X_train_df.columns),
        'VIF': vif_values
    }).sort_values(by='VIF', ascending=False)

    return {
        'beta_hat': beta_hat,  
        'coefficients': format_coefficients(beta_hat, inv_X_T_X, sigma2_hat, list(X_train_df.columns), n - p - 1),
        'sigma2_hat': sigma2_hat,
        'rss': metrics["RSS"],
        'r2': metrics["R2"],
        'adj_r2': metrics["Adjusted_R2"],
        'vif': vif_df,
        'f_stat': metrics["F_statistic"],
        'llf': llf,
        'aic': aic,
        'bic': bic,
        'df_model': p,
        'df_resid': n - p - 1,
        'nobs': n
    }
def ridge_predict(X, beta_ridge, fit_intercept=True):
    """Dự báo giá trị y dựa trên ma trận X và hệ số beta_ridge (Đầu ra Vector 1D)."""
    if fit_intercept:
        X_design = [[1.0] + row for row in X]
    else:
        X_design = [row[:] for row in X]
    return multiply(X_design, beta_ridge)

def ridge_cv(X_train, y_train, lambdas, k_folds=5, fit_intercept=True, random_state=42):
    """Thực hiện K-Fold Cross Validation tận dụng hàm mat_vec_mult và ridge_fit từ file nhóm."""
    y_train_vec = series_to_vector(y_train)
    
    lambda_avg_mse = {}
    
    for lam in lambdas:
        cv_score = kfold_cv(X_train, y_train_vec, k=k_folds, lam=lam, fit_intercept=fit_intercept)
        lambda_avg_mse[lam] = cv_score
        
    best_lambda = min(lambda_avg_mse, key=lambda k: float(lambda_avg_mse[k]))
    
    return best_lambda, lambda_avg_mse

def plot_cv_error(lambda_avg_mse, best_lambda):
    """Vẽ biểu đồ sự biến thiên của lỗi CV theo hệ số lambda."""
    lams = list(lambda_avg_mse.keys())
    mses = list(lambda_avg_mse.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(lams, mses, marker='o', color='b', label='CV Mean MSE')
    plt.axvline(x=best_lambda, color='r', linestyle='--', label=f'Best $\\lambda$ = {best_lambda:.4f}')
    plt.xscale('log')
    plt.xlabel('$\\lambda$ (log scale)', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.title('Cross-Validation Error vs. Lambda', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.show()