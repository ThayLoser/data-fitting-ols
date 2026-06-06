import math
import scipy.stats as stats
from utils import transpose, matmul, get_inverse

def ols_fit(X, y):
    """OLS Regression. Trả về beta_hat và sigma2_hat."""
    XT = transpose(X)
    XTX_inv = get_inverse(matmul(XT, X))
    beta_hat = matmul(matmul(XTX_inv, XT), y)
    
    n = len(X)
    k = len(X[0])
    
    y_hat = matmul(X, beta_hat)
    
    if isinstance(y[0], list):
        rss = sum((yi[0] - yhi[0])**2 for yi, yhi in zip(y, y_hat))
    else:
        rss = sum((yi - yhi)**2 for yi, yhi in zip(y, y_hat))
        
    sigma2_hat = rss / (n - k)
    
    return beta_hat, sigma2_hat

def hat_matrix(X):
    """Tính Hat Matrix H = X (X' X)^{-1} X' và kiểm tra idempotent."""
    XT = transpose(X)
    XTX = matmul(XT, X)
    XTX_inv = get_inverse(XTX)
    H = matmul(matmul(X, XTX_inv), XT)
    
    # Kiểm tra tính lũy đẳng (idempotent)
    H_sq = matmul(H, H)
    is_idem = True
    for i in range(len(H)):
        for j in range(len(H[0])):
            if abs(H_sq[i][j] - H[i][j]) > 1e-9:
                is_idem = False
                break
        if not is_idem:
            break
            
    return H, is_idem

def coef_inference(X, y, beta_hat, sigma2):
    """
    Tính standard errors, t-statistics, p-values và khoảng tin cậy 95%.
    """
    n = len(X)
    k = len(X[0])
    df = n - k
    
    XT = transpose(X)
    XTX = matmul(XT, X)
    XTX_inv = get_inverse(XTX)
    
    std_errors = []
    t_stats = []
    p_values = []
    conf_intervals = []
    
    # Critical value for 95% CI (two-tailed)
    t_crit = stats.t.ppf(0.975, df)
    
    for j in range(k):
        # standard error của beta_j
        se = math.sqrt(sigma2 * XTX_inv[j][j])
        std_errors.append(se)
        
        # t-statistic
        t_val = beta_hat[j] / se
        t_stats.append(t_val)
        
        # p-value (two-tailed)
        p_val = stats.t.sf(abs(t_val), df) * 2
        p_values.append(p_val)
        
        # 95% Confidence interval
        ci_lower = beta_hat[j] - t_crit * se
        ci_upper = beta_hat[j] + t_crit * se
        conf_intervals.append((ci_lower, ci_upper))
        
    return {
        "std_errors": std_errors,
        "t_stats": t_stats,
        "p_values": p_values,
        "conf_intervals": conf_intervals
    }
