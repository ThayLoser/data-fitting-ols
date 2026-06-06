import matplotlib.pyplot as plt
from utils import transpose, matmul as mat_mult, get_inverse as invert_matrix

def mat_add(A, B):
    """Cộng hai ma trận cùng kích thước."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def scalar_mult(mat, scalar):
    """Nhân ma trận với một số vô hướng."""
    return [[mat[i][j] * scalar for j in range(len(mat[0]))] for i in range(len(mat))]

def eye(n):
    """Tạo ma trận đơn vị kích thước n x n."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def ridge_fit(X, y, lam, fit_intercept=True):
    """
    X là ma trận design (đã có cột 1 ở đầu).
    """
    if fit_intercept:
        # Thêm số 1.0 vào đầu mỗi dòng của X
        X_design = [[1.0] + row for row in X]
    else:
        X_design = [row[:] for row in X] 

    n_cols = len(X_design[0])

    I = eye(n_cols)
    
    if fit_intercept:
        I[0][0] = 0.0
        
    lam_I = scalar_mult(I, lam)

    X_T = transpose(X_design)
    X_T_X = mat_mult(X_T, X_design)
    
    A = mat_add(X_T_X, lam_I)
    A_inv = invert_matrix(A)
    X_T_y = mat_mult(X_T, y)
    
    beta_ridge = mat_mult(A_inv, X_T_y)
    
    if isinstance(beta_ridge[0], list):
        beta_ridge = [b[0] for b in beta_ridge]
    
    return beta_ridge

def plot_ridge_trace(X, y, lambdas, fit_intercept=True):
    coefs = []
    for lam in lambdas:
        beta_ridge = ridge_fit(X, y, lam, fit_intercept=fit_intercept)
        coefs.append(beta_ridge)
        
    plt.figure(figsize=(10, 6))
    
    num_features = len(X[0]) 
    
    start_idx = 1 if fit_intercept else 0
    end_idx = num_features + 1 if fit_intercept else num_features
    
    # Không vẽ intercept
    for j in range(start_idx, end_idx):
        coef_j = [c[j] for c in coefs]
        plt.plot(lambdas, coef_j, label=f'$\\beta_{{{j}}}$')
        
    plt.xscale('log')
    plt.xlabel('$\\lambda$ (log scale)', fontsize=12)
    plt.ylabel('Coefficients (Hệ số hồi quy)', fontsize=12)
    plt.title('Ridge Trace', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.show()

def ridge_regression(X, y, lambda_param: float = 0.1) -> list:
    """
    Ridge Regression phục vụ cho bài toán kiểm chứng / vẽ Ridge Trace.
    
    Giải pháp tối ưu toán học:
    Vì X từ generate_X_and_y đã đính sẵn cột 1.0 (intercept) ở đầu,
    ta lột bỏ cột này đi, sau đó gọi ridge_fit với fit_intercept=True.
    Như vậy ridge_fit sẽ tự chèn lại cột 1.0 và gán I[0][0] = 0.0 để KHÔNG phạt hệ số chặn!
    """
    X_no_intercept = [row[1:] for row in X]
    return ridge_fit(X_no_intercept, y, lambda_param, fit_intercept=True)