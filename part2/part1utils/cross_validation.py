import random

from .ridge_lasso import ridge_fit
from .utils import matmul

def kfold_cv(X, y, k, lam=0.0, fit_intercept=True):
    n = len(X)
    
    indicies = list(range(n))
    random.seed(42)
    random.shuffle(indicies)
    
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    
    folds = []
    current_idx = 0
    
    for size in fold_sizes:
        folds.append(indicies[current_idx:current_idx+size])
        current_idx += size
        
    mse_list = []
    
    for i in range(k):
        test_idx = folds[i]
        train_idx = []
        for j in range(k):
            if i != j:
                train_idx.extend(folds[j])
                
        X_train = [X[idx] for idx in train_idx]
        y_train = [y[idx] for idx in train_idx]
        X_test = [X[idx] for idx in test_idx]
        y_test = [y[idx] for idx in test_idx]
        
        beta_hat = ridge_fit(X_train, y_train, lam, fit_intercept=fit_intercept)
        
        if fit_intercept:
            X_test_design = [[1.0] + row for row in X_test]
        else:
            X_test_design = [row[:] for row in X_test]
        
        y_pred = matmul(X_test_design, beta_hat)
        
        mse = sum((y_pred[idx] - y_test[idx])**2 for idx in range(len(test_idx))) / len(y_test)
        mse_list.append(mse)
    
    cv_score = sum(mse_list) / k
    return cv_score