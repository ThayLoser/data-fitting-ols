def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def matmul(A, B):
    """Nhân ma trận A với vector hoặc ma trận B"""
    if not A or not B:
        raise ValueError("Ma trận rỗng")
    
    # Nhân ma trận với vector
    if isinstance(B[0], (int, float)):
        return [sum(a * b for a, b in zip(row, B)) for row in A]
    
    # Nhân hai ma trận
    B_t = list(zip(*B))
    return [
        [sum(a * b for a, b in zip(row, col)) for col in B_t]
        for row in A
    ]


def get_inverse(matrix):
    """Tính ma trận nghịch đảo bằng phương pháp Gauss-Jordan"""
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] 
           for i, row in enumerate(matrix)]
    
    for i in range(n):
        # Tìm pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
                
        aug[i], aug[max_row] = aug[max_row], aug[i]
        
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Ma trận suy biến!")
        
        for j in range(2 * n):
            aug[i][j] /= pivot
            
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]
    
    return [row[n:] for row in aug]
