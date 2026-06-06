import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.special import erfinv
from typing import Dict, List, Optional, Union, TypedDict
from utils import matmul
from ols import hat_matrix

def get_residual_diagnostics(X, y, beta_hat):
    """Hàm phụ trợ tính các phần dư phục vụ cho 4 biểu đồ."""
    n = len(y)
    k = len(X[0])
    y_hat = matmul(X, beta_hat)
    if isinstance(y[0], list):
        residuals = [yi[0] - yhi[0] for yi, yhi in zip(y, y_hat)]
    else:
        residuals = [yi - yhi for yi, yhi in zip(y, y_hat)]
    rss = sum(ei**2 for ei in residuals)
    sigma2_hat = rss / (n - k)
    sigma_hat = math.sqrt(sigma2_hat)

    H, _ = hat_matrix(X)
    h_ii = [H[i][i] for i in range(n)]

    std_res = [
        r / (sigma_hat * math.sqrt(max(1.0 - h, 1e-12)))
        for r, h in zip(residuals, h_ii)
    ]
    cooks_d = [
        (sr**2 / k) * (h / max(1.0 - h, 1e-12))
        for sr, h in zip(std_res, h_ii)
    ]

    return {
        'y_hat': y_hat,
        'residuals': residuals,
        'sigma2_hat': sigma2_hat,
        'sigma_hat': sigma_hat,
        'h_ii': h_ii,
        'std_res': std_res,
        'cooks_d': cooks_d,
    }


class ScatterKwargs(TypedDict):
    edgecolors: str
    facecolors: str
    alpha: float
    linewidths: float


_SCATTER_KW: ScatterKwargs = {'edgecolors': 'steelblue', 'facecolors': 'none', 'alpha': 0.75, 'linewidths': 0.9}


def normal_ppf(p: float) -> float:
    """
    Inverse CDF (quantile function) của phân phối chuẩn tắc N(0,1).
    
    Sử dụng công thức: Φ^(-1)(p) = √2 * erfinv(2p - 1)
    
    """
    return math.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def _lowess_simple(x: List[float], y: List[float], frac: float = 0.5) -> List[float]:
    """
    LOWESS (Locally Weighted Scatterplot Smoothing) đơn giản.
    """
    n = len(x)
    r = max(1, int(frac * n))
    y_smooth = []
    for i in range(n):
        dists = sorted(range(n), key=lambda j: abs(x[j] - x[i]))
        neighbors = dists[:r]
        x_nb = [x[j] for j in neighbors]
        y_nb = [y[j] for j in neighbors]
        # Weighted linear fit
        max_d = max(abs(x[j] - x[i]) for j in neighbors) + 1e-12
        w = [(1 - (abs(x[j] - x[i]) / max_d) ** 3) ** 3 for j in neighbors]
        sw = sum(w)
        swx = sum(w[k] * x_nb[k] for k in range(len(neighbors)))
        swy = sum(w[k] * y_nb[k] for k in range(len(neighbors)))
        swxx = sum(w[k] * x_nb[k] ** 2 for k in range(len(neighbors)))
        swxy = sum(w[k] * x_nb[k] * y_nb[k] for k in range(len(neighbors)))
        denom = sw * swxx - swx ** 2
        if abs(denom) < 1e-12:
            y_smooth.append(swy / sw)
        else:
            b1 = (sw * swxy - swx * swy) / denom
            b0 = (swy - b1 * swx) / sw
            y_smooth.append(b0 + b1 * x[i])
    return y_smooth



def _get_fig_ax(ax=None, figsize=(7, 5)):
    """Trả về (fig, ax): tạo mới nếu ax=None, dùng lại nếu đã có."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax

def residuals_vs_fitted_plot(
    y_hat: List[float],
    e: List[float],
    ax=None,
    save_path: Optional[str] = None
):
    """ 
    Residuals vs Fitted Values. 
    
    Dùng để kiểm tra:
    - Linearity: phần dư không có xu hướng hệ thống theo ŷ.
    - Homoscedasticity: phương sai phần dư đồng đều.
    PARAMETERS:
    
    y_hat      : danh sách giá trị dự đoán ŷ
    e          : danh sách phần dư e = y - ŷ
    ax         : matplotlib Axes (tuỳ chọn). Nếu None sẽ tạo figure mới.
    save_path  : đường dẫn lưu ảnh (tuỳ chọn)
    
    """
    fig, ax = _get_fig_ax(ax)
    n = len(y_hat)
 
    ax.scatter(y_hat, e, **_SCATTER_KW, label='Residuals (e_i)')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.2, label='Zero line')
 
    order = sorted(range(n), key=lambda i: y_hat[i])
    xo = [y_hat[i] for i in order]
    eo = [e[i] for i in order]
    y_smooth = _lowess_simple(xo, eo, frac=0.5)
    ax.plot(xo, y_smooth, color='darkred', linewidth=2, label='LOWESS trend')
    ax.axhline(0, color='gray', alpha=0.5, linewidth=0.8)
 
    ax.set_title('(1) Residuals vs Fitted Values', fontsize=13, fontweight='bold')
    ax.set_xlabel('ŷ (Fitted values)', fontsize=11)
    ax.set_ylabel('e = y - ŷ (Residuals)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=9, loc='best')
 
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[residuals_vs_fitted_plot] Đã lưu: {save_path}")
    return fig

def qq_plot(
    std_res: List[float],
    ax=None,
    save_path: Optional[str] = None
):
    """
    Normal Q-Q Plot.
    Dùng để kiểm tra:
    - Normality: phần dư chuẩn hoá có phân phối chuẩn không?
    Parameters:
    std_res  : danh sách phân dư chuẩn hóa ri
    ax       : matplotlib Axes
    save_path: đường dẫn lưu ảnh
    """
    fig, ax = _get_fig_ax(ax)
    n = len(std_res)
 
    std_res_sorted = sorted(std_res)
    theo = [normal_ppf((i - 0.5) / n) for i in range(1, n + 1)]
    ax.scatter(theo, std_res_sorted, **_SCATTER_KW, label='Data quantiles')
 
    q1_idx, q3_idx = int(0.25 * n), int(0.75 * n)
    x_ref = [theo[q1_idx], theo[q3_idx]]
    y_ref = [std_res_sorted[q1_idx], std_res_sorted[q3_idx]]
    slope = (y_ref[1] - y_ref[0]) / (x_ref[1] - x_ref[0] + 1e-12)
    intercept = y_ref[0] - slope * x_ref[0]
    x_line = [min(theo), max(theo)]
    y_line = [slope * xi + intercept for xi in x_line]
    ax.plot(x_line, y_line, color='red', linewidth=2, label='Q1-Q3 reference line')
 
    ax.set_title('(2) Normal Q-Q Plot', fontsize=13, fontweight='bold')
    ax.set_xlabel('q(p): Theoretical quantiles (standard normal)', fontsize=11)
    ax.set_ylabel('r_i: Sample quantiles (standardized residuals)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=9, loc='best')
 
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[qq_plot] Đã lưu: {save_path}")
    return fig

def scale_location_plot(
    y_hat: List[float],
    std_res: List[float],
    ax=None,
    save_path: Optional[str] = None
):
    """Scale-Location
    Dùng để kiểm tra:
    - Homoscedasticity: phần dư chuẩn hoá có phương sai đồng đều không?
    Parameters:
    y_hat      : danh sách giá trị dự đoán ŷ
    std_res     : danh sách phần dư chuẩn hoá ri
    ax         : matplotlib Axes
    save_path  : đường dẫn lưu ảnh
    """
    fig, ax = _get_fig_ax(ax)
    n = len(y_hat)
 
    sqrt_abs_res = [math.sqrt(abs(sr)) for sr in std_res]
    ax.scatter(y_hat, sqrt_abs_res, **_SCATTER_KW, label='√|r_i|')
 
    order2 = sorted(range(n), key=lambda i: y_hat[i])
    xo2 = [y_hat[i] for i in order2]
    yo2 = [sqrt_abs_res[i] for i in order2]
    y_smooth2 = _lowess_simple(xo2, yo2, frac=0.5)
    ax.plot(xo2, y_smooth2, color='darkred', linewidth=2, label='LOWESS trend')
 
    ax.set_title('(3) Scale-Location (Spread-Location)', fontsize=13, fontweight='bold')
    ax.set_xlabel('ŷ (Fitted values)', fontsize=11)
    ax.set_ylabel(r'$\sqrt{|r_i|}$ (Sqrt of abs. std. residuals)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=9, loc='best')
 
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[scale_location_plot] Đã lưu: {save_path}")
    return fig

def cooks_distance_plot(
    cooks_d: List[float],
    n: Optional[int] = None,
    ax=None,
    save_path: Optional[str] = None
):
    """Cook's Distance.
    Dùng để kiểm tra:
    - Influential points: điểm có ảnh hưởng lớn đến mô hình không?
    Parameters:
    cooks_d     : danh sách Cook's distance D_i
    n           : số lượng quan sát
    ax          : matplotlib Axes
    save_path   : đường dẫn lưu ảnh
    """
    fig, ax = _get_fig_ax(ax)
    if n is None:
        n = len(cooks_d)
 
    threshold = 4.0 / n
    colors = ['crimson' if d > threshold else 'steelblue' for d in cooks_d]
    ax.bar(range(n), cooks_d, color=colors, alpha=0.8, width=0.9)
    ax.axhline(threshold, color='red', linestyle=':', linewidth=1.5,
               label=f'Threshold = 4/n = {threshold:.4f}')
 
    for i, d in enumerate(cooks_d):
        if d > threshold:
            ax.annotate(str(i), xy=(i, d), xytext=(0, 3),
                        textcoords='offset points', ha='center',
                        fontsize=7, color='crimson', fontweight='bold')
 
    num_influential = sum(1 for d in cooks_d if d > threshold)
    if num_influential > 0:
        ax.text(0.98, 0.95, f'n_influential = {num_influential}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
 
    ax.set_title("(4) Cooks Distance (Influence)", fontsize=13, fontweight='bold')
    ax.set_xlabel('i: Observation index', fontsize=11)
    ax.set_ylabel("D_i: Cook's distance", fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax.legend(fontsize=9, loc='best')
 
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[cooks_distance_plot] Đã lưu: {save_path}")
    return fig

def residual_plots(
    X: List[List[float]],
    y: List[float],
    beta_hat: List[float],
    save_path: Optional[str] = None
):
    """Gọi lần lượt các biểu đồ
    Parameters:
    X           : ma trận đặc trưng (n x k)
    y           : vector quan sát (n,)
    beta_hat    : vector hệ số OLS ước lượng
    save_path   : thư mục lưu ảnh
    """
    metrics = get_residual_diagnostics(X, y, beta_hat)
    y_hat   = metrics['y_hat']
    e       = metrics['residuals']
    std_res = metrics['std_res']
    cooks_d = metrics['cooks_d']
    n       = len(y)
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plt.subplots_adjust(hspace=0.38, wspace=0.32)
 
    residuals_vs_fitted_plot(y_hat, e,       ax=axes[0, 0])
    qq_plot(std_res,                         ax=axes[0, 1])
    scale_location_plot(y_hat, std_res,      ax=axes[1, 0])
    cooks_distance_plot(cooks_d, n=n,        ax=axes[1, 1])
 
    fig.suptitle(
        'OLS Regression Diagnostic Plots (Residual Analysis)',
        fontsize=15, fontweight='bold', y=0.995
    )
 
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[residual_plots] Đã lưu: {save_path}")
    else:
        plt.show()
 
    return fig