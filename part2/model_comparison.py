# So sánh các mô hình
def model_comparison(model_list):
    rows = []
    for entry in model_list:
        name   = entry["model_name"]
        y_true = [float(x) for x in entry["y_true"]]
        y_pred = [float(x) for x in entry["y_pred"]]
        n      = len(y_true)

        # MAE
        mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

        # RMSE
        rmse = math.sqrt(sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n)

        # R-squared
        y_mean  = sum(y_true) / n
        ss_res  = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot  = sum((y_true[i] - y_mean)    ** 2 for i in range(n))
        r2      = 1 - ss_res / ss_tot

        rows.append({
            "Mô hình"         : name,
            "MAE"             : round(mae,  4),
            "RMSE"            : round(rmse, 4),
            "R-squared (Test)": round(r2,   4),
        })

    df = pd.DataFrame(rows)
    return df

# Gọi hàm
comparison_table = model_comparison([
    {"model_name": "OLS Full",          "y_pred": y_pred_full,  "y_true": y_test_raw},
    {"model_name": "OLS Chọn Biến",     "y_pred": y_pred_short, "y_true": y_test_raw},
    {"model_name": "Ridge Regression",  "y_pred": y_pred_ridge, "y_true": y_test_raw},
])

print(comparison_table.to_string(index=True))