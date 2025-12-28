"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import os
import random
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold


SEED = 322
np.random.seed(SEED)
random.seed(SEED)


def mean_iou_1d(y_true_low, y_true_high, y_pred_low, y_pred_high, eps=1e-3):
    y_true_low = np.array(y_true_low)
    y_true_high = np.array(y_true_high)
    y_pred_low = np.array(y_pred_low)
    y_pred_high = np.array(y_pred_high)

    true_w = np.maximum(eps, y_true_high - y_true_low)
    pred_w = np.maximum(eps, y_pred_high - y_pred_low)

    inter = np.maximum(
        0.0,
        np.minimum(y_true_high, y_pred_high) - np.maximum(y_true_low, y_pred_low)
    )
    union = true_w + pred_w - inter
    return np.mean(inter / union)


def create_submission(predictions):
    """
    Создание файла results/submission.csv
    """

    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'

    submission = pd.DataFrame(predictions)
    submission.to_csv(submission_path, index=False)

    print(f"Submission файл сохранен: {submission_path}")

    return submission_path


def main():
    """
    Главная функция программы
    """

    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    train = pd.read_csv('/app/data/train.csv')
    test = pd.read_csv('/app/data/test.csv')

    prod_stats = train.groupby("product_id")[[
        "n_stores", "avg_temperature", "avg_humidity", "avg_wind_level"
    ]].mean()

    scaler_prod = StandardScaler()
    prod_scaled = scaler_prod.fit_transform(prod_stats)

    kmeans_prod = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    prod_clusters = kmeans_prod.fit_predict(prod_scaled)
    prod_cluster_map = dict(zip(prod_stats.index, prod_clusters))

    train["prod_cluster"] = train["product_id"].map(prod_cluster_map)
    test["prod_cluster"] = (
        test["product_id"].map(prod_cluster_map).fillna(-1).astype(int)
    )

    cat_cols = [
        "product_id", "management_group_id", "first_category_id",
        "second_category_id", "third_category_id", "dow", "month", "prod_cluster"
    ]

    num_cols = [
        "day_of_month", "week_of_year", "n_stores", "holiday_flag",
        "activity_flag", "precpt", "avg_temperature",
        "avg_humidity", "avg_wind_level"
    ]

    mid = (train["price_p05"] + train["price_p95"]) / 2
    width = train["price_p95"] - train["price_p05"]

    y_mid = np.log1p(mid)
    y_w = np.log1p(width)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(train[num_cols])
    X_test_num = scaler.transform(test[num_cols])

    pca = PCA(n_components=min(8, X_num.shape[1]), random_state=SEED)
    X_pca = pca.fit_transform(X_num)
    X_test_pca = pca.transform(X_test_num)

    X_full = pd.concat(
        [train[cat_cols].reset_index(drop=True), pd.DataFrame(X_pca)],
        axis=1
    )
    X_test_full = pd.concat(
        [test[cat_cols].reset_index(drop=True), pd.DataFrame(X_test_pca)],
        axis=1
    )

    cat_features_idx = list(range(len(cat_cols)))

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    mid_preds = np.zeros(len(X_test_full))
    w_preds = np.zeros(len(X_test_full))
    val_iou_list = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_full)):
        X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
        y_mid_tr, y_mid_val = y_mid.iloc[tr_idx], y_mid.iloc[val_idx]
        y_w_tr, y_w_val = y_w.iloc[tr_idx], y_w.iloc[val_idx]

        model_mid = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            loss_function='RMSE',
            random_seed=SEED,
            verbose=0
        )
        model_mid.fit(X_tr, y_mid_tr, cat_features=cat_features_idx)
        mid_val_pred = np.expm1(model_mid.predict(X_val))
        mid_test_pred = np.expm1(model_mid.predict(X_test_full))

        model_w = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            loss_function='Quantile:alpha=0.95',
            random_seed=SEED,
            verbose=0
        )
        model_w.fit(X_tr, y_w_tr, cat_features=cat_features_idx)
        w_val_pred = np.expm1(model_w.predict(X_val))
        w_test_pred = np.expm1(model_w.predict(X_test_full))

        low_val = mid_val_pred - w_val_pred / 2
        high_val = mid_val_pred + w_val_pred / 2
        true_low_val = np.expm1(y_mid_val) - np.expm1(y_w_val) / 2
        true_high_val = np.expm1(y_mid_val) + np.expm1(y_w_val) / 2

        val_iou = mean_iou_1d(
            true_low_val, true_high_val, low_val, high_val
        )
        val_iou_list.append(val_iou)
        print(f"Fold {fold + 1} IoU: {val_iou:.4f}")

        mid_preds += mid_test_pred / kf.n_splits
        w_preds += w_test_pred / kf.n_splits

    print(f"CV mean IoU: {np.mean(val_iou_list):.4f}")

    low_test = mid_preds - w_preds / 2
    high_test = mid_preds + w_preds / 2
    low_test, high_test = np.minimum(low_test, high_test), np.maximum(low_test, high_test)

    predictions = {
        "row_id": test["row_id"].values,
        "price_p05": low_test,
        "price_p95": high_test
    }

    create_submission(predictions)

    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
