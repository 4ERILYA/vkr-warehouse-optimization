"""
clusterer.py — вторичная кластеризация SKU методом k-means++.

Внутри каждой ABC-XYZ-группы выполняется разбиение SKU на k подкластеров
по нормализованным операционным признакам:
  - lead_mean   (срок поставки)
  - holding_cost_rate (стоимость хранения)
  - criticality (критичность позиции)

По умолчанию k=2: alpha (короткий L, низкая c) и beta (длинный L, высокая c).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


FEATURE_COLS = ['lead_mean', 'holding_cost_rate', 'criticality']


def cluster_group(group_df, k=2, n_init=10, random_state=42,
                   feature_cols=FEATURE_COLS):
    """
    Кластеризация SKU одной ABC-XYZ-группы.

    Параметры
    ----------
    group_df : pd.DataFrame
        Строки SKU одной группы
    k : int
        Число кластеров (по умолчанию 2)
    n_init : int
        Число перезапусков k-means
    random_state : int
        Сид для воспроизводимости

    Возвращает
    ----------
    pd.Series с метками 'alpha' / 'beta' (или 'cluster_i' при k>2).
    """
    if len(group_df) < k:
        # Меньше SKU чем кластеров - все в alpha
        return pd.Series('alpha', index=group_df.index)

    X = group_df[feature_cols].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, init='k-means++',
                n_init=n_init, random_state=random_state)
    raw_labels = km.fit_predict(X_sc)

    # alpha — кластер с меньшим средним lead_mean (нормализованным)
    centroids = km.cluster_centers_
    if k == 2:
        alpha_idx = int(np.argmin(centroids[:, 0]))
        beta_idx = int(np.argmax(centroids[:, 0]))
        label_map = {alpha_idx: 'alpha', beta_idx: 'beta'}
    else:
        # Для k>2: упорядочить по lead_mean и пронумеровать
        order = np.argsort(centroids[:, 0])
        names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        label_map = {int(idx): names[i] if i < len(names)
                     else f'cluster_{i}'
                     for i, idx in enumerate(order)}

    return pd.Series([label_map[l] for l in raw_labels],
                     index=group_df.index)


def cluster_all_groups(skus_df, k=2, group_col='abc_xyz'):
    """
    Кластеризация всех ABC-XYZ-групп независимо.

    Возвращает pd.Series с метками подкластеров.
    """
    if group_col not in skus_df.columns:
        raise ValueError(f"Column '{group_col}' not found")

    result = pd.Series(index=skus_df.index, dtype=str)
    for gname, gdf in skus_df.groupby(group_col):
        result.loc[gdf.index] = cluster_group(gdf, k=k)
    return result


def silhouette_per_group(skus_df, k=2, group_col='abc_xyz',
                          feature_cols=FEATURE_COLS):
    """
    Силуэтный коэффициент для кластеризации в каждой группе.

    Возвращает dict: имя группы -> силуэт.
    """
    out = {}
    for gname, gdf in skus_df.groupby(group_col):
        if len(gdf) < k + 1:
            out[gname] = float('nan')
            continue
        X = gdf[feature_cols].values
        X_sc = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=k, init='k-means++',
                    n_init=10, random_state=42)
        labels = km.fit_predict(X_sc)
        if len(np.unique(labels)) < 2:
            out[gname] = float('nan')
        else:
            out[gname] = float(silhouette_score(X_sc, labels))
    return out


def cluster_summary(skus_df, group_col='abc_xyz', cluster_col='cluster'):
    """Сводная таблица: сколько SKU в каждом подкластере каждой группы."""
    return pd.crosstab(skus_df[group_col], skus_df[cluster_col],
                        margins=True, margins_name='Всего')


if __name__ == '__main__':
    from data_generator import generate_skus_dataset
    from abc_xyz import abc_xyz_matrix

    skus, demand = generate_skus_dataset(n=100, T=365, seed=2025)
    skus = abc_xyz_matrix(skus, demand)
    skus['cluster'] = cluster_all_groups(skus, k=2)

    print("Clustering result by group:")
    print(cluster_summary(skus))

    print("\nSilhouette scores by group:")
    sils = silhouette_per_group(skus, k=2)
    for g, s in sorted(sils.items()):
        print(f"  {g}: {s:.3f}" if not np.isnan(s)
              else f"  {g}: N/A (too few SKUs)")
    avg = np.nanmean(list(sils.values()))
    print(f"  Average: {avg:.3f}")
