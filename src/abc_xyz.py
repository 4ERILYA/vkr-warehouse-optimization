"""
abc_xyz.py — ABC-XYZ-классификация складской номенклатуры.

ABC: классификация по кумулятивной доле в обороте (правило Парето).
XYZ: классификация по коэффициенту вариации спроса.
"""
import numpy as np
import pandas as pd


def abc_classify(skus_df, value_col='annual_value',
                 thresh_a=0.80, thresh_b=0.95):
    """
    ABC-классификация по кумулятивной доле в обороте.

    Параметры
    ----------
    skus_df : pd.DataFrame
        Таблица SKU, должна содержать колонку value_col
    value_col : str
        Имя столбца с годовым денежным оборотом SKU
    thresh_a : float
        Верхняя граница группы A (доля). По умолчанию 0.80
    thresh_b : float
        Верхняя граница группы B (доля). По умолчанию 0.95

    Возвращает
    ----------
    pd.Series с метками 'A', 'B', 'C', индексированная как skus_df.
    """
    if value_col not in skus_df.columns:
        raise ValueError(f"Column '{value_col}' not found in skus_df")

    df = skus_df.sort_values(value_col, ascending=False).copy()
    total = df[value_col].sum()
    if total <= 0:
        return pd.Series('C', index=skus_df.index)

    df['cumulative_share'] = df[value_col].cumsum() / total
    df['abc'] = 'C'
    df.loc[df['cumulative_share'] <= thresh_b, 'abc'] = 'B'
    df.loc[df['cumulative_share'] <= thresh_a, 'abc'] = 'A'

    return df.reindex(skus_df.index)['abc']


def xyz_classify(demand_matrix, thresh_x=0.10, thresh_y=0.25):
    """
    XYZ-классификация по коэффициенту вариации потребления.

    Параметры
    ----------
    demand_matrix : pd.DataFrame
        Истории спроса, строки = SKU, столбцы = периоды
    thresh_x : float
        Верхняя граница группы X (CV). По умолчанию 0.10
    thresh_y : float
        Верхняя граница группы Y (CV). По умолчанию 0.25

    Возвращает
    ----------
    pd.Series с метками 'X', 'Y', 'Z'.
    """
    mu = demand_matrix.mean(axis=1)
    std = demand_matrix.std(axis=1)
    cv = std / mu.replace(0, np.nan)

    labels = pd.Series('Z', index=demand_matrix.index)
    labels[cv <= thresh_y] = 'Y'
    labels[cv <= thresh_x] = 'X'
    labels[mu == 0] = 'Z'  # позиции без продаж

    return labels


def abc_xyz_matrix(skus_df, demand_matrix, **kwargs):
    """
    Совмещённая ABC-XYZ-классификация.

    Возвращает pd.DataFrame с дополнительными столбцами:
    'abc', 'xyz', 'abc_xyz' (например, 'AX', 'BY', 'CZ').
    """
    result = skus_df.copy()

    # Согласование индексов: используем sku_id если он есть
    if 'sku_id' in skus_df.columns:
        skus_indexed = skus_df.set_index('sku_id')
        # demand_matrix индексирован по sku_id
        abc_kw = {k: v for k, v in kwargs.items()
                  if k in ('value_col', 'thresh_a', 'thresh_b')}
        xyz_kw = {k: v for k, v in kwargs.items()
                  if k in ('thresh_x', 'thresh_y')}

        abc_labels = abc_classify(skus_indexed, **abc_kw)
        xyz_labels = xyz_classify(demand_matrix, **xyz_kw)

        result = result.copy()
        result['abc'] = result['sku_id'].map(abc_labels)
        result['xyz'] = result['sku_id'].map(xyz_labels)
    else:
        abc_kw = {k: v for k, v in kwargs.items()
                  if k in ('value_col', 'thresh_a', 'thresh_b')}
        xyz_kw = {k: v for k, v in kwargs.items()
                  if k in ('thresh_x', 'thresh_y')}
        result['abc'] = abc_classify(skus_df, **abc_kw)
        result['xyz'] = xyz_classify(demand_matrix, **xyz_kw)

    result['abc_xyz'] = result['abc'] + result['xyz']
    return result


def classification_summary(skus_df):
    """Сводная таблица распределения SKU по группам ABC-XYZ."""
    if 'abc_xyz' not in skus_df.columns:
        raise ValueError("DataFrame must contain 'abc_xyz' column")

    summary = pd.crosstab(skus_df['abc'], skus_df['xyz'],
                           margins=True, margins_name='Итого')
    return summary


if __name__ == '__main__':
    from data_generator import generate_skus_dataset

    skus, demand = generate_skus_dataset(n=100, T=365, seed=2025)
    skus = abc_xyz_matrix(skus, demand)

    print("Classification summary:")
    print(classification_summary(skus))
    print(f"\nTotal SKUs: {len(skus)}")
    print(f"Group AX SKUs: {len(skus[skus['abc_xyz'] == 'AX'])}")
