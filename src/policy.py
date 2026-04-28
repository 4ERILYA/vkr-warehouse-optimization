"""
policy.py — расчёт параметров политики управления запасами:
страховой запас (SS), точка заказа (ROP), оптимальный размер заказа (EOQ).

Модуль реализует четыре политики:
  - 'uniform'    : единый CSL для всех SKU (P0 или P3)
  - 'abcxyz'     : CSL по ABC-XYZ-группе (P1)
  - 'multilevel' : CSL по 18 подкластерам (P2 - предложенная)
"""
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Сценарная матрица CSL (18 значений)
# ============================================================
CSL_MATRIX = {
    ('AX', 'alpha'): 0.95, ('AX', 'beta'): 0.99,
    ('AY', 'alpha'): 0.95, ('AY', 'beta'): 0.98,
    ('AZ', 'alpha'): 0.93, ('AZ', 'beta'): 0.97,
    ('BX', 'alpha'): 0.93, ('BX', 'beta'): 0.97,
    ('BY', 'alpha'): 0.90, ('BY', 'beta'): 0.95,
    ('BZ', 'alpha'): 0.88, ('BZ', 'beta'): 0.93,
    ('CX', 'alpha'): 0.90, ('CX', 'beta'): 0.93,
    ('CY', 'alpha'): 0.87, ('CY', 'beta'): 0.90,
    ('CZ', 'alpha'): 0.85, ('CZ', 'beta'): 0.88,
}

DEFAULT_CSL = 0.95
ORDER_COST = 1500.0  # руб./заказ — типовое значение


def sigma_ltd(mean_d, std_d, lead_mean, lead_std):
    """
    σ_LTD = sqrt(L*sd^2 + d^2*sl^2).

    Учитывает вариативность спроса и срока поставки.
    """
    return np.sqrt(lead_mean * std_d ** 2 + mean_d ** 2 * lead_std ** 2)


def compute_params(row, policy='multilevel', csl_override=None,
                    order_cost=ORDER_COST):
    """
    Параметры политики для одной SKU.

    Параметры
    ----------
    row : pd.Series
        Строка таблицы SKU с атрибутами
    policy : str
        'multilevel' | 'abcxyz' | 'uniform'
    csl_override : float, optional
        Для 'uniform' — единый CSL
    order_cost : float
        Стоимость размещения одного заказа

    Возвращает
    ----------
    dict с ключами csl, SS, ROP, EOQ.
    """
    if policy == 'multilevel':
        key = (row['abc_xyz'], row['cluster'])
        csl = CSL_MATRIX.get(key, DEFAULT_CSL)
    elif policy == 'abcxyz':
        ka = (row['abc_xyz'], 'alpha')
        kb = (row['abc_xyz'], 'beta')
        csl = (CSL_MATRIX.get(ka, DEFAULT_CSL) +
                CSL_MATRIX.get(kb, DEFAULT_CSL)) / 2
    elif policy == 'uniform':
        csl = csl_override if csl_override is not None else DEFAULT_CSL
    else:
        raise ValueError(f"Unknown policy: {policy}")

    z = stats.norm.ppf(csl)
    sig = sigma_ltd(row['mean_demand'], row['std_demand'],
                     row['lead_mean'], row['lead_std'])
    ss = z * sig
    rop = row['mean_demand'] * row['lead_mean'] + ss
    H = row['holding_cost_rate'] * row['price']
    eoq = np.sqrt(2 * row['annual_demand'] * order_cost / max(H, 1e-9))

    return {
        'csl': csl,
        'SS': ss,
        'ROP': rop,
        'EOQ': max(1.0, eoq),
    }


def assign_policy(skus_df, policy='multilevel', csl_override=None,
                   order_cost=ORDER_COST):
    """
    Применить политику ко всем SKU.

    Возвращает копию skus_df с дополнительными столбцами csl, SS, ROP, EOQ.
    """
    rows = [compute_params(r, policy, csl_override, order_cost)
            for _, r in skus_df.iterrows()]
    params_df = pd.DataFrame(rows, index=skus_df.index)
    return pd.concat([skus_df, params_df], axis=1)


def policy_summary(skus_df):
    """Сводная статистика рассчитанных параметров."""
    cols = ['SS', 'ROP', 'EOQ', 'csl']
    return skus_df[cols].describe().round(2)


if __name__ == '__main__':
    from data_generator import generate_skus_dataset
    from abc_xyz import abc_xyz_matrix
    from clusterer import cluster_all_groups

    skus, demand = generate_skus_dataset(n=100, T=365, seed=2025)
    skus = abc_xyz_matrix(skus, demand)
    skus['cluster'] = cluster_all_groups(skus, k=2)

    # Применим P2 (multilevel)
    skus_p2 = assign_policy(skus, policy='multilevel')
    print("Policy P2 parameters summary:")
    print(policy_summary(skus_p2))

    # Сравним P1 и P2 для нескольких SKU
    skus_p1 = assign_policy(skus, policy='abcxyz')
    print("\nComparison P1 vs P2 (first 5 SKUs):")
    cmp = pd.DataFrame({
        'group': skus['abc_xyz'],
        'cluster': skus['cluster'],
        'CSL_P1': skus_p1['csl'].round(3),
        'CSL_P2': skus_p2['csl'].round(3),
        'SS_P1': skus_p1['SS'].round(0).astype(int),
        'SS_P2': skus_p2['SS'].round(0).astype(int),
    }, index=skus['sku_id'])
    print(cmp.head(10).to_string())
