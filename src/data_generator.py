"""
data_generator.py — генерация тестового набора SKU с реалистичными
параметрическими распределениями.

Параметры распределений согласованы с типичными отраслевыми диапазонами,
описанными в Silver, Pyke, Thomas (2017); Chopra, Meindl (2021).
"""
import numpy as np
import pandas as pd


def generate_skus_dataset(n=100, T=365, seed=2025):
    """
    Сгенерировать таблицу SKU и матрицу истории спроса.

    Параметры
    ----------
    n : int
        Число SKU
    T : int
        Длина истории спроса (дни)
    seed : int
        Seed случайности для воспроизводимости

    Возвращает
    ----------
    skus : pd.DataFrame со столбцами:
        sku_id, mean_demand, std_demand, lead_mean, lead_std,
        holding_cost_rate, criticality, price, annual_demand,
        annual_value
    demand_matrix : pd.DataFrame
        Истории спроса (строки = SKU, столбцы = дни)
    """
    rng = np.random.default_rng(seed)

    # Средний спрос: логнормальное распределение
    # Медиана около 30 ед./день для реалистичного асимметричного профиля
    mean_demand = rng.lognormal(mean=2.8, sigma=0.85, size=n)
    mean_demand = np.clip(mean_demand, 1.0, 200.0)

    # Коэффициент вариации спроса: смесь распределений для покрытия
    # стабильных (X), умеренных (Y) и нестабильных (Z) позиций.
    # Сдвигаем к более высоким CV для реалистичного разброса по XYZ.
    cv = rng.beta(1.8, 2.0, size=n) * 0.55 + 0.05
    cv = np.clip(cv, 0.04, 0.65)
    std_demand = mean_demand * cv

    # Срок поставки: гамма-распределение
    # Часть SKU - короткий лаг (отечественные), часть - длинный (импорт)
    # Реальные σ_L составляют 20-40% от L̄ (логистические задержки,
    # таможня, производственные сбои поставщика).
    is_long_lead = rng.random(n) < 0.45
    lead_mean = np.where(
        is_long_lead,
        rng.uniform(20, 50, size=n),     # "длинный" канал (импорт)
        rng.uniform(2, 8, size=n),       # "короткий" канал
    )
    lead_std = lead_mean * rng.uniform(0.20, 0.35, size=n)

    # Стоимость хранения: 15-30% годовых
    holding_cost_rate = np.where(
        is_long_lead,
        rng.uniform(0.15, 0.22, size=n),
        rng.uniform(0.20, 0.30, size=n),
    )

    # Критичность: 1-5 с уклоном в сторону низких значений.
    # Сильно коррелирована со сроком поставки.
    criticality = np.where(
        is_long_lead,
        rng.choice([3, 4, 5], size=n, p=[0.25, 0.40, 0.35]),
        rng.choice([1, 2, 3], size=n, p=[0.50, 0.35, 0.15]),
    )

    # Цены: логнормальные в более скромном диапазоне 30-1500 руб.
    # для получения реалистичных совокупных запасов в районе 800 тыс. руб.
    price = rng.lognormal(mean=4.5, sigma=0.7, size=n)
    price = np.clip(price, 30.0, 1500.0).round(0)

    # Годовой спрос
    annual_demand = mean_demand * 365
    annual_value = annual_demand * price  # вклад в оборот

    # Сборка DataFrame
    skus = pd.DataFrame({
        'sku_id': [f'SKU-{i:03d}' for i in range(1, n + 1)],
        'mean_demand': mean_demand.round(2),
        'std_demand': std_demand.round(2),
        'lead_mean': lead_mean.round(1),
        'lead_std': lead_std.round(2),
        'holding_cost_rate': holding_cost_rate.round(3),
        'criticality': criticality.astype(int),
        'price': price,
        'annual_demand': annual_demand.round(0).astype(int),
        'annual_value': annual_value.round(0).astype(int),
    })

    # История спроса: логнормальная для X/Y, прерывистая для Z
    demand_matrix = np.zeros((n, T))
    for i in range(n):
        m = mean_demand[i]
        s = std_demand[i]
        if m <= 0:
            continue
        cv_i = s / m
        if cv_i > 0.3:
            # Высокая вариативность: добавим прерывистость
            # (часть дней — нулевые продажи)
            zero_prob = min(0.4, (cv_i - 0.3) * 2)
            sigma_ln = np.sqrt(np.log(1 + (cv_i * 1.3) ** 2))
            mu_ln = np.log(m / (1 - zero_prob)) - sigma_ln ** 2 / 2
            raw = rng.lognormal(mu_ln, sigma_ln, size=T)
            mask = rng.random(T) < zero_prob
            raw[mask] = 0
            demand_matrix[i] = raw
        else:
            # Стабильный спрос: чистое логнормальное
            sigma_ln = np.sqrt(np.log(1 + cv_i ** 2))
            mu_ln = np.log(m) - sigma_ln ** 2 / 2
            demand_matrix[i] = rng.lognormal(mu_ln, sigma_ln, size=T)

    demand_df = pd.DataFrame(
        demand_matrix.round(2),
        index=skus['sku_id'],
        columns=[f'day_{t}' for t in range(T)],
    )

    return skus, demand_df


if __name__ == '__main__':
    skus, demand = generate_skus_dataset(n=100, T=365, seed=2025)
    print(f"Generated {len(skus)} SKUs")
    print(f"Demand matrix shape: {demand.shape}")
    print("\nSample of SKUs:")
    print(skus.head(5).to_string())
    print("\nStatistics:")
    print(skus[['mean_demand', 'std_demand', 'lead_mean',
                'holding_cost_rate', 'criticality',
                'price', 'annual_value']].describe().round(2))
