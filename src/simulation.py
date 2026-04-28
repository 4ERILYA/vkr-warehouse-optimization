"""
simulation.py — дискретно-событийная имитационная модель на SimPy.

Класс SkuInventory реализует политику (s, Q) с непрерывным контролем
уровня запаса. Стохастические процессы:
  - спрос: логнормальное распределение
  - срок поставки: гамма-распределение

Требуется библиотека simpy (>= 4.0). Установка: pip install simpy

Если simpy не установлен — используется минимальная встроенная замена
из _simpy_fallback.py (поддерживает только подмножество API,
необходимое для этой модели).
"""
import numpy as np
import pandas as pd

try:
    import simpy
except ImportError:
    import _simpy_fallback as simpy
    import warnings
    warnings.warn(
        "SimPy не установлен — используется встроенный fallback. "
        "Для production рекомендуется: pip install simpy",
        ImportWarning,
    )


def make_lognorm_demand(mean_d, std_d, rng):
    """Логнормальный генератор дневного спроса."""
    if mean_d <= 0:
        return lambda: 0.0
    cv = std_d / mean_d if mean_d > 0 else 0
    sigma_ln = np.sqrt(np.log(1 + cv ** 2))
    mu_ln = np.log(max(mean_d, 1e-6)) - sigma_ln ** 2 / 2
    return lambda: rng.lognormal(mu_ln, sigma_ln)


def make_gamma_lead(mean_L, std_L, rng):
    """Гамма-генератор срока поставки."""
    if std_L <= 0 or mean_L <= 0:
        return lambda: max(1, mean_L)
    shape = (mean_L / std_L) ** 2
    scale = std_L ** 2 / mean_L
    return lambda: rng.gamma(shape, scale)


# ============================================================
# Класс SkuInventory
# ============================================================
class SkuInventory:
    """
    Состояние и логика управления запасом одной SKU.

    Реализует политику (s, Q) с непрерывным контролем уровня запаса:
    при достижении ROP размещается заказ объёмом EOQ, который поступает
    через случайный срок поставки.
    """

    def __init__(self, env, sku_id, params, demand_dist, lead_dist,
                  warmup_days=30):
        self.env = env
        self.id = sku_id
        self.ROP = params['ROP']
        self.EOQ = params['EOQ']
        self.SS = params['SS']
        # Стартовый запас: ROP (мин. покрытие) — модель сразу
        # сталкивается с реальной динамикой пополнения, без избытка
        self.on_hand = params['ROP']
        self.on_order = 0

        self.demand_dist = demand_dist
        self.lead_dist = lead_dist
        self.warmup_days = warmup_days

        # Метрики (собираются после периода прогрева)
        self.demand_total = 0
        self.demand_lost = 0
        self.cycles_total = 0
        self.cycles_with_stockout = 0
        self.current_cycle_stockout = False
        self.daily_inventory = []
        self.orders_placed = 0

        # Запуск процесса спроса
        env.process(self.demand_process())

    def _is_active(self):
        """Период наблюдения (после прогрева)."""
        return self.env.now >= self.warmup_days

    def demand_process(self):
        """Дневная генерация спроса и обработка дефицита."""
        while True:
            yield self.env.timeout(1)  # шаг = 1 день

            d = max(0, int(round(self.demand_dist())))
            served = min(d, self.on_hand)
            unserved = d - served
            self.on_hand -= served

            if self._is_active():
                self.demand_total += d
                if unserved > 0:
                    self.demand_lost += unserved
                    self.current_cycle_stockout = True
                self.daily_inventory.append(self.on_hand)

            # Проверка точки заказа
            if self.on_hand + self.on_order <= self.ROP:
                self.env.process(self.place_order())

    def place_order(self):
        """Размещение заказа и ожидание поставки."""
        Q = self.EOQ
        self.on_order += Q

        if self._is_active():
            self.cycles_total += 1
            self.orders_placed += 1

        lt = max(1, int(round(self.lead_dist())))
        yield self.env.timeout(lt)

        self.on_hand += Q
        self.on_order -= Q

        if self._is_active() and self.current_cycle_stockout:
            self.cycles_with_stockout += 1
            self.current_cycle_stockout = False


# ============================================================
# Запуск симуляции
# ============================================================
def run_simulation(skus_df, sim_days=395, warmup_days=30, seed=42):
    """
    Один имитационный прогон для всего набора SKU.

    Параметры
    ----------
    skus_df : pd.DataFrame
        Таблица SKU с рассчитанными параметрами политики
        (должна содержать SS, ROP, EOQ)
    sim_days : int
        Полная длительность симуляции (дни), включая прогрев
    warmup_days : int
        Длительность периода прогрева (метрики не собираются)
    seed : int
        Seed случайности для воспроизводимости

    Возвращает
    ----------
    pd.DataFrame с метриками по каждой SKU.
    """
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    inventories = {}

    for _, row in skus_df.iterrows():
        params = {
            'ROP': float(row['ROP']),
            'EOQ': float(row['EOQ']),
            'SS': float(row['SS']),
        }
        d_dist = make_lognorm_demand(row['mean_demand'],
                                       row['std_demand'], rng)
        l_dist = make_gamma_lead(row['lead_mean'],
                                   row['lead_std'], rng)
        inv = SkuInventory(env, row['sku_id'], params,
                            d_dist, l_dist, warmup_days=warmup_days)
        inventories[row['sku_id']] = inv

    env.run(until=sim_days)

    # Сбор метрик
    records = []
    for sku_id, inv in inventories.items():
        fr = (1 - inv.demand_lost / inv.demand_total
              if inv.demand_total > 0 else 1.0)
        cs = (1 - inv.cycles_with_stockout / inv.cycles_total
              if inv.cycles_total > 0 else 1.0)
        avg_inv = (float(np.mean(inv.daily_inventory))
                   if inv.daily_inventory else 0.0)

        records.append({
            'sku_id': sku_id,
            'fill_rate': fr,
            'cycle_service_level': cs,
            'avg_inventory': avg_inv,
            'demand_total': inv.demand_total,
            'lost_demand': inv.demand_lost,
            'orders_count': inv.cycles_total,
        })

    return pd.DataFrame(records)


if __name__ == '__main__':
    from data_generator import generate_skus_dataset
    from abc_xyz import abc_xyz_matrix
    from clusterer import cluster_all_groups
    from policy import assign_policy

    skus, demand = generate_skus_dataset(n=100, T=365, seed=2025)
    skus = abc_xyz_matrix(skus, demand)
    skus['cluster'] = cluster_all_groups(skus, k=2)
    skus_p = assign_policy(skus, policy='multilevel')

    print("Running single simulation (P2 policy, 395 days)...")
    result = run_simulation(skus_p, sim_days=395, warmup_days=30, seed=42)

    weighted_fr = ((result['fill_rate'] * skus_p['annual_demand']).sum()
                    / skus_p['annual_demand'].sum())
    inv_value = (result['avg_inventory'] * skus_p['price']).sum()

    print(f"\nResults of one run (seed=42):")
    print(f"  Weighted Fill Rate: {weighted_fr * 100:.2f} %")
    print(f"  Total inventory value: {inv_value / 1000:.0f} k.rub")
    print(f"  Total lost demand: {result['lost_demand'].sum():.0f} units")
    print(f"  Total orders placed: {result['orders_count'].sum()}")
