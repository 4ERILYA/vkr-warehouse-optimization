"""
run_experiment.py — главный скрипт запуска полного эксперимента.

Запуск:
    python run_experiment.py

Выполняет:
1) Генерация 100 SKU
2) ABC-XYZ-классификация
3) Кластеризация k-means++ внутри групп
4) Расчёт параметров для 4 политик (P0, P1, P2, P3)
5) Имитационный эксперимент: 30 прогонов на каждую политику
6) Сводка + графики в папке results/

Время выполнения: ~3-5 минут.
"""
import os
import sys
import time

# Добавляем src/ в путь
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

from data_generator import generate_skus_dataset
from abc_xyz import abc_xyz_matrix, classification_summary
from clusterer import (cluster_all_groups, silhouette_per_group,
                          cluster_summary)
from policy import assign_policy
from experiment import compare_policies, format_comparison_table
from reporting import (plot_policy_comparison, plot_metrics_bars,
                          plot_groups_effect_p2_vs_p1,
                          plot_inventory_dynamics,
                          plot_cost_decomposition,
                          save_csv_reports)


# ============================================================
# Параметры эксперимента
# ============================================================
N_SKUS = 100
N_DAYS_HISTORY = 365
N_RUNS = 30
SIM_DAYS = 395       # 30 warmup + 365 observation
WARMUP_DAYS = 30
DATA_SEED = 2025
EXP_BASE_SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'results')


def main():
    start = time.time()

    print("=" * 70)
    print("  ВЫЧИСЛИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ — МНОГОУРОВНЕВАЯ СЦЕНАРНАЯ МОДЕЛЬ")
    print("  И.В. Черников, ВГУ, ФКН, 2026")
    print("=" * 70)
    print()

    # ----- ЭТАП 1 -----
    print("[1/6] Генерация тестового набора SKU ...")
    skus, demand = generate_skus_dataset(n=N_SKUS, T=N_DAYS_HISTORY,
                                            seed=DATA_SEED)
    print(f"      Сгенерировано {len(skus)} SKU, "
          f"история спроса {N_DAYS_HISTORY} дней")

    # ----- ЭТАП 2 -----
    print("\n[2/6] ABC-XYZ-классификация ...")
    skus = abc_xyz_matrix(skus, demand)
    print("      Распределение SKU по группам ABC-XYZ:")
    cls_summary = classification_summary(skus)
    print(cls_summary.to_string())

    # ----- ЭТАП 3 -----
    print("\n[3/6] Вторичная кластеризация k-means++ внутри групп ...")
    skus['cluster'] = cluster_all_groups(skus, k=2)
    print("      Распределение SKU по подкластерам:")
    print(cluster_summary(skus).to_string())

    print("\n      Силуэтные коэффициенты по группам:")
    sils = silhouette_per_group(skus, k=2)
    for g in sorted(sils.keys()):
        s = sils[g]
        s_str = f"{s:.3f}" if not np.isnan(s) else "N/A"
        print(f"        {g}: {s_str}")
    avg_sil = np.nanmean(list(sils.values()))
    print(f"      Среднее значение силуэта: {avg_sil:.3f}")

    # ----- ЭТАП 4 -----
    print("\n[4/6] Расчёт параметров политики P2 и сохранение ...")
    skus_with_p2 = assign_policy(skus, policy='multilevel')
    print("      Сводка параметров P2:")
    print(skus_with_p2[['SS', 'ROP', 'EOQ', 'csl']].describe().round(2)
            .to_string())

    # ----- ЭТАП 5: имитационный эксперимент -----
    print(f"\n[5/6] Имитационный эксперимент: {N_RUNS} прогонов x 4 политики ...")
    print(f"      Длительность: {SIM_DAYS} дней (включая прогрев "
          f"{WARMUP_DAYS} дней)")
    print(f"      Это занимает ~3-5 минут...")
    print()

    summary, full_runs = compare_policies(
        skus, n_runs=N_RUNS, sim_days=SIM_DAYS,
        warmup_days=WARMUP_DAYS, base_seed=EXP_BASE_SEED,
        verbose=True,
    )

    print("\n" + format_comparison_table(summary, full_runs))

    # ----- ЭТАП 6: отчёты и графики -----
    print(f"\n[6/6] Сохранение результатов в {RESULTS_DIR}/ ...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CSV
    save_csv_reports(skus_with_p2, summary, full_runs, RESULTS_DIR)
    print("      ✓ skus_with_params.csv")
    print("      ✓ summary.csv")
    print("      ✓ all_runs.csv")

    # Графики
    plot_policy_comparison(summary,
                              os.path.join(RESULTS_DIR,
                                            'comparison.png'))
    print("      ✓ comparison.png")

    plot_metrics_bars(summary,
                        os.path.join(RESULTS_DIR, 'metrics_bars.png'))
    print("      ✓ metrics_bars.png")

    plot_cost_decomposition(summary,
                              os.path.join(RESULTS_DIR,
                                            'cost_decomposition.png'))
    print("      ✓ cost_decomposition.png")

    plot_groups_effect_p2_vs_p1(skus, full_runs,
                                   os.path.join(RESULTS_DIR,
                                                 'groups_effect.png'))
    print("      ✓ groups_effect.png")

    # Динамика запаса для одной SKU из группы AY (как в ВКР)
    ay_idx = skus_with_p2[skus_with_p2['abc_xyz'] == 'AY'].index
    if len(ay_idx) > 0:
        idx_to_show = ay_idx[0]
        plot_inventory_dynamics(skus_with_p2,
                                   os.path.join(RESULTS_DIR,
                                                 'inventory_dynamics.png'),
                                   sku_idx=skus_with_p2.index.get_loc(
                                       idx_to_show),
                                   sim_days=200, seed=42)
        print("      ✓ inventory_dynamics.png")

    # ----- ЗАВЕРШЕНИЕ -----
    elapsed = time.time() - start
    print()
    print("=" * 70)
    print(f"  ЭКСПЕРИМЕНТ ЗАВЕРШЁН за {elapsed:.1f} сек "
          f"({elapsed / 60:.1f} мин)")
    print(f"  Все результаты в: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
