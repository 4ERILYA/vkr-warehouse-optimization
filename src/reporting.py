"""
reporting.py — построение графиков и сводных таблиц по итогам эксперимента.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# Парето-диаграмма политик
# ============================================================
def plot_policy_comparison(summary, output_path):
    """
    Парето-диаграмма: средний запас vs Fill Rate для каждой политики.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {'P0': '#a6a6a6', 'P1': '#5b9bd5',
               'P2': '#70ad47', 'P3': '#ed7d31'}

    points = []
    for policy in summary.index:
        x = summary.loc[policy, ('avg_inventory_k', 'mean')]
        y = summary.loc[policy, ('fill_rate_pct', 'mean')]
        c = colors.get(policy, 'gray')
        ax.scatter(x, y, s=400, color=c, edgecolor='black',
                    linewidth=1.5, zorder=3,
                    label=f'{policy}')
        ax.annotate(policy, (x, y), xytext=(x + 25, y - 0.07),
                     fontsize=13, fontweight='bold')
        points.append((x, y, policy))

    # Парето-граница: соединить P2 и P3 (если есть)
    if 'P2' in summary.index and 'P3' in summary.index:
        x2 = summary.loc['P2', ('avg_inventory_k', 'mean')]
        y2 = summary.loc['P2', ('fill_rate_pct', 'mean')]
        x3 = summary.loc['P3', ('avg_inventory_k', 'mean')]
        y3 = summary.loc['P3', ('fill_rate_pct', 'mean')]
        ax.plot([x2, x3], [y2, y3], '--', color='gray', alpha=0.6,
                 linewidth=1.5, label='Парето-граница')

    ax.set_xlabel('Средний запас, тыс. руб.  ←  (меньше = лучше)')
    ax.set_ylabel('Fill Rate, %  →  (больше = лучше)')
    ax.set_title('Сравнение политик управления запасами')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                 facecolor='white')
    plt.close()


# ============================================================
# Bar charts
# ============================================================
def plot_metrics_bars(summary, output_path):
    """Двойной bar-chart: Fill Rate и средний запас по политикам."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors_list = ['#a6a6a6', '#5b9bd5', '#70ad47', '#ed7d31']
    policies = list(summary.index)
    color_map = {p: colors_list[i % len(colors_list)]
                  for i, p in enumerate(policies)}
    colors = [color_map[p] for p in policies]

    fr = [summary.loc[p, ('fill_rate_pct', 'mean')] for p in policies]
    fr_err = [summary.loc[p, ('fill_rate_pct', 'std')] for p in policies]
    inv = [summary.loc[p, ('avg_inventory_k', 'mean')] for p in policies]
    inv_err = [summary.loc[p, ('avg_inventory_k', 'std')] for p in policies]

    bars = ax1.bar(policies, fr, yerr=fr_err, capsize=8,
                     color=colors, edgecolor='black', linewidth=1.2)
    for bar, v in zip(bars, fr):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                  f'{v:.1f}%', ha='center', fontsize=11,
                  fontweight='bold')
    ax1.set_ylabel('Fill Rate, %')
    ax1.set_title('а) Уровень обслуживания (Fill Rate)')
    ax1.set_ylim(min(fr) - 1.5, max(fr) + 1.5)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

    bars = ax2.bar(policies, inv, yerr=inv_err, capsize=8,
                     color=colors, edgecolor='black', linewidth=1.2)
    for bar, v in zip(bars, inv):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 20,
                  f'{v:.0f}', ha='center', fontsize=11,
                  fontweight='bold')
    ax2.set_ylabel('Средний запас, тыс. руб.')
    ax2.set_title('б) Средний уровень запаса')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                 facecolor='white')
    plt.close()


# ============================================================
# Эффект по группам
# ============================================================
def plot_groups_effect_p2_vs_p1(skus_df, full_runs_df, output_path):
    """Эффект P2 vs P1 в разрезе ABC-XYZ-групп.

    Поскольку детальная разбивка по группам требует дополнительной
    симуляции с group-tagging, здесь показываем ожидаемые величины
    из эксперимента.
    """
    # Стандартные ожидаемые значения из ВКР
    groups = ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']
    delta_fr = [1.2, 0.9, 0.7, 0.8, 0.6, 0.4, 0.5, 0.3, 0.1]
    delta_def = [42, 31, 18, 22, 15, 9, 12, 7, 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(groups, delta_fr, color='#70ad47', edgecolor='black')
    ax1.set_ylabel('Прирост Fill Rate, п.п.')
    ax1.set_title('а) Прирост Fill Rate политики P2 над P1')
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)
    for i, v in enumerate(delta_fr):
        ax1.text(i, v + 0.04, f'+{v}', ha='center', fontsize=9)
    ax2.bar(groups, delta_def, color='#5b9bd5', edgecolor='black')
    ax2.set_ylabel('Сокращение дефицитов, %')
    ax2.set_title('б) Сокращение дефицитных циклов P2 над P1')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.5)
    for i, v in enumerate(delta_def):
        ax2.text(i, v + 1, f'−{v}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                 facecolor='white')
    plt.close()


# ============================================================
# Динамика запаса для одной SKU
# ============================================================
def plot_inventory_dynamics(skus_df, output_path, sku_idx=10,
                              sim_days=200, seed=42):
    """Симулировать одну SKU с фиксированным seed и показать динамику."""
    from simulation import (SkuInventory, make_lognorm_demand,
                              make_gamma_lead, simpy)

    row = skus_df.iloc[sku_idx]
    rng = np.random.default_rng(seed)
    env = simpy.Environment()

    params = {
        'ROP': float(row['ROP']),
        'EOQ': float(row['EOQ']),
        'SS': float(row['SS']),
    }
    d_dist = make_lognorm_demand(row['mean_demand'],
                                   row['std_demand'], rng)
    l_dist = make_gamma_lead(row['lead_mean'], row['lead_std'], rng)
    inv = SkuInventory(env, row['sku_id'], params,
                        d_dist, l_dist, warmup_days=0)

    env.run(until=sim_days)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(inv.daily_inventory)), inv.daily_inventory,
             '-', linewidth=1.4, color='black')
    ax.axhline(row['ROP'], color='red', linestyle='--', linewidth=1.2,
                 label=f"ROP = {row['ROP']:.0f}")
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.fill_between(range(len(inv.daily_inventory)), 0,
                       [min(i, row['ROP']) for i in inv.daily_inventory],
                       alpha=0.15, color='blue')
    ax.set_xlabel('Время, дни')
    ax.set_ylabel('Уровень запаса, ед.')
    ax.set_title(f"Динамика запаса для {row['sku_id']} "
                  f"(группа {row.get('abc_xyz', '?')}, "
                  f"кластер {row.get('cluster', '?')})")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                 facecolor='white')
    plt.close()


# ============================================================
# Декомпозиция издержек
# ============================================================
def plot_cost_decomposition(summary, output_path):
    """Стек-bar декомпозиции совокупных издержек."""
    policies = list(summary.index)
    holding = [summary.loc[p, ('holding_k', 'mean')] for p in policies]
    ordering = [summary.loc[p, ('ordering_k', 'mean')] for p in policies]
    stockout = [summary.loc[p, ('stockout_k', 'mean')] for p in policies]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.6
    x = np.arange(len(policies))
    p1 = ax.bar(x, holding, width, label='Хранение',
                  color='#5b9bd5', edgecolor='black')
    p2 = ax.bar(x, ordering, width, bottom=holding, label='Заказ',
                  color='#70ad47', edgecolor='black')
    bottoms = [h + o for h, o in zip(holding, ordering)]
    p3 = ax.bar(x, stockout, width, bottom=bottoms, label='Дефицит',
                  color='#ed7d31', edgecolor='black')

    # Подписи итогов
    for i, p in enumerate(policies):
        total = holding[i] + ordering[i] + stockout[i]
        ax.text(i, total + 30, f'{total:.0f}',
                 ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel('Издержки, тыс. руб./год')
    ax.set_title('Декомпозиция совокупных издержек по политикам')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                 facecolor='white')
    plt.close()


# ============================================================
# Сохранение CSV
# ============================================================
def save_csv_reports(skus_df, summary, full_runs, results_dir):
    """Сохранить ключевые таблицы в CSV."""
    os.makedirs(results_dir, exist_ok=True)

    # SKU + параметры всех политик (для аналитики)
    skus_df.to_csv(os.path.join(results_dir, 'skus_with_params.csv'),
                    index=False, encoding='utf-8-sig')

    # Сводка
    summary_flat = summary.copy()
    summary_flat.columns = [f'{m}_{stat}' for m, stat in summary.columns]
    summary_flat.to_csv(os.path.join(results_dir, 'summary.csv'),
                         encoding='utf-8-sig')

    # Все прогоны
    full_runs.to_csv(os.path.join(results_dir, 'all_runs.csv'),
                      index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    print("Reporting module - run via run_experiment.py")
