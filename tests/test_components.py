"""
test_components.py — тесты ключевых модулей программной реализации.

Запуск:
    python -m pytest tests/ -v
или:
    python tests/test_components.py
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'src'))

from data_generator import generate_skus_dataset
from abc_xyz import abc_classify, xyz_classify, abc_xyz_matrix
from clusterer import cluster_group, cluster_all_groups
from policy import (compute_params, assign_policy, sigma_ltd,
                       CSL_MATRIX)
from simulation import run_simulation
from experiment import compare_policies


class TestDataGenerator(unittest.TestCase):
    def test_generates_correct_number(self):
        skus, demand = generate_skus_dataset(n=50, T=180, seed=1)
        self.assertEqual(len(skus), 50)
        self.assertEqual(demand.shape, (50, 180))

    def test_required_columns(self):
        skus, _ = generate_skus_dataset(n=10, T=30, seed=1)
        required = {'sku_id', 'mean_demand', 'std_demand',
                    'lead_mean', 'lead_std',
                    'holding_cost_rate', 'criticality', 'price',
                    'annual_demand', 'annual_value'}
        self.assertTrue(required.issubset(set(skus.columns)))

    def test_reproducibility(self):
        s1, _ = generate_skus_dataset(n=20, T=30, seed=42)
        s2, _ = generate_skus_dataset(n=20, T=30, seed=42)
        # Числовые столбцы должны совпасть
        for col in ['mean_demand', 'lead_mean', 'price']:
            np.testing.assert_array_equal(s1[col].values, s2[col].values)

    def test_value_ranges(self):
        skus, _ = generate_skus_dataset(n=100, T=365, seed=2025)
        self.assertTrue((skus['mean_demand'] > 0).all())
        self.assertTrue((skus['lead_mean'] >= 2).all())
        self.assertTrue((skus['holding_cost_rate'] >= 0.15).all())
        self.assertTrue((skus['holding_cost_rate'] <= 0.30).all())
        self.assertTrue(skus['criticality'].isin([1, 2, 3, 4, 5]).all())


class TestAbcXyz(unittest.TestCase):
    def test_abc_classification(self):
        # 10 SKU, явно различающиеся по обороту
        df = pd.DataFrame({
            'sku_id': [f'S{i}' for i in range(10)],
            'annual_value': [1000, 800, 600, 100, 80, 60,
                              40, 20, 10, 5],
        })
        labels = abc_classify(df)
        # Топ-2 → A, средние → B, остальные → C
        self.assertEqual(labels.iloc[0], 'A')
        # Должны быть все три категории
        self.assertIn('A', labels.values)
        self.assertIn('B', labels.values)
        self.assertIn('C', labels.values)

    def test_xyz_classification(self):
        # SKU-X: стабильный спрос; SKU-Z: нестабильный
        demand = pd.DataFrame({
            'd1': [10, 5],
            'd2': [10, 50],
            'd3': [10, 1],
            'd4': [10, 100],
        }, index=['SKU-X', 'SKU-Z'])
        labels = xyz_classify(demand)
        self.assertEqual(labels['SKU-X'], 'X')
        self.assertEqual(labels['SKU-Z'], 'Z')

    def test_full_classification(self):
        skus, demand = generate_skus_dataset(n=100, T=365, seed=2025)
        skus = abc_xyz_matrix(skus, demand)
        # Должны быть колонки abc, xyz, abc_xyz
        self.assertIn('abc', skus.columns)
        self.assertIn('xyz', skus.columns)
        self.assertIn('abc_xyz', skus.columns)
        # abc_xyz - комбинация
        for _, row in skus.head(5).iterrows():
            self.assertEqual(row['abc_xyz'], row['abc'] + row['xyz'])


class TestClusterer(unittest.TestCase):
    def test_cluster_group_two_clusters(self):
        # 6 SKU явно разделены на 2 кластера по lead_mean
        df = pd.DataFrame({
            'lead_mean': [3, 4, 5, 30, 35, 40],
            'holding_cost_rate': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            'criticality': [1, 2, 1, 5, 5, 4],
        })
        labels = cluster_group(df, k=2, random_state=42)
        # 3 alpha + 3 beta
        self.assertEqual((labels == 'alpha').sum(), 3)
        self.assertEqual((labels == 'beta').sum(), 3)
        # Первые 3 - alpha (короткий L)
        self.assertTrue((labels.iloc[:3] == 'alpha').all())
        # Последние 3 - beta
        self.assertTrue((labels.iloc[3:] == 'beta').all())

    def test_cluster_handles_few_skus(self):
        # 1 SKU - всё в alpha
        df = pd.DataFrame({
            'lead_mean': [10], 'holding_cost_rate': [0.2],
            'criticality': [3],
        })
        labels = cluster_group(df, k=2)
        self.assertEqual(labels.iloc[0], 'alpha')


class TestPolicy(unittest.TestCase):
    def test_csl_matrix_complete(self):
        # Все 18 комбинаций должны быть в матрице
        groups = ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ',
                   'CX', 'CY', 'CZ']
        for g in groups:
            for c in ['alpha', 'beta']:
                self.assertIn((g, c), CSL_MATRIX)
                csl = CSL_MATRIX[(g, c)]
                self.assertGreater(csl, 0.5)
                self.assertLess(csl, 1.0)

    def test_csl_beta_higher_than_alpha(self):
        # CSL для beta всегда выше чем для alpha
        groups = ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ',
                   'CX', 'CY', 'CZ']
        for g in groups:
            self.assertGreaterEqual(CSL_MATRIX[(g, 'beta')],
                                       CSL_MATRIX[(g, 'alpha')])

    def test_sigma_ltd_formula(self):
        # σ²_LTD = L*sd² + d²*sl²
        sig = sigma_ltd(mean_d=100, std_d=10, lead_mean=5, lead_std=1)
        expected = np.sqrt(5 * 100 + 10000 * 1)
        self.assertAlmostEqual(sig, expected, places=4)

    def test_compute_params_multilevel(self):
        row = pd.Series({
            'mean_demand': 100, 'std_demand': 10,
            'lead_mean': 5, 'lead_std': 1,
            'holding_cost_rate': 0.2, 'price': 100,
            'annual_demand': 36500,
            'abc_xyz': 'AX', 'cluster': 'alpha',
        })
        params = compute_params(row, policy='multilevel')
        self.assertEqual(params['csl'], 0.95)  # AX-alpha
        # SS = z*sigma_LTD = 1.645 * sqrt(5*100 + 10000*1)
        # = 1.645 * sqrt(10500) = 1.645 * 102.47 ≈ 168.6
        self.assertAlmostEqual(params['SS'], 168.6, delta=0.5)

    def test_compute_params_uniform(self):
        row = pd.Series({
            'mean_demand': 50, 'std_demand': 10,
            'lead_mean': 4, 'lead_std': 1,
            'holding_cost_rate': 0.25, 'price': 200,
            'annual_demand': 18250,
            'abc_xyz': 'BY', 'cluster': 'alpha',
        })
        # P0: единый CSL=0.95
        p0 = compute_params(row, policy='uniform', csl_override=0.95)
        self.assertEqual(p0['csl'], 0.95)
        # P3: единый CSL=0.99
        p3 = compute_params(row, policy='uniform', csl_override=0.99)
        self.assertEqual(p3['csl'], 0.99)
        # SS_P3 > SS_P0 (более высокий CSL)
        self.assertGreater(p3['SS'], p0['SS'])


class TestSimulation(unittest.TestCase):
    def test_simulation_runs(self):
        skus, demand = generate_skus_dataset(n=10, T=180, seed=1)
        skus = abc_xyz_matrix(skus, demand)
        skus['cluster'] = cluster_all_groups(skus, k=2)
        skus = assign_policy(skus, policy='multilevel')

        result = run_simulation(skus, sim_days=120, warmup_days=20,
                                  seed=42)
        # Должно быть 10 строк
        self.assertEqual(len(result), 10)
        # Fill rate в [0, 1]
        self.assertTrue((result['fill_rate'] >= 0).all())
        self.assertTrue((result['fill_rate'] <= 1).all())

    def test_simulation_reproducibility(self):
        skus, demand = generate_skus_dataset(n=5, T=100, seed=1)
        skus = abc_xyz_matrix(skus, demand)
        skus['cluster'] = cluster_all_groups(skus, k=2)
        skus = assign_policy(skus, policy='multilevel')

        r1 = run_simulation(skus, sim_days=80, warmup_days=10, seed=42)
        r2 = run_simulation(skus, sim_days=80, warmup_days=10, seed=42)
        # При одинаковом seed результаты должны совпасть
        np.testing.assert_array_equal(r1['lost_demand'].values,
                                         r2['lost_demand'].values)


class TestExperiment(unittest.TestCase):
    def test_compare_policies(self):
        skus, demand = generate_skus_dataset(n=20, T=180, seed=1)
        skus = abc_xyz_matrix(skus, demand)
        skus['cluster'] = cluster_all_groups(skus, k=2)

        summary, full = compare_policies(skus, n_runs=3,
                                            sim_days=120,
                                            warmup_days=20,
                                            verbose=False)
        # Все 4 политики должны быть в сводке
        self.assertEqual(len(summary), 4)
        for p in ['P0', 'P1', 'P2', 'P3']:
            self.assertIn(p, summary.index)


if __name__ == '__main__':
    unittest.main(verbosity=2)
