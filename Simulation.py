import pandas as pd
import numpy as np
from numpy.random import choice
from utils import print_func_names, print_indices, choose_indices, choose_indices_bounded
from Visualize import Visualize
from params_modification import *
from tqdm.auto import tqdm

from metrics_signals import signals, metrics

SIZE = 1000
N_USERS = 1000
N_ITERATIONS = 1000
eps = 1e-8


class Simulation:
    
    """Класс для симуляции табличных данных.
    
    API сигналов: signal(parameter, size).
    
    API метрик: metric(data),
        где data - данные в виде DataFrame.
    
    Каждая метрика возвращает список: набор
    своих значений для каждого пользователя"""
    
    def __init__(self):
        self.signals = []
        self.metrics = []
        self.params = None
    
    def set_params(self, params):
        self.params = params
    
    def add_signal(self, signal):
        self.signals.append(signal)
    
    def add_signals(self, signals):
        for signal in signals:
            self.add_signal(signal)
    
    def add_metric(self, metric):
        self.metrics.append(metric)
        
    def add_metrics(self, metrics):
        for metric in metrics:
            self.add_metric(metric)
    
    def generate_data(self, n_users=N_USERS):
        self.data = []
        for i, signal in enumerate(self.signals):
            self.data.append(signal(self.params[i], n_users))
        
        self.data = np.array(self.data)
        
    def get_data(self):
        return self.data
    
    def get_data_df(self):
        n_users = self.data.shape[1]
        n_signals = self.data.shape[0]
        
        return pd.DataFrame(self.data.T,
                            columns=[f'signal {i}' for i in range(n_signals)],
                            index=[f'user {i}' for i in range(n_users)])
    
    def conduct_metrics(self):
        data = self.get_data_df()
        metrics_result = []
        
        for metric in self.metrics:
            metrics_result.append(metric(data).mean())
        
        return np.array(metrics_result)


def show_deltas_distribution(init_params, transform_functions, indices, signals, metrics, coeffs):
    params = modify_params(init_params, transform_functions, indices)
    
    metrics_diff = []

    sim = Simulation()
    vis = Visualize()
    
    sim.add_signals(signals)
    sim.set_params(params)
    
    sim.add_metrics(metrics)

    for _ in range(N_ITERATIONS):
        sim.set_params(params)
        sim.generate_data()
        metrics_res1 = sim.conduct_metrics()

        sim.set_params(params * coeffs)
        sim.generate_data()
        metrics_res2 = sim.conduct_metrics()

        metrics_diff.append(metrics_res2 - metrics_res1)

    metrics_diff = np.array(metrics_diff)
    
    vis.draw_deltas(metrics_diff.T)


def init_sim_draw_deltas():
    init_params = np.array([0.1, 0.5, 1, 2, 5, 10, 15, 20, 30])
    functions = [L1, L2, L4, geom_mean, arifm_mean]
    transform_functions = []
    n_params = len(init_params)

    for i in range(n_params):
        transform_functions.append(choice(functions))

    expon_indices = 0
    expon2_indices = 0
    n_queries_indices = 0
    n_sessions_indices = 0
    n_sessions_with_ads_clicks_indices = 0
    page_load_time_indices = 0
    log_record_size_indices = 0
    time_to_click_indices = 0
    time_spent_indices = 0

    indices = [expon_indices, expon2_indices, n_queries_indices, n_sessions_indices,
               n_sessions_with_ads_clicks_indices, page_load_time_indices,
               log_record_size_indices, time_to_click_indices, time_spent_indices]

    for i in range(len(indices)):
        indices[i] = np.array(choose_indices_bounded(np.arange(n_params), 5))

    print("Transform_functions:\n")
    print_func_names(transform_functions)

    print("Indices:\n")
    print_indices(indices)
    
    coeffs_factors = [0.1, 0.3, 0.5, 1.2, 1.5, 2, 5, 10]
    
    for i in tqdm(range(n_params)):
        for coeff_factor in coeffs_factors:
            coeffs = np.ones_like(init_params)
            coeffs[i] *= coeff_factor
            print(f"Factor position: {i}")
            print(f"Coeff factor: {coeff_factor}\n")
            show_deltas_distribution(init_params, transform_functions, indices,
                                     signals, metrics, coeffs)
    
