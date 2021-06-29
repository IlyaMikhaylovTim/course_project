import numpy as np
import scipy.stats as ss

SIZE = 1000
N_USERS = 1000
N_ITERATIONS = 1000
eps = 1e-8


def p_norm(params, p=1):
    """Возвращает среднее в смысле Lp"""
    return np.linalg.norm(params, ord=p) / (len(params) ** (1/p))


def modify_params(init_params, transform_functions, indices):
    """Преобразует исходные параметры в итоговые.
    --------
    
    init_params - исходные параметры.
    
    transform functions - для каждого итогового параметра
        задается вид зависимости от исходных параметров.
    
    indices - для каждого итогового параметра задаются
        позиции исходных параметров, от которых он зависит.
        
    Пример: init_params = [0, 1, 5]
            transform_functions = [L1, L2, L4, L1]
            indices = [[0, 1], [2], [1, 2]]
    """
    
    out_params = np.zeros_like(init_params)
    
    for i in range(len(init_params)):
        out_params[i] = transform_functions[i](init_params, indices[i])
    
    return out_params


def L1(init_params, indices):
    params_of_interest = init_params[indices]
    
    return p_norm(params_of_interest, p=1)

def L2(init_params, indices):
    params_of_interest = init_params[indices]
    
    return p_norm(params_of_interest, p=2)

def L4(init_params, indices):
    params_of_interest = init_params[indices]
    
    return p_norm(params_of_interest, p=4)

def geom_mean(init_params, indices):
    params_of_interest = init_params[indices]
    
    return ss.gmean(params_of_interest)

def arifm_mean(init_params, indices):
    params_of_interest = init_params[indices]
    
    return params_of_interest.mean()

