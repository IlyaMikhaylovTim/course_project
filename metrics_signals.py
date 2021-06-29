import scipy.stats as ss
import pandas as pd
import numpy as np

SIZE = 1000
N_USERS = 1000
N_ITERATIONS = 1000
eps = 1e-8



# signals

def expon(param, n_users=N_USERS):
    return ss.expon.rvs(loc=param, size=n_users)

def expon2(param, n_users=N_USERS):
    return ss.expon.rvs(loc=param, size=n_users)

def n_queries(param, n_users=N_USERS):
    return ss.poisson.rvs(mu=param, size=n_users)

def n_sessions(param, n_users=N_USERS):
    return ss.poisson.rvs(mu=param, size=n_users)

def n_sessions_with_ads_clicks(param, n_users=N_USERS):
    return ss.poisson.rvs(mu=param, size=n_users)

def page_load_time(param, n_users=N_USERS):
    return ss.lognorm.rvs(s=0.3, loc=param, size=n_users)

def log_record_size(param, n_users=N_USERS):
    return ss.poisson.rvs(mu=param, size=n_users)

def time_to_click(param, n_users=N_USERS):
    return ss.lognorm.rvs(s=0.8, loc=param, size=n_users)

def time_spent(param, n_users=N_USERS):
    return ss.lognorm.rvs(s=0.9, loc=param, size=n_users)




# metrics

def ctr(data):
    return data.iloc[:, 0:2].min(axis=1) / (data.iloc[:, 0:2].max(axis=1) + eps)

def clip_ctr(data):
    return np.clip(data.iloc[:, 0:2].min(axis=1) / (data.iloc[:, 0:2].max(axis=1) + eps), 0, 1)

def queries_per_user(data):
    return data.iloc[:, 2]

def sessions_per_user(data):
    return data.iloc[:, 3]

def queries_per_session(data):
    return np.clip(data.iloc[:, 2] / (data.iloc[:, 3] + eps), 0, 1)

def ads_click_rate(data):
    return np.clip(data.iloc[:, 4] / (queries_per_user(data) + eps), 0, 1)

def page_load_time_per_user(data):
    return data.iloc[:, 5]

def log_record_size_per_user(data):
    return data.iloc[:, 6]

def time_to_click_per_user(data):
    return data.iloc[:, 7]

def time_spent_per_user(data):
    return data.iloc[:, 8]


signals = [expon, expon2, n_queries, n_sessions,
           n_sessions_with_ads_clicks, page_load_time,
           log_record_size, time_to_click, time_spent]

metrics = [ctr, clip_ctr, queries_per_user, sessions_per_user,
           queries_per_session, ads_click_rate, page_load_time_per_user,
           log_record_size_per_user, time_to_click_per_user,
           time_spent_per_user]

