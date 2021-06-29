from itertools import chain, combinations
from numpy.random import choice

SIZE = 1000
N_USERS = 1000
N_ITERATIONS = 1000
eps = 1e-8


def powerset(iterable):
    s = list(iterable)
    
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

def choose_indices(indices):
    pws = list(powerset(indices))
    ind = list(choice(pws))
    
    return indices[ind]
    
def powerset_bounded(iterable, k):
    s = list(iterable)
    
    return list(chain.from_iterable(combinations(s, r) for r in range(1, k)))

def choose_indices_bounded(indices, k):
    pws = list(powerset_bounded(indices, k))
    ind = list(choice(pws))
    
    return indices[ind]
    
def print_func_names(lst):
    for i, func in enumerate(lst):
        print(f'{i}:    {func.__name__}')
    print()
    print()

def print_indices(indices):
    for i, ind in enumerate(indices):
        print(f"{i}:     ", ind)
    print()
    print()
