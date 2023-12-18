import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from copy import copy
from collections import namedtuple

BasicExercise = namedtuple('BasicExercise', 'arr result')

def __highlight_pos(s):
    return ['background-color: cyan' if v > 0 else '' for v in s]

def __highlight_highbit(s):
    return ['background-color: cyan' if v > 127 else '' for v in s]

def __no_highlight(s):
    return [''] * len(s)

def __highlight_transform(fn, s):
    return ['background-color: cyan' if v > 0 else '' for v in fn(s)]

def __highlight_nums(s):
    colors = {1: 'cyan', 2: 'magenta', 3: 'yellow', 4: 'lightblue'}
    return ['background-color: %s' % colors.get(v, 'white') for v in s]

def __format(arr, result=np.array(0), *, fn=__highlight_pos, show_result=False):
    arr, result = arr.copy(), result.copy()
    varr = result if show_result else arr
    if varr.ndim == 1:
        df = pd.DataFrame(varr.reshape(1, -1))
    else:
        df = pd.DataFrame(varr)
    ret = df.style.apply(fn).set_properties(**{'font-size': '12pt'})
    ret.arr, ret.result = arr, result
    return ret

# Exercise 0
ex0 = np.arange(1, 17).reshape(4, 4) * -1
ex0[1, 1] = 456
ex0[2, 2] = 789
ex0[3, 3] = 555
result = ex0.copy().T
ex0 = __format(ex0, result)

# Exercise 2.1
ex2_1 = np.zeros((5, 5)) - 1
result = np.arange(100, 105).reshape(5, 1)
ex2_1[:, 2:3] = result
ex2_1 = __format(ex2_1, result)

# Exercise 2.2
ex2_2 = np.zeros((5, 5)) - 1
result = np.arange(100, 102).reshape(1, 2)
ex2_2[1:2, 3:5] = result
ex2_2 = __format(ex2_2, result)

# Exercise 2.3
ex2_3 = np.zeros((5, 5)) - 1
result = np.arange(100, 106).reshape(2, 3)
ex2_3[2::2, ::2] = result
ex2_3 = __format(ex2_3, result)

# Exercise 2.4
ex2_4 = np.zeros((5, 5)) - 1
result = np.arange(100, 104).reshape(2, 2)
ex2_4[-2:, -2:] = result
ex2_4 = __format(ex2_4, result)

# Exercise 2.5
ex2_5 = np.zeros((4, 6)) - 1
result = np.arange(100, 104).reshape(2, 2)
ex2_5[-2:, -2:] = result
ex2_5 = __format(ex2_5, result)

# Exercise 2.6
result = ex2_1.arr.copy()
result[:, 4] = result[:, 2]
ex2_6 = __format(ex2_1.arr, result)

# Exercise 2.7
result = ex2_1.arr.copy()
result[0] = [2, 3, 5, 7, 11]
ex2_7 = __format(ex2_1.arr, result, show_result=True)

# Exercise 2.8
result = ex2_1.arr.copy()
result[:] = 999
ex2_8 = __format(ex2_1.arr, result, show_result=True)

# Exercise 2.9
result = ex2_5.arr.copy()
result[:2, :2] = np.arange(100, 104).reshape(2, 2)
ex2_9 = __format(ex2_5.arr, result, show_result=True)

# Exercise 2.10
arr = ex2_3.arr.copy()
result = np.where(arr > 0, arr + 100, arr)
ex2_10 = __format(arr, result, show_result=True)

# Exercise 2.11
arr = np.random.randint(0, 256, 64, dtype=np.uint8).reshape(8, 8)
result = arr << 1 >> 1
assert np.all(result == arr * 2 // 2)
assert np.all(result == np.mod(arr, 128))
ex2_11 = __format(arr, result, fn=__highlight_highbit)

# Exercise 2.12
arr = np.linspace(0, np.pi/2, 12).reshape(1, 12).T
arr = np.hstack((arr, arr+np.pi/2, arr+np.pi, arr+1.5*np.pi))
result = arr.copy()
result[:, :2] = np.sin(arr[:, :2])
result[:, 2:] = np.cos(arr[:, 2:])
ex2_12 = __format(arr, result, fn=__no_highlight)

# Exercise 2.13
class Oslo:
    def __init__(self):
        self._day = np.arange(0, 364, 1.0)
        self.arr = 11 * (-1 * np.cos(2*np.pi*self._day/364) + 1) 
        self.arr += np.random.randn(364) * 2
        self.result = self.arr.reshape(52, 7).mean(axis=1)
        assert np.allclose(self.result, 
                           np.array([self.arr[day:day+7].mean() 
                                     for day in range(0, 364, 7)]))
        
    def __repr__(self):
        return ("HINT:\n"
                "  This object's .arr contains a simulation of daily highs in Oslo\n"
                "  You can visualize these using its .graph attribute")
    
    @property
    def graph(self):
        plt.plot(self._day, self.arr)
        plt.plot(np.arange(6, 365, 7), self.result, "_", color="red")
        plt.title("Simulation of daily highs in Oslo, Norway")

ex2_13 = Oslo()
del Oslo

# Exercise 2.14
class Wallis:
    def __init__(self, N):
        self.update(N)
        
    def __repr__(self):
        return ("HINT:\n"
                "  A good approach is to create an array with the terms of\n"
                "  the product, then global reduction to get the final result.\n"
                "  The .update() method recalculates with more terms\n")
    
    def update(self, N):
        "Recalculate pi using N terms to the product"
        foursq = 4 * np.arange(1, N)**2
        self.arr = foursq / (foursq - 1)
        self.result = 2 * self.arr.prod() 
        return self.result
   
ex2_14 = Wallis(30)
del Wallis

# Exercise 3.1
arr = ex2_12.arr.copy()
arr2 = arr.copy()
arr[np.cos(arr) < 0] = 0
result = np.sin(arr)
res2 = np.where(np.signbit(np.cos(arr2)), 0, np.sin(arr))
assert np.allclose(result, res2)
ex3_1 = __format(ex2_12.arr.copy(), result, 
                 fn=partial(__highlight_transform, lambda s: np.cos(s) > 0))

# Exercise 3.2
arr = ex2_12.arr.copy()
fn = lambda arr: (np.sin(arr) < 0) & (np.cos(arr) < 0)
result = np.sort(arr[fn(arr)])
ex3_2 = __format(arr, result, fn=partial(__highlight_transform, fn))

# Exercise 3.3
ex3_3 = copy(ex3_2)
arr = ex3_2.arr.copy()
ex3_3.arr = arr.copy()
insert = arr[arr > ex3_2.result.max()].min()
arr[(np.sin(arr) < 0) & (np.cos(arr) < 0)] = insert
ex3_3.result = arr

# NumPy Sieve of Erotosthenes
def numpy_sieve(limit):
    is_prime = np.full(limit, True)
    is_prime[:2] = False
    for d in range(2, int(np.sqrt(limit) + 1)):
        if is_prime[d]:
            is_prime[d*d::d] = False
    return is_prime

# Exercise 3.4
is_prime = numpy_sieve(10_000)
arr = np.random.randint(5_000, 10_000, 100).reshape(10, 10)
result = arr[np.where(is_prime[arr] & (arr > 6_000))]
fn = lambda arr, result=result: np.isin(arr, result)
ex3_4 = __format(arr, result, fn=partial(__highlight_transform, fn))


# Exercise 4.1
class Heat:
    def __init__(self, size=10, hot=1, warm=1):
        self.size = size
        self.hot = hot
        self.warm = warm
        self.new()
        self.__initial = self.arr.copy()

    def new(self):
        self.timestep = 0
        N, hot, warm = self.size, self.hot, self.warm
        self.arr = np.zeros(shape=(N, N))
        hotx = np.random.randint(0, N, hot)
        hoty = np.random.randint(0, N, hot)
        warmx = np.random.randint(0, N, warm)
        warmy = np.random.randint(0, N, warm)
        self.arr[(hotx, hoty)] = 310
        self.arr[(warmx, warmy)] = 273
        return self.arr
    
    def reset(self):
        self.arr = self.__initial.copy()
        self.timestep = 0
        return self.arr
        
    def __repr__(self):
        return ("HINT:\n"
                "  This object's .arr contains temperatures across a plate.\n"
                "  You can visualize these using its .graph attribute.\n"
                "  The .reset() method reinitializes a plate.\n"
                "  the .new() method creates a new randomized plate.\n"
                "  The .step() method moves the system forward by timesteps.")

    def step(self, steps=1):
        N = self.size
        for _ in range(steps):
            self.timestep += 1
            env = np.zeros(shape=(N+2, N+2))
            env[1:-1, 1:-1] = self.arr
            # top + bottom + left + right + 4*self
            self.arr[:] = (env[:-2, 1:-1] + env[2:, 1:-1] + 
                           env[1:-1, :-2] + env[1:-1, 2:] +
                           4 * env[1:-1, 1:-1]) / 8
        return self.arr
    
    @property
    def frozen_step(self):
        self.reset()
        N = self.size
        zeros = np.zeros(shape=(N, N))
        while True:
            if np.allclose(self.arr, zeros):
                break
            self.step()
        ts = self.timestep
        self.reset()
        return ts
    
    @property
    def result(self):
        self.step(10)
        _result = self.arr.copy()
        self.reset()
        return _result
    
    @property
    def graph(self):
        N = self.size
        heat = np.zeros(shape=(N+2, N+2))
        heat[1:-1, 1:-1] = self.arr.copy()
        heat[0, -1] = 310
        plt.imshow(np.sqrt(heat), cmap="coolwarm")
        plt.axis('off')
        plt.title("Temperatures at timestep %d (reference @corner)" 
                  % self.timestep)

ex4_1 = Heat(size=10, hot=10, warm=5)
del Heat

# Exercise 5.1
a = np.ones(4).reshape(2, 2)
b, c, d = a * 2, a * 3, a * 4
arr = np.r_[np.c_[a, b], np.c_[c, d]]
result = np.c_[a, b, c, d]
ex5_1 = __format(arr, result, fn=__highlight_nums)

# Exercise 5.2
ex5_2 = __format(ex5_1.result, ex5_1.arr, fn=__highlight_nums)

# Exercise 5.3
ex5_3 = copy(ex5_2)
ex5_3.arr = ex5_2.arr.copy()
ex5_3.result = np.concatenate(
    [ex5_3.arr[:, n:n+2].flatten() for n in range(0, 8, 2)])
 
# Exercise 5.4
# Note, the below is one possible solution, but we just used the 
# arrays calculated in last exercise as .arr and .result
# np.concatenate([arr[n:n+4].reshape(2, 2) for n in range(0, 16, 4)], axis=1)
ex5_4 = __format(ex5_3.result, ex5_3.arr, fn=__highlight_nums)

# Exercise 5.5
ex5_5 = copy(ex5_1)
ex5_5.arr = ex5_1.arr.copy()
ex5_5.result = ex5_5.arr.reshape(2, 2, 4)

# Exercise 5.6
arr = ex5_1.arr.copy()
a, b, c, d = [arr[n:n+2, m:m+2].reshape(1, 2, 2) 
              for n in range(0, 4, 2) for m in range(0, 4, 2)]
result = np.concatenate([a, b, c, d])
ex5_6 = __format(arr, result, fn=__highlight_nums)

# Exercise 5.7
arr = ex5_6.result.copy()
result = arr * np.tile(np.diag([-1, 10]), (4, 1, 1))
ex5_7 = BasicExercise(arr, result)

# Exercise 6.1
"""Ideas for solution:

fields = open('data/wisconsin.csv').readline().strip().split(',')
dtypes = [(field, np.float) for field in fields]
data = np.loadtxt("data/wisconsin.csv", 
                  skiprows=1, 
                  delimiter=",", 
                  dtype=dtypes)
larger_than_median = data[data['mean radius'] > np.median(data['mean radius'])]
concavity_error = larger_than_median['concavity error']
concavity_error.mean(), concavity_error.std()
flat_data = data.view(np.float64).reshape(data.shape + (-1,))
abs_var = np.abs(np.var(flat_data, axis=0))
abs_std = np.abs(np.std(flat_data, axis=0))
val_range = flat_data.max(axis=0) - flat_data.min(axis=0)
fields[np.argmax(abs_std[:-1]/val_range[:-1])]
"""


# Cleanup names we don't want to export
del result
del res2
del arr
del arr2
del insert
del fn
del a, b, c, d