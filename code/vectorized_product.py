import numpy as np
import time

a = np.random.randn(1000000)
b = np.random.randn(1000000)

def non_vectorized():
    start = time.time()
    z = 0
    for i in range(len(a)):
        z += a[i] * b[i]
    end = time.time()
    return end - start

def vectorized():
    start = time.time()
    z = np.dot(a, b)
    end = time.time()
    return end - start

if __name__ == '__main__':
    nv = []
    v = []
    for i in range(10):
        nv.append(non_vectorized())
        v.append(vectorized())
    print("Average Non-vectorized: ", 1000 * np.mean(nv))
    print("Average Vectorized: ", 10000 * np.mean(v))