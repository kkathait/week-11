
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from time import time


# ==========================
# Exercise 1: K-means
# ==========================

def kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    centroids = model.cluster_centers_
    labels = model.labels_

    return centroids, labels


# ==========================
# Exercise 2: Diamonds
# ==========================

# Load diamonds dataset
diamonds = sns.load_dataset('diamonds')

# Select the seven numerical columns
diamonds_numeric = diamonds.select_dtypes(include=[np.number])


def kmeans_diamonds(n, k):
    X = diamonds_numeric.iloc[:n].to_numpy()

    return kmeans(X, k)


# ==========================
# Exercise 3: Timer
# ==========================

def kmeans_timer(n, k, n_iter=5):
    times = []

    for i in range(n_iter):
        start = time()

        kmeans_diamonds(n, k)

        end = time()
        times.append(end - start)

    return np.mean(times)


# ==========================
# Bonus: Binary Search
# ==========================

step_count = 0


def bin_search(n):
    global step_count

    step_count = 0

    arr = np.arange(n)
    left = 0
    right = n - 1

    x = n - 1

    while left <= right:
        step_count += 1

        middle = left + (right - left) // 2

        step_count += 1

        if arr[middle] == x:
            return middle

        step_count += 1

        if arr[middle] < x:
            left = middle + 1
        else:
            right = middle - 1

    return -1
