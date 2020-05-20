import numpy as np

from scipy.optimize import minimize
import matplotlib.pyplot as plt

def extract_data():
    data = open('./input.data')
    data_points = []
    for data_point in data:
        data_point = [float(value) for value in data_point.split()]
        data_points.append(data_point)

    data_points = np.array(data_points)
    return data_points

if __name__ == "__main__":
    data_points = extract_data()

    beta = 5
