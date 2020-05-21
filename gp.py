import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def extract_data():
    data = open('./input.data')
    data_points = []
    for row in data:
        data_point = [float(value) for value in row.split()]
        data_points.append(data_point)

    data_points = np.array(data_points)
    return data_points

def make_C_matrix(data_points, beta, amplitude, lengthscale, scale_mixture):
    C = []
    num = len(data_points)
    for n in range(num):
        row = []
        for m in range(num):
            kernel = rational_quadratic_kernel(data_points[n][0], data_points[m][0], amplitude, lengthscale, scale_mixture)
            if n == m:
                kernel = kernel + (1/beta)
            row.append(kernel)
        C.append(row)
    C = np.array(C)
    return C

# using rational quadratic kernel
def rational_quadratic_kernel(x_n, x_m, amplitude, lengthscale, scale_mixture):
    return amplitude * ((1 + (((x_n - x_m) ** 2) / (2 * scale_mixture * (lengthscale ** 2)))) ** (-1 * scale_mixture))

# implement Gaussian Process Regression
def gp_regressiong(test, data_points, C, beta, amplitude, lengthscale, scale_mixture):
    result = []
    for x_star in test:
        kernels = []
        for x_index in range(len(data_points)):
            kernel = rational_quadratic_kernel(data_points[x_index][0], x_star, amplitude, lengthscale, scale_mixture)
            kernels.append(kernel)
        kernels = np.array(kernels).reshape(34,1)
        mean = np.dot(np.transpose(kernels), np.dot(np.linalg.inv(C), data_points[:, 1]))
        k_star = rational_quadratic_kernel(x_star, x_star, amplitude, lengthscale, scale_mixture) + (1/beta)
        var = k_star - np.dot(np.transpose(kernels), np.dot(np.linalg.inv(C), kernels))
        result.append([mean[0], var[0][0]])
    result = np.array(result)
    return result

def plot_result(test, result, data_points, state):
    if state == 'origin':
        plt.subplot(2, 1, 1)
    elif state == 'optimal':
        plt.subplot(2, 1, 2)
    plt.xlim((-60, 60))
    plt.fill_between(x=test, y1=result[:, 0] + (1.96 * np.sqrt(result[:, 1])), y2=result[:, 0] - (1.96 * np.sqrt(result[:, 1])), color="gray")
    plt.scatter(data_points[:, 0], data_points[:, 1], s=15, c='blue')
    plt.plot(test, result[:, 0], linestyle='-', color='red')

# using Sequential Least SQuares Programming (SLSQP) for optimization
def SLSQP(theta, data_points, beta):
    C = make_C_matrix(data_points, beta, theta[0], theta[1], theta[2])
    return 0.5 * np.log(np.linalg.det(C)) + 0.5 * np.dot(np.transpose(data_points[:, 1]), np.dot(np.linalg.inv(C), data_points[:, 1])) + 0.5 * len(data_points) * np.log(2 * np.pi)

if __name__ == "__main__":
    data_points = extract_data()

    beta = 5
    amplitude = 1
    lengthscale = 1
    scale_mixture = 1

    C = make_C_matrix(data_points, beta, amplitude, lengthscale, scale_mixture)

    test = np.linspace(-60, 60, 1000)
    result = gp_regressiong(test, data_points, C, beta, amplitude, lengthscale, scale_mixture)
    plot_result(test, result, data_points, 'origin')

    theta = np.array([amplitude, lengthscale, scale_mixture])
    optimal = minimize(SLSQP, theta, args=(data_points, beta), method='SLSQP')

    [amplitude, lengthscale, scale_mixture] = optimal.x

    C = make_C_matrix(data_points, beta, amplitude, lengthscale, scale_mixture)
    result = gp_regressiong(test, data_points, C, beta, amplitude, lengthscale, scale_mixture)
    plot_result(test, result, data_points, 'optimal')

    plt.show()