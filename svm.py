import csv
import numpy as np
from libsvm.svmutil import *
from libsvm.svm import *
from scipy.spatial.distance import cdist, pdist, squareform

def extract_data(file):
    csvfile = open(file)
    data = csv.reader(csvfile)
    data_points = []
    for row in data:
        data_points.append(row)

    data_points = np.array(data_points).astype(float)
    return data_points

X_train = extract_data('./X_train.csv')
X_test = extract_data('./X_test.csv')
Y_train = extract_data('./Y_train.csv').reshape(5000)
Y_test = extract_data('./Y_test.csv').reshape(2500)

# implement Grid Search
def grid_search(method):
    gamma = np.logspace(-9, 3, 13)
    Cost = np.logspace(-2, 10, 13)
    parameters = []
    for g in gamma:
        for c in Cost:
            if method == 'linear':
                model = svm_train(Y_train, X_train, f'-t 0 -q -g {g} -c {c} -v 5')
            elif method == 'polynomial':
                model = svm_train(Y_train, X_train, f'-t 1 -q -g {g} -c {c} -v 5')
            elif method == 'rbf':
                model = svm_train(Y_train, X_train, f'-t 2 -q -g {g} -c {c} -v 5')

            parameters.append([model, g, c])

    best_parameter = max(parameters)
    print (f'best score {best_parameter[0]}% for {method} with gamma {best_parameter[1]} and cost {best_parameter[2]}')

    if method == 'linear':
        model = svm_train(Y_train, X_train, f'-t 0 -q -g {best_parameter[1]} -c {best_parameter[2]}')
    elif method == 'polynomial':
        model = svm_train(Y_train, X_train, f'-t 1 -q -g {best_parameter[1]} -c {best_parameter[2]}')
    elif method == 'rbf':
        model = svm_train(Y_train, X_train, f'-t 2 -q -g {best_parameter[1]} -c {best_parameter[2]}')
    predict = svm_predict(Y_test, X_test, model)

    return best_parameter

# Using linear kernel & RBF kernel together
def linear_rbf_kernel(X, Train, gamma):
    Size = len(X)
    linear_kernel = np.dot(X, np.transpose(Train))
    rbf_kernel = np.exp(gamma * cdist(X, Train, 'sqeuclidean'))
    kernel = linear_kernel + rbf_kernel
    kernel = np.hstack((np.arange(1, Size+1).reshape(-1,1), kernel))

    return kernel

if __name__ == "__main__":
    # refer to https://www.cnblogs.com/Finley/p/5329417.html
    # linear kernel functions
    model_linear = svm_train(Y_train, X_train, '-t 0 -q')
    predict_linear = svm_predict(Y_test, X_test, model_linear)

    # polynomial kernel functions
    model_polynomial = svm_train(Y_train, X_train, '-t 1 -q')
    predict_polynomial = svm_predict(Y_test, X_test, model_polynomial)

    # RBF kernel functions
    model_rbf = svm_train(Y_train, X_train, '-t 2 -q')
    predict_rbf = svm_predict(Y_test, X_test, model_rbf)

    # doing grid search
    grid_linear = grid_search('linear')

    grid_polynomial = grid_search('polynomial')
    grid_rbf = grid_search('rbf')

    # doing the User Define Kernel
    gamma = -1/4
    X_train_kernel = linear_rbf_kernel(X_train, X_train, gamma)
    linear_rbf_model = svm_train(Y_train, X_train_kernel, '-t 4 -q')

    X_test_kernel = linear_rbf_kernel(X_test, X_train, gamma)
    linear_rbf_predict = svm_predict(Y_test, X_test_kernel, linear_rbf_model)
