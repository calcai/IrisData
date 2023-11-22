import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

'''
This program uses the single layer neural network previously implemented and a mean squared error function to implement gradient descent
Gradient descent allows any starting 'weights' from the neural network to continuously learn and improve until it finds boundary with minimum error.
'''


random.seed(10)

data = pd.read_csv('irisdata.csv')
data = data.drop(data[data['species'] == 'setosa'].index)
data = data.reset_index()

weights = np.array([-4.27, 0.51, 1])


def calculate_sigmoid(weights, petal_data):
    weights[0] = weights[0] / weights[2]
    weights[1] = weights[1] / weights[2]
    weights[2] = 1
    x = np.dot(weights, petal_data)
    return (1/(1 + math.exp(-x)))


def mse(data, weights):
    true_class = np.zeros(len(data))
    for index, row in data.iterrows():
        if row['species'] == 'virginica':
            true_class[index] = 1
    sigmoid_class = np.zeros(len(data))
    for index, row in data.iterrows():
        inputs = np.ones(3)
        inputs[1] = row['petal_length']
        inputs[2] = row['petal_width']
        sigmoid_class[index] = calculate_sigmoid(weights, inputs)
    return np.square(np.subtract(sigmoid_class,true_class)).sum()


def compare_boundaries():
    versicolor = []
    virginica = []

    #identifying versicolor and virgnica data points
    for index, rows in data.iterrows():

        if(rows['species'] == 'versicolor'):
            versicolor.append((rows['petal_length'], rows['petal_width']))
        if(rows['species'] == 'virginica'):
            virginica.append((rows['petal_length'], rows['petal_width']))

    def plot_true_classes():
        for f in versicolor:
            plt.plot(f[0], f[1], 'o', color = 'green')

        for f in virginica:
            plt.plot(f[0], f[1], 'o', color = 'orange')
        plt.xlabel("Petal length")
        plt.ylabel("Petal width")
        plt.title("Classes of Irises")

    print(f"Reasonable boundary of weights (-4.27, 0.51, 1): {mse(data, weights)} ")
    weights2 = np.array([-2, 0.7, 1])
    print(f"Unreasonable boundary of weights (-2, 0.7, 1): {mse(data, weights2)} ")
    plot_true_classes()
    x = np.linspace(3, 7, 100)
    y = -weights[1] * x + -weights[0]

    x2 = np.linspace(3, 7, 100)
    y2= -weights2[1] * x + -weights2[0]

    plt.plot(x, y, linestyle='dashed', color='red', label = 'Reasonable Boundary')
    plt.plot(x2, y2, linestyle='dashed', color='blue', label = 'Inaccurate Boundary')
    plt.legend(loc = 'lower left')
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.show()


def compute_gradient(data, weights, epsilon):
    true_class = np.zeros(len(data))
    for index, row in data.iterrows():
        if row['species'] == 'virginica':
            true_class[index] = 1
    sigmoid_class = np.zeros(len(data))
    past_inputs = []
    for index, row in data.iterrows():
        inputs = [1, row['petal_length'], row['petal_width']]
        sigmoid_class[index] = calculate_sigmoid(weights, np.array(inputs))
        past_inputs.append(inputs)
    past_inputs = np.array(past_inputs)
    gradient = np.zeros(3)
    for i in range(3):
        for n in range(len(past_inputs)):
            gradient[i] += ((sigmoid_class[n] - true_class[n]) * (sigmoid_class[n] * (1 - sigmoid_class[n])) * past_inputs[n][i])
    return weights - (epsilon * gradient)

def get_weights(data, weights, epsilon):
    curr_weight = weights
    weights_list = [curr_weight.tolist()]
    while True:
        next_weight = compute_gradient(data, curr_weight, epsilon)
        if abs(mse(data, next_weight) - mse(data, curr_weight)) < 0.001:
            break
        else:
            curr_weight = next_weight
            weights_list.append(next_weight.tolist())
    return weights_list

def plot_gradient(data, weights, epsilon):
    weights_list = get_weights(data, weights, epsilon)
    fig, (init_plot, one_iter, final_plot) = plt.subplots(1,3, figsize = (14,5))
    init_weight = weights_list[0]
    second_weight = weights_list[1]
    final_weight = weights_list[-1]

    sigmoid_virginica = []
    sigmoid_versicolor = []

    for index, row in data.iterrows():
        if calculate_sigmoid(init_weight, np.array([1, row['petal_length'], row['petal_width']])) < 0.5:
            sigmoid_versicolor.append((row['petal_length'], row['petal_width']))
        else:
            sigmoid_virginica.append((row['petal_length'], row['petal_width']))

    for point in sigmoid_virginica:
        init_plot.plot(point[0], point[1], 'o', color = 'blue')
    for point in sigmoid_versicolor:
        init_plot.plot(point[0], point[1], 'o', color = 'green')
    x_vals = np.linspace(3, 7, 100)
    y_vals = -init_weight[1] * x_vals + -init_weight[0]

    init_plot.plot(x_vals, y_vals, linestyle='dashed', color='red')
    init_plot.set_xlabel("Petal length")
    init_plot.set_ylabel("Petal width")
    init_plot.set_title("Initial Plot")

    sigmoid_virginica = []
    sigmoid_versicolor = []

    for index, row in data.iterrows():
        if calculate_sigmoid(second_weight, np.array([1, row['petal_length'], row['petal_width']])) < 0.5:
            sigmoid_versicolor.append((row['petal_length'], row['petal_width']))
        else:
            sigmoid_virginica.append((row['petal_length'], row['petal_width']))

    for point in sigmoid_virginica:
        one_iter.plot(point[0], point[1], 'o', color = 'blue')
    for point in sigmoid_versicolor:
        one_iter.plot(point[0], point[1], 'o', color = 'green')
    x_vals = np.linspace(3, 7, 100)
    y_vals = -second_weight[1] * x_vals + -second_weight[0]

    one_iter.plot(x_vals, y_vals, linestyle='dashed', color='red')
    one_iter.set_xlabel("Petal length")
    one_iter.set_ylabel("Petal width")
    one_iter.set_title("After One Iteration")

    sigmoid_virginica = []
    sigmoid_versicolor = []

    for index, row in data.iterrows():
        if calculate_sigmoid(final_weight, np.array([1, row['petal_length'], row['petal_width']])) < 0.5:
            sigmoid_versicolor.append((row['petal_length'], row['petal_width']))
        else:
            sigmoid_virginica.append((row['petal_length'], row['petal_width']))

    for point in sigmoid_virginica:
        final_plot.plot(point[0], point[1], 'o', color = 'blue')
    for point in sigmoid_versicolor:
        final_plot.plot(point[0], point[1], 'o', color = 'green')
    x_vals = np.linspace(3, 7, 100)
    y_vals = -final_weight[1] * x_vals + -final_weight[0]

    final_plot.plot(x_vals, y_vals, linestyle='dashed', color='red')
    final_plot.set_xlabel("Petal length")
    final_plot.set_ylabel("Petal width")
    final_plot.set_title("Converged Plot")

    plt.show()

#Excercise 4

def run_gradient_descent(data, weights, epsilon):
    weight_list = get_weights(data, weights, epsilon)
    descent_result = weight_list[-1]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (20,5))

    color = ['orange' if x['species'] == 'virginica' else 'green' for index, x in data.iterrows()]
    data.plot.scatter(x = 'petal_length', y = 'petal_width', ax = axes[0], c = color)

    x1 = np.linspace(3, 7, 100)
    y1 = -weights[1] * x1 + -weights[0]

    axes[0].plot(x1, y1, linestyle='dashed', color='red')
    axes[0].set_title('Starting Boundary')
    axes[0].set_xlabel('Petal Length')
    axes[0].set_ylabel('Petal Width')
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error Summed")


    iteration = [i for i in range(len(weight_list))]
    mse_vals = [mse(data, np.array(i)) for i in weight_list]
    axes[1].plot(iteration, mse_vals)
    plt.show()

def random_weights():
    weights = np.array([random.uniform(-6, 0), random.uniform(-1,1), 1])
    return weights

def plot_stages(data, weights, epsilon):
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20,8))
    color = ['orange' if x['species'] == 'virginica' else 'green' for index, x in data.iterrows()]
    data.plot.scatter(x = 'petal_length', y = 'petal_width', ax = axes[0,0], c = color)
    data.plot.scatter(x = 'petal_length', y = 'petal_width', ax = axes[0,1], c = color)
    data.plot.scatter(x = 'petal_length', y = 'petal_width', ax = axes[0,2], c = color)
    weight_list = get_weights(data, weights, epsilon)

    x_1 = np.linspace(3, 7, 100)
    y_1 = -weights[1] * x_1 + -weights[0]
    axes[0, 0].plot(x_1, y_1, linestyle='dashed', color='red')

    x_2 = np.linspace(3, 7, 100)
    y_2 = -weight_list[len(weight_list)//2][1] * x_1 + -weight_list[len(weight_list)//2][0]
    axes[0, 1].plot(x_2, y_2, linestyle='dashed', color='red')

    x_3 = np.linspace(3, 7, 100)
    y_3 = -weight_list[-1][1] * x_1 + -weight_list[-1][0]
    axes[0, 2].plot(x_3, y_3, linestyle='dashed', color='red')

    iteration = [i for i in range(len(weight_list))]
    mse_vals = [mse(data, np.array(i)) for i in weight_list]
    axes[1, 0].plot(iteration, mse_vals)
    axes[1, 1].plot(iteration, mse_vals)
    axes[1, 2].plot(iteration, mse_vals)

    axes[1, 0].plot(0, mse(data, np.array(weight_list[0])), 'bo')
    axes[1, 1].plot(len(weight_list)//2, mse(data, np.array(weight_list[len(weight_list)//2])), 'bo')
    axes[1, 2].plot(len(weight_list), mse(data, np.array(weight_list[len(weight_list) - 1])), 'bo')

    axes[0, 0].set_title('Starting Boundary')
    axes[0, 1].set_title('Halfway Boundary')
    axes[0, 2].set_title('Final Boundary')

    axes[1, 0].set_xlabel('Iterations')
    axes[1, 1].set_xlabel('Iterations')
    axes[1, 2].set_xlabel('Iterations')
    axes[1, 0].set_ylabel('Mean Squared Error Summed')
    axes[1, 1].set_ylabel('Mean Squared Error Summed')
    axes[1, 2].set_ylabel('Mean Squared Error Summed')

    plt.show()



    



options = input("What would you like to do?\n1: find the mean squared error given data point and weights\n2: view example mean squared error instances\n3: calculate next weight\n4: plot iterations of gradient descent (try weights -6, 0.4, 1) epsilon = 0.001\n5: view gradient descent after one iteration and learning curve\n6: view gradient descent from random weights\n")
if options == "1":
    w1 = float(input("Weight 1:"))
    w2 = float(input("Weight 2:"))
    w3 = float(input("Weight 3:"))
    print(mse(data, np.array([w1,w2,w3])))
elif options == "2":
    compare_boundaries()
elif options == "3":
    w1 = float(input("Weight 1:"))
    w2 = float(input("Weight 2:"))
    w3 = float(input("Weight 3:"))
    epsilon = float(input("Epsilon:"))
    print(compute_gradient(data, np.array([w1,w2,w3]), epsilon))
elif options == "4":
    w1 = float(input("Weight 1:"))
    w2 = float(input("Weight 2:"))
    w3 = float(input("Weight 3:"))
    epsilon = float(input("Epsilon:"))
    plot_gradient(data, np.array([w1,w2,w3]), epsilon)
elif options == "5":
    w1 = float(input("Weight 1:"))
    w2 = float(input("Weight 2:"))
    w3 = float(input("Weight 3:"))
    epsilon = float(input("Epsilon:"))
    run_gradient_descent(data, np.array([w1,w2,w3]), epsilon)
elif options == "6":
    epsilon = float(input("Epsilon:"))
    plot_stages(data, random_weights(), epsilon)