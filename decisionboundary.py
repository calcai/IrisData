import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
This program uses linear classification with sigmoid nonlinearity to classify iris data into one of two classes - versicolor and virginica.
The sigmoid nonlinearity function applies a single layer neural network to the input data and helps "smoothen" out the classification, 
    returning values between 0 and 1, where lower values are likely versicolor and higher values are likely virginica.
'''

data = pd.read_csv('irisdata.csv')
data = data.drop(data[data['species'] == 'setosa'].index)
data = data.drop(['sepal_width', 'sepal_length'], axis = 1)

versicolor = []
virginica = []

sigmoid_versicolor = []
sigmoid_virginica = []


#identifying versicolor and virgnica data points
for index, rows in data.iterrows():

    if(rows['species'] == 'versicolor'):
        versicolor.append((rows['petal_length'], rows['petal_width']))
    if(rows['species'] == 'virginica'):
        virginica.append((rows['petal_length'], rows['petal_width']))
    

#plotting versicolor and virginica true points
def plot_true_classes():
    for f in versicolor:
        plt.plot(f[0], f[1], 'o', color = 'green')

    for f in virginica:
        plt.plot(f[0], f[1], 'o', color = 'orange')
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.title("Classes of Irises")
    plt.show()

#classify flowers based on sigmoid value
def sigmoid_nonlinearity(data):
    for index, row in data.iterrows():
        if calculate_sigmoid(row['petal_length'], row['petal_width']) < 0.5:
            sigmoid_versicolor.append((row['petal_length'], row['petal_width']))
        else:
            sigmoid_virginica.append((row['petal_length'], row['petal_width']))

def calculate_sigmoid(x1, x2):
    x0 = 1
    #weights determined by boundary line found in k means clustering
    w0 = -4.27
    w1 = 0.51
    w2 = 1
    dot_prod = (x0 * w0 + x1 * w1 + x2 * w2)
    sigmoid = 1/(1 + np.exp(-dot_prod))
    return sigmoid

#plots a decision boundary for iris dataset based on sigmoid classificaiton
def plot_sigmoid(slope, intercept):
    for point in sigmoid_virginica:
        plt.plot(point[0], point[1], 'o', color='blue')
    for point in sigmoid_versicolor:
        plt.plot(point[0], point[1], 'o', color='green')
    x_vals = np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max(), 100)
    y_vals = slope * x_vals + intercept

    plt.title('Linear Boundary for Iris Classes')
    plt.plot(x_vals, y_vals, linestyle='dashed', color='red')
    plt.plot()
    plt.show()

#3d plot of petal length, width, and resulting simgoid value
def plot_3d_sigmoid():
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    x = np.linspace(-25, 25, 100)
    y = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x, y)
    Z = calculate_sigmoid(X, Y)

    ax.plot_wireframe(X, Y, Z, rstride = 5, cstride = 5)

    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_zlabel('Sigmoid Value')
    ax.set_title('3D Sigmoid Plot')

    ax.view_init(elev=20, azim=30)
    ax.set_zlim(0,1)

    plt.show()

#example classifications
def example_classifier():
    print("Values above 0.5 are classified as Virginica")
    print(f"Petal Length: {data.loc[75, 'petal_length']}, Petal Width: {data.loc[75, 'petal_width']}, Species: {data.loc[75, 'species']} Sigmoid Class: {calculate_sigmoid(data.loc[75, 'petal_length'], data.loc[75, 'petal_width'])}")
    print(f"Petal Length: {data.loc[119, 'petal_length']}, Petal Width: {data.loc[119, 'petal_width']}, Species: {data.loc[119, 'species']} Sigmoid Class: {calculate_sigmoid(data.loc[119, 'petal_length'], data.loc[119, 'petal_width'])}")
    print(f"Petal Length: {data.loc[83, 'petal_length']}, Petal Width: {data.loc[83, 'petal_width']}, Species: {data.loc[83, 'species']} Sigmoid Class: {calculate_sigmoid(data.loc[83, 'petal_length'], data.loc[83, 'petal_width'])}")
    print(f"Petal Length: {data.loc[98, 'petal_length']}, Petal Width: {data.loc[98, 'petal_width']}, Species: {data.loc[98, 'species']} Sigmoid Class: {calculate_sigmoid(data.loc[98, 'petal_length'], data.loc[98, 'petal_width'])}")
    print(f"Petal Length: {data.loc[118, 'petal_length']}, Petal Width: {data.loc[118, 'petal_width']}, Species: {data.loc[118, 'species']} Sigmoid Class: {calculate_sigmoid(data.loc[118, 'petal_length'], data.loc[118, 'petal_width'])}")

options = input("What would you like to do?\n1: plot versicolor and virginica classes\n2: determine sigmoid function given a petal length & width\n3: plot decision boundary for nonlinearity\n4: plot 3d output of neural network over input space\n5: view example classifications\n")
if options == "1":
    plot_true_classes()
elif options == "2":
    petal_length_input = float(input("Petal length:"))
    petal_width_input = float(input("Petal width:"))
    print(calculate_sigmoid(petal_length_input, petal_width_input))
elif options == "3":
    sigmoid_nonlinearity(data)
    plot_sigmoid(-0.51, 4.27)
elif options == "4":
    plot_3d_sigmoid()
elif options == "5":
    example_classifier()