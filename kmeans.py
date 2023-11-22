import pandas as pd
import matplotlib.pyplot as plt
import random
import copy 
import numpy as np

'''
This program uses k means clustering to identify and plot classifications of the three species of iris. 
'''

random.seed(4)

data = pd.read_csv('irisdata.csv')
petal_data = data.copy()
petal_data.drop(["sepal_length", "sepal_width", "species"], axis=1, inplace=True)

past_centers = []
past_clusters = []

#this function runs through iterations of k means clustering until the means converge (stop changing)
def kmeans(k: int, data: pd.DataFrame):
    global past_centers
    global past_clusters

    #parse data and identify points as tuples
    points = []
    for index, rows in data.iterrows():
        point = ()
        for i in rows:
            if(isinstance(i, (float, int))):
                point += ((i,))
        points.append(point)

    # randomize k centers
    centers = []
    for i in range(k):
        center = []
        for col in data.columns:
            if data[col].dtype in (float, int):  # Check if the column contains numeric data
                center.append(round(random.uniform(data[col].min(), data[col].max()), 3))
        centers.append(center)

    clusters = [[] for i in range(k)]
    prev_centers = []
    while centers != prev_centers:
        clusters = [[] for i in range(k)]
        prev_centers = copy.deepcopy(centers)

    #classify points
        for point in points:
            #find distance from clusters
            index = -1
            min_dist = float('inf')
            for i, center in enumerate(centers):
                if euclidian_distance(center, point) < min_dist:
                    min_dist = euclidian_distance(center, point)
                    index = i
            clusters[index].append(point) 

        #update centers
        for i, c in enumerate(clusters):
            if c:  # Check if cluster is not empty
                new_center = [sum(a) for a in zip(*c)]
                new_center = [round((x / len(c)), 3) for x in new_center]
                centers[i] = new_center
            else:
                # If cluster is empty, reinitialize the center
                centers[i] = [round(random.uniform(data[col].min(), data[col].max()), 3) for col in data.columns]
        past_centers.append(centers[:])
        past_clusters.append(clusters[:])

def euclidian_distance(center, point):
    sum_of_squares = 0
    for n in range(len(point)):
        sum_of_squares += (center[n] - point[n])**2
    return sum_of_squares

#this function plots distortion, which is the sum of euclidian distances from each point to its associated mean
def plot_distortion(k, data):
    kmeans(3, data)
    distortion_values = []
    for i, centers in enumerate(past_centers):
        total_distortion = 0
        for j, center in enumerate(centers):
            for point in past_clusters[i][j]:
                total_distortion += euclidian_distance(center, point)
        distortion_values.append(total_distortion)

    plt.plot(distortion_values, marker='o')
    plt.title('K-Means Distortion Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Distortion')
    plt.show()

#plots clusters and means
def plot_clusters(k, data):
    kmeans(k, data)
    fig, (init_plot, halfway_plot, final_plot) = plt.subplots(1,3, figsize = (14,5))

    initial_centers = past_centers[0]
    halfway_centers = past_centers[len(past_centers)//2]
    final_centers = past_centers[-1]

    initial_clusters = past_clusters[0]
    halfway_clusters = past_clusters[len(past_centers)//2]
    final_clusters = past_clusters[-1]

    init_plot.set_title('Initial centers')
    init_plot.set_xlabel('Petal Length')
    init_plot.set_ylabel('Petal Width')
    halfway_plot.set_title('Halfway centers')
    halfway_plot.set_xlabel('Petal Length')
    halfway_plot.set_ylabel('Petal Width')
    final_plot.set_title('Final centers')
    final_plot.set_xlabel('Petal Length')
    final_plot.set_ylabel('Petal Width')

    colors = ["blue", "green", "orange"]

    for i, cluster in enumerate(initial_clusters):
        for point in cluster:
            init_plot.plot(point[0], point[1], 'o', color = colors[i])
    for center in initial_centers:
        init_plot.plot(center[0], center[1],'ro')
    
    for i, cluster in enumerate(halfway_clusters):
        for point in cluster:
            halfway_plot.plot(point[0], point[1], 'ro', color = colors[i])
    for center in halfway_centers:
        halfway_plot.plot(center[0], center[1],'ro')

    for i, cluster in enumerate(final_clusters):
        for point in cluster:
            final_plot.plot(point[0], point[1], 'ro', color = colors[i])
    for center in final_centers:
        final_plot.plot(center[0], center[1],'ro')
    
    plt.show()

#plots reasonable decision boundaries based on perpendicular line to midpoint between means
def plot_decision_boundaries(k, data):
    kmeans(k, data)

    centers = past_centers[-1]
    clusters = past_clusters[-1]
    colors = ["blue", "green", "orange"]


    # Plot data points
    for i, cluster in enumerate(clusters):
        for point in cluster:
            plt.plot(point[0], point[1], 'ro', color = colors[i])
    for center in centers:
        plt.plot(center[0], center[1],'ro')

    centers.sort()
    # Plot decision boundaries
    for i in range(k - 1):
        midpoint = [(centers[i][0] + centers[i + 1][0]) / 2, (centers[i][1] + centers[i + 1][1]) / 2]
        slope = -1 / ((centers[i + 1][0] - centers[i][0]) / (centers[i + 1][1] - centers[i][1])) 
        intercept = midpoint[1] - slope * midpoint[0]
        print(slope)
        print(intercept)

        x_vals = np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max(), 100)
        y_vals = slope * x_vals + intercept

        plt.plot(x_vals, y_vals, linestyle='dashed', color='blue')
    # Set y-axis limits based on data range
    plt.ylim(data.iloc[:, 1].min() - 1, data.iloc[:, 1].max() + 1)

    plt.title(f'K-Means Clustering with {k} clusters')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.show()

options = input("What would you like to do?\n1: plot distortion\n2: plot clusters\n3: plot decision boundaries\n")
print(options)
if options == "1":
    k = int(input("How many clusters?"))
    if isinstance(k, int) or k > 3:
        plot_distortion(k, data)
    else:
        print("Please enter an integer of value 3 or less")
elif options == "2":
    k = int(input("How many clusters?"))
    if isinstance(k, int) or k > 3:
        plot_clusters(k, petal_data)
    else:
        print("Please enter an integer of value 3 or less")
elif options == "3":
    k = int(input("How many clusters?"))
    if isinstance(k, int) or k > 3:
        plot_decision_boundaries(k, petal_data)
    else:
        print("Please enter an integer of value 3 or less")
else:
    print("Invalid input, choose a value between 1 and 3")
