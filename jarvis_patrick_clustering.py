"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numpy import float32
import pickle


######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def jarvis_patrick(
    data: np.ndarray, labels: np.ndarray, params_dict: dict
) -> tuple[np.ndarray | None, float | None, float | None]:
    """
    Implemdentation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    # Use in a function annotation
    def some_function(data: NDArray[float32]):
    # function implementation
        pass
    
    k = params_dict.get('k', 5)
    smin = params_dict.get('smin', 5)

    
    n = data.shape[0]
    computed_labels = np.zeros(n, dtype=np.int32)
    
    #Compute distances
    distances = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=-1)
    
    #Compute k_nearest neighbors in each point
    k_nearest_neighbors = np.argsort(distances, axis=1)[:, 1:k+1]
    
    #Compute shared nearest neighbors
    snn_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in k_nearest_neighbors[i]:
            snn_matrix[i, j] = 1
            
    #Assign labels on shared nearest neighbors
    cluster_id = 1
    for i in range(n):
        if np.sum(snn_matrix[i]) >= smin:
            computed_labels[i] = cluster_id
            cluster_id += 1
    
    
    # Compute centroids of clusters
    centroids = np.array([data[computed_labels == i].mean(axis=0) for i in range(1, cluster_id)])

    # Check if centroids array is empty
    if len(centroids) == 0:
        return None, None, None


    # Compute Sum of Squared Error (SSE)
    SSE = np.sum((data - centroids[computed_labels - 1])**2)

    
    #Compute ARI
    ARI = (labels, computed_labels)
    
    #computed_labels: NDArray[np.int32] | None = None
    #SSE: float | None = None
    #ARI: float | None = None

    return computed_labels, SSE, ARI

def jarvis_patrick_clustering(data):
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """
    data = np.load('question1_cluster_data.npy')
    labels = np.load('question1_cluster_labels.npy')
    
    answers = {'data': data, 'labels': labels}

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    subset_data = data[0:10000]
    
    
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').
    parameter_pairs = [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (4, 6), (5, 7), (6, 8), (4, 7), (5, 8)]
    
    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    for idx, (k, smin) in enumerate(parameter_pairs, start=1):
        params_dict = {'k':k, 'smin': smin}
        computed_labels, _, _ = jarvis_patrick(subset_data, None, params_dict)
        
        
        answers[f"cluster_labels_{idx}"] = computed_labels
        
    groups = {}

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}
    
    for i in range(10):
        if i == 0:
            subset_data = data[0:10000]
        else:
            subset_data = data[10000*i:10000*(i+1)]
            
        #parameters
        k = 0.1
        smin = 0.1
        
        #parameters and for ARI and SSE in the groups dictionary
        groups[i] = {"k": k, "Smin": smin, "ARI": None, "SSE": None}
        
        
    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}

    # Create two scatter plots 
    # axes are the parameters used, with # \sigma on the horizontal axis
    s_sigma = [0.1, 0.2, 0.3] 
    s_xi = [0.2, 0.3, 0.4]      
    s_sse = [0.5, 0.6, 0.7]    
    s_ari = [0.8, 0.9, 1.0]     
    
    # Find index of maximum ARI and minimum SSE
    max_ari_index = s_ari.index(max(s_ari))
    min_sse_index = s_sse.index(min(s_sse))

    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.
    # Create scatter plot for SSE
    plt.figure(figsize=(8, 6))
    plt.scatter(s_sigma, s_xi, c=s_sse, cmap='viridis')
    plt.colorbar(label='SSE')
    plt.xlabel('k')
    plt.ylabel('Smin')
    plt.title('Scatter Plot of Parameters with SSE')
    plt.grid(True)
    plt.show()
    
    # Create scatter plot for ARI
    plt.figure(figsize=(8, 6))
    plt.scatter(s_sigma, s_xi, c=s_ari, cmap='viridis')
    plt.colorbar(label='ARI')
    plt.xlabel('k')
    plt.ylabel('Smin')
    plt.title('Scatter Plot of Parameters with ARI')
    plt.grid(True)
    plt.show()

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    plt.figure(figsize=(8, 6))
    plt.scatter(s_sigma[max_ari_index], s_xi[max_ari_index], c='red', label='Cluster with largest ARI')
    plt.xlabel('k')
    plt.ylabel('Smin')
    plt.title('Cluster Scatter Plot with Largest ARI')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Do the same for the cluster with the smallest value of SSE.
    plt.figure(figsize=(8, 6))
    plt.scatter(s_sigma[min_sse_index], s_xi[min_sse_index], c='blue', label='Cluster with smallest SSE')
    plt.xlabel('k')
    plt.ylabel('Smin')
    plt.title('Cluster Scatter Plot with Smallest SSE')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter([1,2,3], [4,5,6])
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    
    ARI_values = []
    SSE_values = []

    
    # Compute mean and standard deviation of ARI and SSE
    mean_ARIs = np.mean(ARI_values)
    std_ARIs = np.std(ARI_values)
    mean_SSEs = np.mean(SSE_values)
    std_SSEs = np.std(SSE_values)
    
    #A single float
    answers["mean_ARIs"] = mean_ARIs
    answers["std_ARIs"] = std_ARIs
    answers["mean_SSEs"] = mean_SSEs
    answers["std_SSEs"] = std_SSEs

    return answers
data = np.load('question1_cluster_data.npy')
# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering(data)
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
