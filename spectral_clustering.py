import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from numpy import ndarray
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh 
from scipy.cluster.vq import kmeans2
from scipy.cluster.vq import vq

import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

#def spectral(
#    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
#) -> tuple[
#    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
#]:
"""
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """


def spectral(data, labels, params_dict):
    
    sigma_value = params_dict['sigma']
    k_value = params_dict['k']

    
    # Implementation of spectral clustering
    data = np.atleast_2d(data)  # data in 2-dimensional
    
    # Compute the pairwise squared Euclidean distances
    pairwise_distances_sq = np.square(cdist(data, data, 'euclidean'))
    
    # Compute the similarity matrix
    similarity_matrix = np.exp(-pairwise_distances_sq / (2 * sigma_value ** 2))
    np.fill_diagonal(similarity_matrix, 0)
    
    # Compute the degree matrix
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    
    # Compute the Laplacian matrix
    laplacian_matrix = degree_matrix - similarity_matrix
    
   #Compute the first eigenvectors
   #Slicing operation to the corresponding to first k eigenvectors in matrix
    eigenvalues, eigvectors = eigsh(laplacian_matrix, k=k_value + 1, which='SM')
    eigvectors = eigvectors[:, 1:k_value + 1]
    
    #Perform k-means clustering on eigenvectors
    computed_labels = kmeans2(eigvectors, 5, minit='points')[1]
    
    # Compute Adjusted Random Index (ARI)
    ARI = (labels, computed_labels)
    
    # Compute Sum of Squared Error (SSE)
    centroids = kmeans2(eigvectors, 5, minit='points')[0]
    SSE = np.sum((eigvectors - centroids[computed_labels])**2)

    return computed_labels, SSE, ARI, eigenvalues

def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    data = np.load('question1_cluster_data.npy')
    labels = np.load('question1_cluster_labels.npy')

    answers = {'data': data, 'labels': labels}

    groups = {}
    sigmas = np.linspace(0.1, 10, num=10)
    for i, sigma in enumerate(sigmas):
        # Ensure data is 2-dimensional before passing it to spectral function
        computed_labels, SSE, ARI, eigenvalues = spectral(data[0:10000], labels[0:10000], {'sigma': sigma, 'k': 5})
        groups[i] = {"sigma": sigma,  "ARI": ARI,  "SSE": SSE }
        

    answers["cluster parameters"] = groups
    sigmas = [group["sigma"] for group in groups.values()]
    ARI_values = [group["ARI"] for group in groups.values()]
    SSE_values = [group["SSE"] for group in groups.values()]
    
    # Scatter plot ARI value
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(sigmas, ARI_values, c=ARI_values, cmap='viridis')
    plt.xlabel('Sigma')
    plt.ylabel('Adjusted Random Index')
    plt.title('Scatter Plot Colored by ARI')
    plt.colorbar(label='Adjusted Random Index')
    plt.grid(True)

   # Scatter plot SSE value
    plt.subplot(1, 2, 2)
    plt.scatter(sigmas, SSE_values, c=SSE_values, cmap='viridis')
    plt.xlabel('Sigma')
    plt.ylabel('Sum of Squared Error')
    plt.title('Scatter Plot Colored by SSE')
    plt.colorbar(label='Sum of Squared Error')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
      
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}
    
    min_ARI_idx = np.argmin(ARI_values)
    best_sigma_ARI = groups[min_ARI_idx]["sigma"]
    
    plot_ARI = plt.scatter(best_sigma_ARI, ARI_values[min_ARI_idx])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    
    min_SSE_idx = np.argmin(SSE_values)
    best_sigma_SSE = groups[min_SSE_idx]["sigma"]
    plot_SSE = plt.scatter(best_sigma_SSE, SSE_values[min_SSE_idx])
    answers["cluster scatterplot with smallest SSE"] = plot_SSE
    
    
   # Identify the cluster with the lowest value of ARI. This implies
   # that you set the cluster number to 5 when applying the spectral
   # algorithm.

   # Create two scatter plots using `matplotlib.pyplot`` where the two
   # axes are the parameters used, with \sigma on the horizontal axis
   # and \xi and the vertical axis. Color the points according to the SSE value
   # for the 1st plot and according to ARI in the second plot.

   # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
   # Do the same for the cluster with the smallest value of SSE.
   # All plots must have x and y labels, a title, and the grid overlay.

    
   # Plot of the eigenvalues (smallest to largest) as a line plot.
   # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.

    
    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter([1,2,3], [4,5,6])
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE
    
    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    plot_eig = plt.plot([1,2,3], [4,5,6])
    answers["eigenvalue plot"] = plot_eig
    
    # Simulating eigenvalues 
    #np.random.seed(0)  
    eigenvalues_dataset = {
    'Dataset 1': np.sort(np.random.uniform(low=0.5, high=1.0, size=100)),
    'Dataset 2': np.sort(np.random.uniform(low=0.5, high=1.5, size=100)),
    'Dataset 3': np.sort(np.random.uniform(low=0.5, high=2.0, size=100)),
    'Dataset 4': np.sort(np.random.uniform(low=0.5, high=2.5, size=100)),
}

    # Find the largest and smallest eigenvalues for each dataset
    largest_eigenvalues = [np.max(vals) for vals in eigenvalues_dataset.values()]
    smallest_eigenvalues = [np.min(vals) for vals in eigenvalues_dataset.values()]
    datasets = list(eigenvalues_dataset.keys())

    # Plotting the largest and smallest eigenvalues
    plt.figure(figsize=(10, 5))

    # Plotting the largest eigenvalues as a line plot
    plt.plot(datasets, largest_eigenvalues, marker='o', label='Largest Eigenvalues')

    # Plotting the smallest eigenvalues as a line plot
    plt.plot(datasets, smallest_eigenvalues, marker='x', label='Smallest Eigenvalues')

    plt.title('Largest and Smallest Eigenvalues per Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True)
    plt.show()
    
   # Pick the parameters that give the largest value of ARI, and apply these
   # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
   # Calculate mean and standard deviation of ARI for all five datasets.
   
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

      
# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
