# Load modules
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.datasets import make_blobs

from kmm import KMM
from generalized_kmm import Generalized_KMM




# Set random state
random_state = 10
np.random.seed(random_state)


# Parameters
n_samples = [200, 1000, 1000, 300]
test_size = 0.33
cluster_centers = np.array([[-2.0, 5.0], [2.0, 5.0], [-1.0, 0.5], [4.5, 2.0]])
n_components = cluster_centers.shape[0]
cluster_std = [0.6, 0.6, 0.9, 0.6]




# Generate dataset
X, y = make_blobs(n_samples = n_samples, cluster_std = cluster_std, 
                  centers = cluster_centers, random_state = random_state)

train_sample_location = np.where(y < n_components // 2)[0]
test_sample_location = np.where(y >= n_components // 2)[0]

X_train, train_idx = X[train_sample_location], y[train_sample_location]
X_test, test_idx = X[test_sample_location], y[test_sample_location]




# Returns the weights given by KMM & G-KMM
def get_sample_weights(X_train, X_test, 
                       sigma_KMM = None, sigma_G_KMM = None, 
                       B_KMM = 1000, B_G_KMM = 1000, 
                       plot_scale = [0.1, 15]):
    KMM_opt = KMM(sigma = sigma_KMM, B = B_KMM)
    kmm_weights = KMM_opt.fit_predict(X_train, X_test)
    kmm_weights_size = plot_scale[0] + kmm_weights * plot_scale[1]
    
    kmm_sigma = KMM_opt.sigma
    if kmm_sigma is not None:
        kmm_sigma = np.round(kmm_sigma, 2)


    GKMM_opt = Generalized_KMM(sigma = sigma_G_KMM, B = B_G_KMM)
    GKMM_opt.fit(X_train = X_train, X_test = X_test, train_idx = train_idx, test_idx = test_idx)
    g_kmm_weights = GKMM_opt.predict(X_train)
    g_kmm_weights_size = plot_scale[0] + g_kmm_weights * plot_scale[1]
    
    g_kmm_sigma = GKMM_opt.sigma
    if g_kmm_sigma is not None:
        g_kmm_sigma = np.round(g_kmm_sigma, 2)


    return kmm_weights_size, g_kmm_weights_size, kmm_sigma, g_kmm_sigma




# Return the KMM & G-KMM weights for multiple experiments
def get_weights_for_experiments(X_train, X_test, sigma_KMM_list, sigma_G_KMM_list, B_KMM_list, B_G_KMM_list):
    kmm_weights_size_list, g_kmm_weights_size_list, sigma_kmm_list, sigma_g_kmm_list = [], [], [], []
    
    for values in zip(sigma_KMM_list, sigma_G_KMM_list, B_KMM_list, B_G_KMM_list):
        w = get_sample_weights(X_train, X_test, *values)
        kmm_weights_size_list.append(w[0])
        g_kmm_weights_size_list.append(w[1])
        sigma_kmm_list.append(w[2])
        sigma_g_kmm_list.append(w[3])

    return kmm_weights_size_list, g_kmm_weights_size_list, sigma_kmm_list, sigma_g_kmm_list





# Retrieve weights for the cluster experiments
sigma_KMM_list = [1.0, 3.5, 2.0, None]
sigma_G_KMM_list = [1.0, 3.5, 2.0, None]
B_KMM_list = 4 * [1000]
B_G_KMM_list = 4 * [1000]
(kmm_weights_size_list, g_kmm_weights_size_list, 
sigma_KMM_list, sigma_G_KMM_list) = get_weights_for_experiments(X_train,
                                                                X_test,
                                                                sigma_KMM_list, 
                                                                sigma_G_KMM_list,
                                                                B_KMM_list, 
                                                                B_G_KMM_list)





# Show clusters
plt.figure(figsize = (14, 10))

for count in range(4):
    plt.subplot(2, 2, count + 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker = '*', s = 22, c = 'white', label = 'train', edgecolors = 'blue')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker = 'o', s = 22, c = 'yellow', label = 'test', edgecolors = 'red', 
                                                                                                linewidth = 1.2)

    plt.scatter(X_train[:, 0], X_train[:, 1], alpha = 0.75, marker = 'd', s = g_kmm_weights_size_list[count], c = 'gray', 
                label = f'train: G-KMM [sigma = {sigma_G_KMM_list[count]}]', edgecolors = 'darkblue')
    plt.scatter(X_train[:, 0], X_train[:, 1], alpha = 0.75, marker = 'p', s = kmm_weights_size_list[count], c = 'black', 
                label = f'train: KMM [sigma = {sigma_KMM_list[count]}]', edgecolors = 'yellow')

    lgnd = plt.legend(prop = { "size": 12 })
    for i in range(4):
        lgnd.legendHandles[i]._sizes = [35]

plt.savefig('./results/clusters.png', dpi = 300)
plt.show()

