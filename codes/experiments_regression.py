# Load modules
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import SGDRegressor
from generalized_kmm import Generalized_KMM




# Set random state
random_state = 10
np.random.seed(random_state)


# True function
def f(x, noise = 0.2):
    return np.sinc(x) + noise * np.random.randn(len(x))


# Model which generates predictions
def generate_predictions(X_train, y_train, X_test, y_test, lin, sample_weights = None):
    X_train = np.hstack(X_train)
    y_train = np.hstack(y_train)
    sample_weights = sample_weights[:, 0] if sample_weights is not None else sample_weights
    model = SGDRegressor(max_iter = 5000, tol = 1e-5)
    model.fit(X_train.reshape(-1, 1), y_train, sample_weight = sample_weights)
    lin_preds = model.predict(lin.reshape(-1, 1))

    y_preds = model.predict(np.hstack(X_test).reshape(-1, 1))
    mae = np.round(np.mean(np.abs(np.hstack(y_test) - y_preds)), 2)
    return lin_preds, mae





# Function which generates an experiments
def simulate(N_train, N_test, mean_train, sigma_train, mean_test, sigma_test):
    X_train = [ np.random.normal(mean_tr, sigma_tr, N_tr) for N_tr, mean_tr, sigma_tr in zip(N_train, mean_train, sigma_train) ]
    X_test = [ np.random.normal(mean_te, sigma_te, N_te) for N_te, mean_te, sigma_te in zip(N_test, mean_test, sigma_test) ]
    
    y_train = [ f(X_tr) for X_tr in X_train]
    y_test = [ f(X_te) for X_te in X_test ]

    train_idx = [ i * np.ones_like(X_tr, dtype = int) for i, X_tr in enumerate(X_train) ]
    test_idx = [ i * np.ones_like(X_te, dtype = int) for i, X_te in enumerate(X_test) ]

    return [X_train, train_idx], [X_test, test_idx], y_train, y_test





# Function which computes the density ratio weights
def compute_weights(X_train_inputs, X_test_inputs, 
                        sigma, lambd, 
                        B, eps, 
                        gamma, omega, 
                        mixture_density, alpha):

    X_train, train_idx = X_train_inputs
    X_test, test_idx = X_test_inputs

    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)
    train_idx = np.hstack(train_idx)
    test_idx = np.hstack(test_idx)


    GKMM = Generalized_KMM(
                        sigma = sigma, lambd = lambd, B = B, eps = eps,
                        gamma = gamma, omega = omega,
                        mixture_density = mixture_density, alpha = alpha
                        )

    GKMM.fit(X_train = X_train, X_test = X_test, train_idx = train_idx, test_idx = test_idx)
    density_ratio = GKMM.predict(X_train)


    X = np.hstack((X_train, X_test))
    X_min, X_max = np.min(X), np.max(X)
    lin = np.linspace(X_min, X_max, X.shape[0])

    return density_ratio, lin





# Function which plots the results
def plot_parameter_results(X_train, X_test, y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae, str_title):
    
    n_rows = 1 
    n_cols = 2

    colors_1 = ['black', 'blue', 'green']
    colors_2 = ['yellow', 'red', 'grey']
    colors_3 = ['violet']

    edge_colors_1 = ['black']
    edge_colors_2 = ['blue']
    edge_colors_3 = ['yellow']

    
    count = 1
    plt.figure(figsize = (13, 9))

    plt.subplot(n_rows, n_cols, count)
        
    for k, X_tr in enumerate(X_train):
        sns.kdeplot(X_tr,
                    shade = True, alpha = 0.3, 
                    color = colors_1[k], edgecolor = edge_colors_1[0], 
                    linewidth = 3.5, 
                    label = 'train ' + str(k))
        
    for k, X_te in enumerate(X_test):
        sns.kdeplot(X_te, 
                    shade = True, alpha = 0.5, 
                    color = colors_2[k], edgecolor = edge_colors_2[0], 
                    linewidth = 3.5, 
                    label = 'test ' + str(k))
        
    sns.kdeplot(np.hstack(X_train), 
                    shade = True, alpha = 0.4, 
                    color = colors_3[0], edgecolor = edge_colors_3[0], 
                    linewidth = 3.5, label = 'train: G-KMM', 
                    weights = r[:, 0])
    

    count += 1
    plt.legend(loc = 'best', prop = { "size": 19 })
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel('')



    plt.subplot(n_rows, n_cols, count)
    for k, X_tr in enumerate(X_train):
        plt.plot(X_tr, y_train[k], 'o',
                    alpha = 0.7, ms = 10, 
                    color = colors_1[k], markeredgecolor = edge_colors_1[0], 
                    label = 'train ' + str(k))
        
    for k, X_te in enumerate(X_test):
        plt.plot(X_te, y_test[k], '*',
                    alpha = 0.7, ms = 10,
                    color = colors_2[k], markeredgecolor = edge_colors_2[0], 
                    label = 'test ' + str(k))
        
    plt.plot(lin, preds, label = "preds", lw = 3.5, color = "brown")
    plt.plot(lin, weighted_preds, label = "preds: G-KMM", lw = 3.5, color = "black")
    plt.plot(lin, f(lin, 0), label = "ground truth", lw = 2.7, color = "orange")

    score = f"[MAE = {mae}] & [Weighted MAE = {weighted_mae}]"
    plt.title(score, fontsize = 21)
    plt.legend(loc = 'best', prop = { "size": 19 })
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)


    plt.tight_layout()
    plt.savefig('./results/' + str_title + '.png', dpi = 300)
    plt.show()






# Multiple train datasets
print('Multiple train datasets...\n')
input_train, input_test, y_train, y_test = simulate([200, 150, 100], [30], [-0.5, 0.5, 1.5], [0.1, 0.1, 0.1], [1.0], [0.4])
r, lin = compute_weights(input_train, input_test, 0.1, 1e-5, 1000, 1e-1, None, None, False, None)
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae, 
                       'regression_1--multiple_train')


input_train, input_test, y_train, y_test = simulate([200, 150, 100], [30], [-0.5, 0.5, 1.5], [0.1, 0.1, 0.1], [1.0], [0.4])
r, lin = compute_weights(input_train, input_test, 1.0, 1e-5, 1000, 1e-1, [0.5, 0.2, 0.05], None, True, 0.25) 
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae,
                       'regression_2--multiple_train')


input_train, input_test, y_train, y_test = simulate([200, 150, 100], [30], [-0.5, 0.5, 1.5], [0.1, 0.1, 0.1], [1.0], [0.4])
r, lin = compute_weights(input_train, input_test, 100.0, 1e-5, 1000, 1e-1, [0.05, 0.2, 0.25], None, True, 0.5) 
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae,
                       'regression_3--multiple_train')




# Multiple test datasets
print('Multiple test datasets...\n')
input_train, input_test, y_train, y_test = simulate([300], [100, 100], [1.0], [0.25], [-0.5, 1.5], [0.15, 0.15])
r, lin = compute_weights(input_train, input_test, 100, 1e-5, 1000, 1e-1, None, None, False, None)
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae,
                       'regression_4--multiple_test')


input_train, input_test, y_train, y_test = simulate([300], [100, 100], [1.0], [0.25], [-0.5, 1.5], [0.15, 0.15])
r, lin = compute_weights(input_train, input_test, 100, 1e-2, 1000, 1e-1, [0.25], None, True, 0.75) 
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae,
                       'regression_5--multiple_test')


input_train, input_test, y_train, y_test = simulate([300], [100, 100], [0.5], [0.25], [-0.5, 1.5], [0.15, 0.15])
r, lin = compute_weights(input_train, input_test, 100, 1e-2, 1000, 1e-1, [0.25], None, True, 0.75) 
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae,
                       'regression_6--multiple_test')





# Multiple train and test datasets
print('Multiple train & test datasets...\n')
input_train, input_test, y_train, y_test = simulate([200, 150, 100], [100, 100], 
                                                    [-0.5, 0.5, 1.5], [0.1, 0.1, 0.1], 
                                                    [-0.5, 1.5], [0.15, 0.15])
r, lin = compute_weights(input_train, input_test, 0.1, 1e-5, 1000, 1e-1, None, None, False, None)
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae,
                       'regression_7--multiple_train_and_test')



input_train, input_test, y_train, y_test = simulate([200, 150, 100], [100, 100], 
                                                    [-0.5, 0.5, 1.5], [0.1, 0.1, 0.1], 
                                                    [-0.5, 1.5], [0.15, 0.15])
r, lin = compute_weights(input_train, input_test, 10.0, 1e-5, 1000, 1e-1, None, None, False, None)
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae,
                       'regression_8--multiple_train_and_test')



input_train, input_test, y_train, y_test = simulate([200, 150, 100], [100, 100], 
                                                    [-0.5, 0.5, 1.5], [0.1, 0.1, 0.1], 
                                                    [-0.5, 1.5], [0.15, 0.15])
r, lin = compute_weights(input_train, input_test, 100.0, 1e-5, 1000, 1e-1, None, [0.85, 0.15], True, None)
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae,
                       'regression_9--multiple_train_and_test')



input_train, input_test, y_train, y_test = simulate([200, 150, 100], [100, 100], 
                                                    [-0.5, 0.5, 1.5], [0.1, 0.1, 0.1], 
                                                    [-0.5, 1.5], [0.15, 0.15])
r, lin = compute_weights(input_train, input_test, 100.0, 1e-5, 1000, 1e-1, [0.25, 0.2, 0.05], [0.15, 0.85], True, 0.5)
preds, mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin)
weighted_preds, weighted_mae = generate_predictions(input_train[0], y_train, input_test[0], y_test, lin, r)
plot_parameter_results(input_train[0], input_test[0], y_train, y_test, r, lin, preds, weighted_preds, mae, weighted_mae,
                       'regression_10--multiple_train_and_test')



