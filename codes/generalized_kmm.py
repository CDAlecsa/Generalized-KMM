# Load modules
import itertools
import numpy as np
import cvxpy as cp

from scipy.linalg import sqrtm

from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel







class Generalized_KMM():
    """
    Computes the Generalized KMM method with respect to multiple train & test datasets

    
    Parameters
    ----------
    sigma: float (default = None)
        The parameter of the Gaussian kernel.
        If sigma is 'adaptive' then it is computed using the function 'get_kernel_width'.
        If sigma is 'None' then it is computed with the default options from sklearn's kernels.
    
    lambd: float (default = 1e-5)
        Regularization term for the kernel matrix involved in the quadratic problem.
    
    eps: float (default = 1e-2)
        Constraint parameter.

    B: float (default = 50)
        Bounding value for the parameters of the density ratio model.

    gamma: list (default = None)
        List of weights for the mixture probability corresponding to the train subsets.
    
    omega: list (default = None)
        List of weights for the mixture probability corresponding to the test subsets.

    mixture_density: bool (default = False)
        Boolean flag value which enables the 'alpha' - mixture density ratio case.
                         
    alpha: float (default = None)
        The coefficient corresponding to the 'alpha' - mixture density ratio case.

    use_quad_form: bool (default = True)
        Boolean flag related to the matrix of the quadratic problem.
        If set to 'True' then it wrapps the matrix 'P' of the quadratic problem using the operator 'cvxpy.Parameter'.
        If set to 'False' then it converts the quadratic problem to DPP formulation: 
                    'P_sqrt = cxpy.Parameter((n_tr, n_tr))'
                    'quad_form = cvxpy.sum_squares(P_sqrt @ x)'
                    'P_sqrt.value = scipy.linalg.sqrtm(P).real'

    ignore_dpp: bool (default = True)
        Boolean flag related to the quadratic cvxpy problem.
        'When set to True, DPP problem solved will be treated as non-DPP, which may speed up compilation.'
                 
    verbose: bool (default = False)
        Boolean flag which controls the verbosity when fitting and predicting.
     
    scale_data: bool (default = False)
        If set to 'True' then it applies internally the sklearn's 'Normalizer'.
    
    kernel_type: str (default = 'rbf')
        Choose from the following sklearn's kernel methods: 'rbf' and 'laplacian'.

    

    Dataset related attributes
    --------------------------
    X_train: numpy array
        An array containing non-overlapping numpy train datasets
    
    X_test: numpy array
        An array containing non-overlapping numpy test datasets
    
    train_idx: numpy array
        An array containing sample ids corresponding to the non-overlapping train datasets (labels usually start from 0).
        The corresponding id of each sample is related to which train subset that sample belongs to.
    
    test_idx: numpy array
        An array containing the sample ids corresponding to the non-overlapping test datasets (labels usually start from 0).
        The corresponding id of each sample is related to which test subset that sample belongs to.

    """


    def __init__(self,
                 sigma = None, lambd = 1e-5, eps = 1e-2, B = 50,
                 gamma = None, omega = None,
                 mixture_density = False, alpha = None,
                 use_quad_form = True, ignore_dpp = True,
                 verbose = False, scale_data = False, 
                 kernel_type = 'rbf'):

        self.kernel_type = kernel_type
        self.kernel = self.compute_kernel(kernel_type)

        self.sigma = sigma
        self.lambd = lambd
        self.eps = eps
        self.B = B

        self.gamma = gamma
        self.omega = omega
        
        self.mixture_density = mixture_density
        self.alpha = alpha

        self.use_quad_form = use_quad_form
        self.ignore_dpp = ignore_dpp

        self.verbose = verbose

        self.scale_data = scale_data
        self.scaler = None

        self.trained = False


        # Check conditions
        assert ( self.lambd >= 0 and self.lambd < 1 )
        assert ( self.eps > 0 )

        if mixture_density:
            if ((alpha is not None) and (gamma is None)):
                raise ValueError('Since a value for alpha was given, one must select also the vector gamma!')
            if ((alpha is None) and (gamma is not None)):
                raise ValueError('Since a value for gamma was given, one must select also the scalar alpha!')




    def H(self, j, k):
        idx_j = np.where(self.train_idx == self.n_prime_labels[j])[0] 
        idx_k = np.where(self.train_idx == self.n_prime_labels[k])[0]
        
        K = self.kernel(self.X_train[idx_j, :], self.X_train[idx_k, :], gamma = self.sigma)
        
        H_matrix = K * (self.n_prime_max ** 2) * self.gamma_sq[j, k] / ( 2 * self.n_prime_sq[j, k] )
        assert ( H_matrix.shape[0] == self.n_prime[j] and H_matrix.shape[1] == self.n_prime[k] and H_matrix.ndim == 2 )
        return H_matrix
    


    def h(self, i, j):
        idx_i = np.where(self.test_idx == self.n_labels[i])[0] 
        idx_j = np.where(self.train_idx == self.n_prime_labels[j])[0]
        
        K = self.kernel(self.X_train[idx_j, :], self.X_test[idx_i, :], gamma = self.sigma)
        
        ones = np.ones(shape = (self.n[i], 1))
        K = np.dot(K, ones)
        
        h_vector = K * (self.n_prime_max ** 2) * self.gamma_mult_omega[j, i] / ( self.n_prime_mult_n[j, i] )
        assert ( h_vector.shape[0] == self.n_prime[j] and h_vector.shape[1] == 1 and h_vector.ndim == 2 )
        return h_vector



    def xi(self, X):
        K = self.kernel(X, self.X_test, gamma = self.sigma)
        assert ( K.shape[0] == X.shape[0] and K.shape[1] == self.b and K.ndim == 2 )
        return K



    def A(self, j):
        idx_j = np.where(self.train_idx == self.n_prime_labels[j])[0]
        A_matrix = self.xi(self.X_train[idx_j, :])
        
        assert ( A_matrix.shape[0] == self.n_prime[j] and A_matrix.shape[1] == self.b and A_matrix.ndim == 2 )
        return A_matrix



    def grad_matrix(self):
        """
        The computation of the matrix appearing in the quadratic problem
        """
        idx_lst = list( itertools.product( range(self.N_prime), range(self.N_prime) ) )
        G_matrix = 0.0

        for (j, k) in idx_lst:
            H = self.H(j, k)
            A_j = self.A(j)
            A_k = self.A(k)        
            G_matrix += A_j.T @ H @ A_k

        assert ( G_matrix.shape[0] == self.b and G_matrix.shape[1] == self.b and G_matrix.ndim == 2 )
        return G_matrix



    def grad_vector(self):
        """
        The computation of the vector appearing in the quadratic problem
        """
        idx_lst = list( itertools.product( range(self.N), range(self.N_prime) ) )
        G_vector = 0.0
        for (i, j) in idx_lst:
            h = self.h(i, j)
            A_j = self.A(j)
            G_vector += h.T @ A_j
        
        G_vector = G_vector.T

        assert ( G_vector.shape[0] == self.b and G_vector.shape[1] == 1 and G_vector.ndim == 2 )
        return G_vector



    def regularize(self, K):
        if not self.is_positive_semidefinite(K):
            K += self.lambd * np.identity(self.b)
        if not self.is_symmetric(K):
            K = (1 / 2) * (K + K.T)
        return K

    




    def fit(self, X_train, X_test, train_idx, test_idx):
        """
        Estimates the parameters of the density ratio model.        
        """
        
        # Reshape the datasets if needed
        self.X_train = self.reshape_vector(X_train)
        self.X_test = self.reshape_vector(X_test)

        # Apply scaling
        if self.scale_data:
            self.scaler = Normalizer()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

        # Retain the indices of train & test subsets
        self.train_idx = train_idx
        self.test_idx = test_idx

        # In the case of the alpha mixture density, add the whole test set to the training list
        if self.mixture_density:
            self.X_train = np.vstack([self.X_train, self.X_test])
            self.train_idx = np.hstack([ self.train_idx, 
                                        ( np.max(np.unique(self.train_idx)) + 1 ) * np.ones_like(self.test_idx) ])
            

        # Define the lengths of each train & test subset
        self.n_prime_labels, self.n_prime = np.unique(self.train_idx, return_counts = True)
        self.n_labels, self.n = np.unique(self.test_idx, return_counts = True)

        # Define the number of train & test datasets
        self.N_prime = len(self.n_prime)
        self.N = len(self.n)

        # Define the coefficients related to the frequencies of each given train & test subset
        self.gamma = self.n_prime / np.sum(self.n_prime) if self.gamma is None else np.array(self.gamma)        
        self.omega = self.n / np.sum(self.n) if self.omega is None else np.array(self.omega)

        # In the case of the alpha mixture density, add the alpha value to the list of train frequencies
        if self.mixture_density and self.alpha is not None:
            self.gamma = np.hstack([self.gamma, self.alpha])
                
        # Define the length of the largest training subset
        self.n_prime_max = max(self.n_prime)

        # Define the multiplications related to the lengths of train & test subsets
        self.n_prime_sq = self.n_prime[:, None] @ self.n_prime[:, None].T
        self.n_prime_mult_n = self.n_prime[:, None] @ self.n[:, None].T

        # Define the multiplications related to the frequency coefficients
        self.gamma_sq = self.gamma[:, None] @ self.gamma[:, None].T
        self.gamma_mult_omega = self.gamma[:, None] @ self.omega[:, None].T

        # Define the number of parameters
        self.b = np.sum(self.n)

        # Get width
        self.sigma = self.get_kernel_width(self.X_train) if self.sigma == 'adaptive' else self.sigma



        # Check shapes        
        assert ( len(self.train_idx) == self.X_train.shape[0] and len(self.test_idx) == self.X_test.shape[0] )
        assert ( np.allclose(np.sum(self.gamma), 1.0) and np.allclose(np.sum(self.omega), 1.0) )
        assert ( np.all(self.gamma >= 0.0) and np.all(self.omega) >= 0.0 )

        


        # Construct the matrix belonging to the quadratic problem
        P = self.grad_matrix()

        # Regularize the matrix
        self.regularize(P)

        # Construct the vector belonging to the quadratic problem
        q = self.grad_vector()

        # Define the parameters of the density ratio model
        w = cp.Variable((self.b, 1))



        # Define the quadratic formulation
        if self.use_quad_form:
            P = cp.Parameter(shape = P.shape, value = P, PSD = True)
            quadratic_form = cp.quad_form(w, P)
        else:        
            P_sqrt = cp.Parameter((self.b, self.b))
            quadratic_form = cp.sum_squares(P_sqrt @ w)       
            P_sqrt.value = sqrtm(P).real
            assert ( np.allclose(P_sqrt.value @ P_sqrt.value, P ) ) 



        # Define the objective function
        objective = cp.Minimize( quadratic_form - q.T @ w )



        # Define the vector used in the constraints
        constraint_term = np.array([ np.sum(self.A(j), axis = 0) * self.gamma[j] / self.n_prime[j]
                                    for j in range(self.N_prime) ]) 
        constraint_term = np.sum(constraint_term, axis = 0)[:, None]


        # Define the constraints
        constraint_term = constraint_term.T
        E0 = constraint_term
        E1 = - constraint_term
        E = np.vstack([E0, E1, -np.eye(self.b), np.eye(self.b)])
        f = np.array([[self.eps + 1, self.eps - 1]])
        f = np.vstack([f.T, -np.zeros(shape = (self.b, 1)), np.ones(shape = (self.b, 1)) * self.B])
        constraints = [E @ w <= f]


        # Print DCP and DPP status
        if self.verbose:
            print('<G-KMM> problem is DCP: ', objective.is_dcp())
            print('<G-KMM> problem is DPP: ', objective.is_dcp(dpp = True))


        # Solve the quadratic problem with constraints
        prob = cp.Problem( objective, constraints )
        prob.solve(ignore_dpp = self.ignore_dpp)


        # Save the parameters of the density ratio model
        self.weights = w.value
        self.weights[self.weights <= 0] *= 0
        
        # Check the shapes of the parameters
        assert( self.weights.shape[0] == self.b and self.weights.shape[1] == 1 and self.weights.ndim == 2 )  

        # Flag the finished training process
        self.trained = True





    def predict(self, X):
        """
        Predict the density ratio values with respect to the given dataset.
        """
        if not self.trained:
            raise ValueError('Before predict one must fit the Generalized KMM object !')
        
        X_ = self.reshape_vector(X)
        if self.scale_data:
            X_ = self.scaler.transform(X_)
        
        r = self.xi(X_) @ self.weights
        assert ( r.shape[0] == X_.shape[0] and r.shape[1] == 1 and r.ndim == 2 )
        return r






    def is_positive_semidefinite(self, X):
        pos_semidef = np.all(np.linalg.eigvals( (1 / 2) * (X + X.T) ) >= 0)
        if self.verbose:
            print('Positive semidefinite matrix: ', pos_semidef)
        return pos_semidef
    


    def is_symmetric(self, X):
        symm = np.allclose(X, X.T)
        if self.verbose:
            print('Symmetric matrix: ', symm)
        return symm



    @staticmethod
    def get_kernel_width(data):
        """
        Function which computes the width of the Gaussian kernel.
        See [https://github.com/TongtongFANG/DIW/blob/main/kmm.py]
        """
        dist = []
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                dist.append(np.sqrt(np.sum((np.array(data[i]) - np.array(data[j])) ** 2)))
        return np.quantile(np.array(dist), 0.01)



    @staticmethod
    def reshape_vector(X):
        if X.ndim == 1:
            return X[:, None]
        elif X.ndim == 0:
            return X.reshape(-1, 1)
        else:
            return X
        


    @staticmethod
    def compute_kernel(kernel_type = 'rbf'):
        if kernel_type == 'rbf':
            return rbf_kernel
        elif kernel_type == 'laplacian':
            return laplacian_kernel
        else:
            raise ValueError('Choose a suitable kernel function!')
        

    