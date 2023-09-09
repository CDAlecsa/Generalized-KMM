# Load modules
import math
import numpy as np
import cvxpy as cp

from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel





class KMM():
    """
    Kernel mean matching (KMM)

    
    Parameters
    ----------
    sigma: float (default = None)
        The parameter of the Gaussian kernel.
    
    lambd: float (default = 1e-5)
        Regularization term for the kernel matrix involved in the quadratic problem.
    
    eps: float, optional (default = None)
        Constraint parameter. If 'None', 'eps' is set to 'eps = (math.sqrt(n_tr) - 1) / math.sqrt(n_tr)'.

    B: float (default = 1000)
        Bounding value for the density ratio weights.

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
 
 
    """


    def __init__(self, 
                 sigma = None, lambd = 1e-5, eps = None, B = 1000,
                 use_quad_form = True, ignore_dpp = True,
                 verbose = False):
        
        self.sigma = sigma
        self.lambd = lambd
        self.eps = eps
        self.B = B

        self.use_quad_form = use_quad_form
        self.ignore_dpp = ignore_dpp

        self.verbose = verbose


    

    def fit_predict(self, X_train, X_test):
        """
        Estimates the density ratio and returns its value at the training samples.        
        """
        
        # Define the kernel width
        if self.sigma is None:
            self.sigma = self.get_kernel_width(X_train)

        # Get lengths of datasets
        n_train = len(X_train)
        n_test = len(X_test)

        # Compute the Gaussian kernel
        K = rbf_kernel(X_train, X_train, self.sigma)

        # Apply regularization
        K += self.lambd * np.identity(n_train)

        # Compute vector belonging to the quadratic problem
        q = rbf_kernel(X_train, X_test, self.sigma)
        ones = np.ones(shape = (n_test, 1))
        q = np.dot(q, ones)
        q = ( float(n_train) / float(n_test) ) * q

        # Compute the constraint parameter
        if self.eps is None:
            self.eps = (math.sqrt(n_train) - 1) / math.sqrt(n_train)


        # Define the constraints matrices
        A0 = np.ones(shape = (1, n_train))
        A1 = -np.ones(shape = (1, n_train))
        A = np.vstack([A0, A1, -np.eye(n_train), np.eye(n_train)])

        b = np.array([[n_train * (self.eps + 1), n_train * (self.eps - 1)]])
        b = np.vstack([b.T, -np.zeros(shape = (n_train, 1)), np.ones(shape = (n_train, 1)) * self.B])


        # Define the density ratio weights
        w = cp.Variable((n_train, 1))


        # Define the quadratic formulation
        if self.use_quad_form:
            P = cp.Parameter(shape = K.shape, value = K, PSD = True)
            quadratic_form = cp.quad_form(w, P)
        else:        
            P_sqrt = cp.Parameter((n_train, n_train))
            quadratic_form = cp.sum_squares(P_sqrt @ w)       
            P_sqrt.value = sqrtm(K).real
            assert ( np.allclose(P_sqrt.value @ P_sqrt.value, K ) )       
        

        # Define the objective function
        objective = cp.Minimize( (1 / 2) * quadratic_form - q.T @ w )
        
        # Define the constraints
        constraints = [A @ w <= b]

        # Print DCP and DPP status
        if self.verbose:
            print('<KMM> problem is DCP: ', objective.is_dcp())
            print('<KMM> problem is DPP: ', objective.is_dcp(dpp = True))

        # Solve the quadratic problem with constraints
        prob = cp.Problem( objective, constraints )
        prob.solve(ignore_dpp = self.ignore_dpp)

        # Return the weights
        weights = w.value
        weights[weights <= 0] *= 0
        return weights





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


