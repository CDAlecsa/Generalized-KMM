# Generalized-KMM
#### Python implementation of the optimization method introduced in the article *"Some notes concerning a generalized KMM-type optimization method for density ratio estimation"*
  * Mathematical framework:
    * Training subsets: $\mathcal{X}^\prime_l = \lbrace x^{\prime}\_{j, (l)} | x^{\prime}\_{j, (l)} \in \mathbb{R}^d \rbrace_{j = 1}^{n^\prime_l}$ for $l \in \lbrace 1, \ldots, N^\prime \rbrace$
    * Test subsets: $\mathcal{X}\_i = \lbrace x_{k,(i)} | x\_{k, (i)} \in \mathbb{R}^d \rbrace_{k = 1}^{n_i}$ for $i \in \lbrace 1, \ldots, N \rbrace$
    * Mixture of probabilities: $p(x) = \sum\limits_{i = 1}^{N} \omega_i p_i(x)$ and $p^\prime(x) = \sum\limits_{j = 1}^{N^\prime} \gamma_j p^\prime_j(x)$
    * Weights restrictions: $\sum\limits_{i = 1}^{N} \omega_i = \sum\limits_{j = 1}^{N^\prime} \gamma_j = 1$, with $\omega_i \in [0, 1]$ for every $i \in \lbrace 1, \ldots, N \rbrace$ and $\gamma_j \in [0, 1]$ for each $j \in \lbrace 1, \ldots, N^\prime \rbrace$
    * Density ratio model: $\hat{r}(x) = \langle \theta, \xi(x) \rangle$, where $\theta = (\theta_1, \ldots, \theta_b) \in \mathbb{R}^b$ and $\xi: \mathbb{R}^d \to \mathbb{R}^b$ such that $\xi(x) = (\xi_1(x), \ldots, \xi_b(x))$ for each $x \in \mathbb{R}^d$.
    * For the full test dataset $\mathcal{X} = \bigcup\limits_{i = 1}^{N} \mathcal{X}_i$ we select $\xi\_k(x) = K(x, x\_k)$ where $x\_k \in \mathcal{X}$ for each $k \in \lbrace 1, \ldots, b \rbrace$, where $b = \sum\limits\_{i = 1}^{N} n\_i$.
  * Various notations:
    * $\hat{r}\_{\mathcal{X}^\prime_j} := \left( \hat{r} \left( x^\prime_{1, (j)} \right), \ldots, \hat{r} \left( x^\prime_{n^\prime_j, (j)} \right) \right)^T \in \mathbb{R}^{n^\prime_j \times 1}, j \in \lbrace 1, \ldots, N^\prime \rbrace$
    * $\hat{r}\_{\mathcal{X}\_i} := \left( \hat{r} \left( x_{1, (i)} \right), \ldots, \hat{r} \left( x_{n_i, (i)} \right) \right)^T \in \mathbb{R}^{n_i \times 1}, i \in \lbrace 1, \ldots, N \rbrace$
    * For $i \in \lbrace 1, \ldots, N \rbrace$ and $j \in \lbrace 1, \ldots, N^\prime \rbrace$ the vector $h^{[i, j]} \in \mathbb{R}^{n^\prime_j \times 1}$ is defined as
$$h_t^{[i, j]} = \dfrac{\left( n^\prime_{max} \right)^2}{n_i n^\prime_j} \gamma_j \omega_i \sum\limits_{l = 1}^{n_i} K \left( x^\prime_{t, (j)}, x_{l, (i)} \right) \text{ for each } t \in \lbrace 1, \ldots, n^\prime_j \rbrace$$
    * For every $j, k \in \lbrace 1, \ldots, N^\prime \rbrace$ the matrix $H^{[j, k]} \in \mathbb{R}^{n^\prime_j \times n^\prime_k}$ is defined as
$$H^{[j, k]}\_{t, s} = \dfrac{\left( n^\prime_{max} \right)^2}{n^\prime_j n^\prime_k} \dfrac{\gamma_j \gamma_k}{2} K \left( x^\prime_{t, (j)}, x^\prime_{s, (k)} \right) \text{ for each } t \in \lbrace 1, \ldots, n^\prime_j \rbrace \text{ and } s \in \lbrace 1, \ldots, n^\prime_k \rbrace$$
    * For $j \in \lbrace 1, \ldots, N^\prime \rbrace$ the matrix $A^{[j]} \in \mathbb{R}^{n^\prime\_j \times b}$ is defined
$$A^{[j]}\_{t, k} = \xi\_k( x^{\prime}\_{t, (j)} ) \text{ for each } t \in \lbrace 1, \ldots, n^\prime_j \rbrace \text{ and } k \in \lbrace 1, \ldots, b \rbrace$$
    * $\Xi = (\Xi_1, \ldots, \Xi_b) \in \mathbb{R}^b$, where $\Xi := \sum\limits_{j = 1}^{N^\prime} \left( \dfrac{\gamma_j}{n^\prime_j} \right) \sum\limits_{k = 1}^{n^\prime_j} \xi(x_{k, (j)})$
  * Optimization problem:
    * Empirical loss function: $\widehat{\mathcal{L}} = \left[ \theta^T \left( \sum\limits_{j = 1}^{N^\prime}\sum\limits_{k = 1}^{N^\prime} (A^{[j]})^T H^{[j, k]} A^{[k]} \right) \theta - \left( \sum\limits_{j = 1}^{N^\prime}\sum\limits_{i = 1}^{N} (h^{[i, j]})^T A^{[j]} \right) \theta \right]$
    * Constraints: $\theta_k \in [0, B] \text{ for } k \in \lbrace 1, \ldots, b \rbrace$, $\sum\limits_{k = 1}^{b} \theta_k \Xi_k \leq \varepsilon + 1$ and $- \sum\limits_{k = 1}^{b} \theta_k \Xi_k \leq \varepsilon - 1$, respectively.    


# Notes about our KMM & Generalized KMM implementations
* Both implementations of the classical KMM method from [1] and [2] uses the *CVXOPT* package in order to solve the underlying quadratic optimization problem with constraints. But, we have opted for the usage of the *CVXPY* package. The matrix appearing in the KMM's quadratic problem, which we shall denote it with $K$, is in fact the Gaussian kernel applied to points belonging to the training dataset. In [1] the kernel $K$ is modified with the addition of a regularization term in order to prevent nonsingularity (see subsection 3.1.2 entitled *Geometry of least squares* from page 143 of [7]), namely $K$ takes the value $K + \lambda \mathcal{I}$. On the other hand, in the code from the ADAPT package [2] the authors used only the transformation from $K$ to $\dfrac{1}{2} \left( K + K^T \right)$ which induces symmetry on $K$. We have introduced in our codes both the aforementioned transformations by checking if the matrix appearing in our quadratic problem is symmetric and positive semidefinite.
* In order to solve the KMM and Generalized KMM quadratic problems in *CVXPY* one can utilize the following techniques:
  * The basic *quad_form* approach presented in [3]. We have observed that this approach fails due to the issue shown in [6]. This problem is in connection with the fact that the PSD check failed (with respect to the DCP rules), and even if we try to impose a greater value on the parameter $\lambda$ one can't get rid of the convergence problem. The solution that we have found is related to [5] where one can wrapp $K$ with *cp.Parameter(shape = K.shape, value = K, PSD = True)*. We also highlight that, when using this approach, even though the quadratic problem is DCP, the *CVXPY* raises the *warning* that the problem is not DPP: *"UserWarning: You are solving a parameterized problem that is not DPP"*, which can be easily solved by setting *ignore_dpp = True*.
  * A different alternative is to use the technique given in the section *The DPP ruleset* from [4] where we can represent a quadratic problem as *quad_form = cp.sum_squares(P_sqrt @ x)* using *scipy.linalg.sqrtm(P)* (where $P$ is equal to $K$ in the case of KMM), namely *P_sqrt.value = scipy.linalg.sqrtm(P)* (in our codes we have used *scipy.linalg.sqrtm(P).real* in order to retrieve the real value, since the *sqrtm* method from *scipy.linalg* returns a complex value). We have observed that in some cases the *CVXPY* can't allocate enough memory therefore we propose using *ignore_dpp = True* (this mostly happens with KMM and not Generalized KMM, since in KMM the length of the matrix depends on the training samples, while in the case of the Generalized KMM it depends on the test dataset which is smaller). Furthemore, for the Generalized KMM case involving the second approach with the setting *ignored_dpp = False*, *CVXPY* raises *" UserWarning: Your problem has too many parameters for efficient DPP compilation. We suggest setting 'ignore_dpp = True'."*, which can be solved as mentioned in warning by considering the option *ignore_dpp = True*. 
* Takeaways:
  * In our implementations we let the user easily choose between the previously described techniques.
  * Despite the fact that the *PSD* case is handled by the aforementioned approaches, as a safety measure, we also consider the addition of $\lambda \mathcal{I}$ to $K$ for a given value of $\lambda$.



# Remarks:
* Hyper-parameters: $\varepsilon$, $B$.
* Our parameter $\sigma$ is denoted as $\gamma$ in the *sklearn* implementation from [8] (and when $\gamma \approx \tilde{\sigma}^{-2}$ then the aforementioned parameter is associated with the variance of the kernel $K$). Furthermore, $\sigma$ can be treated in different manners:
  * it can be given as a hyper-parameter regardless of which kernel we are using
  * the variance of each kernel $K$ which appears in the computations is calculated with respect to the whole training data using the function *get_kernel_width* from [1] (as in the classical KMM approach). The option can be activated by setting 'sigma = adaptive' in the initialization of *Generalized_KMM*.
  * the default value 'sigma = None' for *Generalized_KMM* objects sets the variance of each kernel $K$ which appears in the computations to the default value of the underlying *sklearn* kernel.
* In the present implementation only the *rbf_kernel* and the *laplacian_kernel* (both belonging to *sklearn.metrics.pairwise* - see [8]) are utilized.
* The *Generalized_KMM* has a scaling option (utilizing *Normalizer* from *sklearn*) which can be used by setting *scale_data = True*. There are some situations when the training data and the test data have totally different scales. The main issue is that the values of the *rbf_kernel* and *laplacian_kernel* become extremely small since they use the exponential mapping applied to the distance between samples. This leads to a *CVXPY* non-convergence problem and in order to alleviate this one can scale the data by normalizing each individual sample to unit norm. It is worth emphasizing that since the weights are computed with respect to the scaled data they don't represent exactly the true estimated weights of the unscaled data. So, when one applies the underlying scaling they need to take this into account and compare through visualization the estimated weights obtained from both scaled and unscaled data (only when there aren't any problems with the convergence in the case of the unscaled data).



# References
1. KMM implementation from DIW: https://github.com/TongtongFANG/DIW/blob/main/kmm.py
2. KMM implementation from the ADAPT package: https://github.com/antoinedemathelin/adapt/blob/master/adapt/instance_based/_kmm.py
3. Basic quadratic programming in CVXPY: https://www.cvxpy.org/examples/basic/quadratic_program.html
4. DPP quadratic programming in CVXPY: https://www.cvxpy.org/tutorial/advanced/index.html
5. CVXPY issue regarding semidefinite positive matrix: https://github.com/cvxpy/cvxpy/issues/407
6. Arpack convergence error: https://stackoverflow.com/questions/63117123/cvxpy-quadratic-programming-arpacknoconvergence-error
7. C.M. Bishop - *Pattern recognition and Machine Learning*: http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf
8. Kernels defined in sklearn: https://scikit-learn.org/stable/modules/metrics.html
