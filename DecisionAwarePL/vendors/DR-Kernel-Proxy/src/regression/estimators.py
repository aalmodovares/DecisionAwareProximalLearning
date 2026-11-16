import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from sklearn.base import BaseEstimator, RegressorMixin

import sys
sys.path.append("..")
from utils.kernel_utils import RBF
from utils.linalg_utils import make_psd, cartesian_product

from typing import Callable, Tuple, Optional, Union, Dict

class KernelRidgeRegression(BaseEstimator, RegressorMixin):
    """
    Kernel Ridge Regression model.

    Parameters
    ----------
    alpha : float
        Regularization parameter.
    kernel : Callable or str
        Kernel function or string specifying the kernel type.

    optimize_regularization_parameters: bool, whether to optimize regularization parameters.
    
    lambda_optimization_range: Tuple[float, float], range for lambda optimization.

    - kwargs: Additional arguments:
        - regularization_grid_points: int, number of points for grid used for optimizing the regularizers (default: 150).
        - make_psd_eps: float, epsilon value for making matrices positive semi-definite (default: 1e-7).
        - kernel_params: dict, additional parameters for kernel (default: None).

    Attributes
    ----------
    alpha : float
        Regularization parameter.
    kernel : Callable
        Kernel function.
    kernel_params : dict or None
        Parameters for the kernel function.
    coefs : jnp.ndarray or None
        Coefficients obtained after fitting the model.
    X_fit_ : jnp.ndarray or None
        Input data used for fitting the model.
    """
    def __init__(
        self,
        kernel: Union[Callable, str],
        alpha: float = 1e-3,
        optimize_regularization_parameters: bool = True,
        alpha_optimization_range: Tuple[float, float] = (1e-9, 1.0),
        **kwargs
    ) -> None:
        self.alpha = alpha
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.alpha_optimization_range = alpha_optimization_range

        kernel_params = kwargs.pop('kernel_params', None)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)

        self.kernel_params = kernel_params
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps

        if (not isinstance(kernel, Callable)) and (not isinstance(kernel, str)):
            raise Exception("Kernel must be callable or string")

        if isinstance(kernel, Callable):
            self.kernel = kernel
            if kernel_params is not None:
                self.kernel.set_params(**kernel_params)

        elif isinstance(kernel, str):
            if kernel == "RBF":
                self.kernel = RBF(**kernel_params)
            else:
                raise NotImplementedError("Possible Kernels: RBF")
            self.kernel_params = kernel_params

    @staticmethod
    @jit
    def _alpha_objective(alpha: float, 
                         K_XX: jnp.ndarray, 
                         y_train: jnp.ndarray,
                         make_psd_eps: float = 1e-9) -> float:
        n = K_XX.shape[0]
        identity_matrix = jnp.eye(n)
        H1 = identity_matrix - K_XX @ jnp.linalg.inv(make_psd(K_XX + n * alpha * identity_matrix, eps = make_psd_eps))
        H1_tilde_inv = jnp.diag(1 / jnp.diag(H1))
        objective = (1 / n) * jnp.trace(H1_tilde_inv @ H1 @ (y_train.reshape(-1,1) @ y_train.reshape(-1,1).T) @ H1 @ H1_tilde_inv)
        return objective

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """
        Fit the Kernel Ridge Regression model.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        None
        """
        n = X.shape[0]
        K_XX_plus_alpha_I = self.kernel(X) #+ self.alpha * jnp.eye(X.shape[0])
        if self.optimize_regularization_parameters:
            alpha_optimization_range = self.alpha_optimization_range
            regularization_grid_points = self.regularization_grid_points
            alpha_list = jnp.logspace(jnp.log(alpha_optimization_range[0]), jnp.log(alpha_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            alpha_objective_list = jnp.array([self._alpha_objective(lambda_, K_XX_plus_alpha_I, y, self.make_psd_eps) for lambda_ in alpha_list])
            alpha = alpha_list[jnp.argmin(alpha_objective_list).item()]
            self.alpha = alpha

        diag_indices = jnp.diag_indices_from(K_XX_plus_alpha_I)
        K_XX_plus_alpha_I = K_XX_plus_alpha_I.at[diag_indices].set(K_XX_plus_alpha_I[diag_indices] + n * self.alpha)
        coefs = jnp.linalg.pinv(K_XX_plus_alpha_I) @ y
        self.coefs = coefs
        self.X_fit_ = X
        
    def predict(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Predict target values for the given input data.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray, optional
            Ignored parameter.

        Returns
        -------
        jnp.ndarray
            Predicted target values of shape (n_samples,).
        """
        K_Xx = self.kernel(self.X_fit_, X)
        return K_Xx.T @ self.coefs

    def fit_predict(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the model and predict target values.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        jnp.ndarray
            Predicted target values of shape (n_samples,).
        """
        self.fit(X, y)
        return self.predict(X)


class NystromKernelRidgeRegression(BaseEstimator, RegressorMixin):
    """
    Nystrom Kernel Ridge Regression model.

    Parameters
    ----------
    alpha : float
        Regularization parameter.
    kernel : Callable or str
        Kernel function or string specifying the kernel type.

    - kwargs: Additional arguments:
        - make_psd_eps: float, epsilon value for making matrices positive semi-definite (default: 1e-7).
        - kernel_params: dict, additional parameters for kernel (default: None).

    Attributes
    ----------
    alpha : float
        Regularization parameter.
    kernel : Callable
        Kernel function.
    kernel_params : dict or None
        Parameters for the kernel function.
    coefs : jnp.ndarray or None
        Coefficients obtained after fitting the model.
    X_fit_ : jnp.ndarray or None
        Input data used for fitting the model.
    """
    def __init__(
        self,
        kernel: Union[Callable, str],
        alpha: float = 1e-3,
        m: int = 100,
        seed: int = 0,
        **kwargs
    ) -> None:
        self.alpha = alpha

        kernel_params = kwargs.pop('kernel_params', None)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        self.m = m
        self.seed = seed
        self.kernel_params = kernel_params
        self.make_psd_eps = make_psd_eps

        if (not isinstance(kernel, Callable)) and (not isinstance(kernel, str)):
            raise Exception("Kernel must be callable or string")

        if isinstance(kernel, Callable):
            self.kernel = kernel
            if kernel_params is not None:
                self.kernel.set_params(**kernel_params)

        elif isinstance(kernel, str):
            if kernel == "RBF":
                self.kernel = RBF(**kernel_params)
            else:
                raise NotImplementedError("Possible Kernels: RBF")
            self.kernel_params = kernel_params

    def sample_landmarks(self, X, m, seed = 0):
        np.random.seed(seed)
        indices = np.random.choice(X.shape[0], m, replace=False)
        return X[indices], indices
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> None:
        """
        Fit the Kernel Ridge Regression model.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        None
        """
        n = X.shape[0]
        X_landmarks, landmark_indices = self.sample_landmarks(X, self.m, self.seed)
        train_indices = np.arange(0, X.shape[0])
        K_XX = self.kernel(X) #+ self.alpha * jnp.eye(X.shape[0])
        K_nm = K_XX[tuple(cartesian_product(train_indices, landmark_indices).T)].reshape(X.shape[0], self.m)
        K_mm = K_XX[tuple(cartesian_product(landmark_indices, landmark_indices).T)].reshape(self.m, self.m)

        coefs =  jnp.linalg.solve(K_nm.T @ K_nm + n * self.alpha * K_mm, K_nm.T @ y)
        self.coefs = coefs
        self.X_fit_ = X_landmarks
        
    def predict(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Predict target values for the given input data.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray, optional
            Ignored parameter.

        Returns
        -------
        jnp.ndarray
            Predicted target values of shape (n_samples,).
        """
        K_Xx = self.kernel(self.X_fit_, X)
        return K_Xx.T @ self.coefs

    def fit_predict(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the model and predict target values.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        jnp.ndarray
            Predicted target values of shape (n_samples,).
        """
        self.fit(X, y)
        return self.predict(X)


class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """
    Gaussian Process Regression Model.
    This estimator is written based on the pseudo code given in 
    Algorithm 2.1 of 'Gaussian Process for Machine Learning' by C. Rasmussen abd C. K. I. Williams.
    """
    def __init__(
        self,
        alpha: float,
        kernel: Union[Callable, str],
        kernel_params: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Gaussian Process Regressor.

        Parameters
        ----------
        alpha : float
            Value added to the diagonal of the kernel matrix during fitting.
        kernel : Union[Callable, str]
            Kernel function or string representing a built-in kernel.
        kernel_params : Optional[dict], optional
            Additional parameters (keyword arguments) to pass to the kernel function.
        **kwargs
            Additional keyword arguments:
            normalize_labels : bool, optional
                Whether to normalize the target labels (default is True).
            optimize_kernel_params : bool, optional
                Whether to optimize the kernel parameters (default is True).
            max_iter_optimizer : int, optional
                Maximum number of iterations for kernel parameter optimization (default is 100).
            lr_optimizer : float, optional
                Learning rate for kernel parameter optimization (default is 1e-3).
            kernel_params_optimization_tol : float, optional
                Tolerance for kernel parameter optimization (default is 1e-9).
        """
        super().__init__()

        normalize_labels = kwargs.pop('normalize_labels', True)
        optimize_kernel_params = kwargs.pop('optimize_kernel_params', True)
        max_iter_optimizer = kwargs.pop('max_iter_optimizer', 100)
        lr_optimizer = kwargs.pop('lr_optimizer', 1e-3)
        kernel_params_optimization_tol = kwargs.pop('kernel_params_optimization_tol', 1e-9)

        self.normalize_labels = normalize_labels
        self.optimize_kernel_params = optimize_kernel_params
        self.max_iter_optimizer = max_iter_optimizer
        self.lr_optimizer = lr_optimizer
        self.kernel_params_optimization_tol = kernel_params_optimization_tol

        self.alpha = alpha
        self.kernel_params = kernel_params

        if (not isinstance(kernel, Callable)) and (not isinstance(kernel, str)):
            raise Exception("Kernel must be callable or string")

        if isinstance(kernel, Callable):
            self.kernel = kernel
            if kernel_params is not None:
                self.kernel.set_params(**kernel_params)

        elif isinstance(kernel, str):
            if kernel == "RBF":
                self.kernel = RBF(**kernel_params)
            else:
                raise NotImplementedError("Possible Kernels: RBF")
            self.kernel_params = kernel_params


    def _log_marginal_likelihood(self, kernel_params: Dict, kernel: Callable, X: jnp.ndarray, y: jnp.ndarray, alpha: float) -> float:
        """
        Compute the log marginal likelihood of the Gaussian Process.

        Parameters
        ----------
        kernel_params : dict
            Parameters of the kernel.
        kernel : Callable
            Kernel function.
        X : jnp.ndarray
            Input data.
        y : jnp.ndarray
            Target values.
        alpha : float
            Value added to the diagonal of the kernel matrix.

        Returns
        -------
        float
            Log marginal likelihood.
        """
        kernel.set_params(**kernel_params)
        K_XX_plus_alpha_I = kernel(X) 
        diag_indices = jnp.diag_indices_from(K_XX_plus_alpha_I)
        K_XX_plus_alpha_I = K_XX_plus_alpha_I.at[diag_indices].set(K_XX_plus_alpha_I[diag_indices] + alpha)
        L = jnp.linalg.cholesky(K_XX_plus_alpha_I)
        coefs = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
        log_likelihood = -0.5 * y.T @ coefs - jnp.sum(jnp.log(jnp.diag(L))) - (y.shape[0] / 2) * jnp.log(2 * jnp.pi)
        return log_likelihood
    
    def _optimize_kernel_params(self, X: jnp.ndarray, y: jnp.ndarray, alpha: float) -> Dict:
        """
        Optimize kernel parameters using stochastic gradient descent.

        Parameters
        ----------
        X : jnp.ndarray
            Input data.
        y : jnp.ndarray
            Target values.
        alpha : float
            Value added to the diagonal of the kernel matrix.

        Returns
        -------
        Dict
            Optimized kernel parameters.
        """
        ## Currently, we run SGD optimization. However, we need to be able to support other optimizer types as well.
        ## TODO: Implement other optimizers, like Adam, Nesterov, etc. 
        max_iter_optimizer = self.max_iter_optimizer
        lr_optimizer = self.lr_optimizer
        kernel_params_optimization_tol = self.kernel_params_optimization_tol

        kernel_params = self.kernel.get_params()
        log_likelihood_value_and_grad = jax.value_and_grad(self._log_marginal_likelihood)
        log_likelihood_value_old = -jnp.inf
        log_likelihood_value_list = []

        for j in range(max_iter_optimizer):
            log_likelihood, log_likelihood_grad = log_likelihood_value_and_grad(kernel_params, self.kernel, X, y, alpha)
            log_likelihood = log_likelihood.item()
            log_likelihood_value_list.append(log_likelihood)
            for param_name, param_value in log_likelihood_grad.items():
                kernel_params[param_name] = kernel_params[param_name] + lr_optimizer * param_value.item()
            if abs(log_likelihood - log_likelihood_value_old) < kernel_params_optimization_tol:
                break
            else:
                log_likelihood_value_old = log_likelihood
        
        self.log_likelihood_value_list = log_likelihood_value_list
        return kernel_params
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> 'GaussianProcessRegressor':
        """
        Fit the Gaussian Process Regression model.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray
            Target values of shape (n_samples, 1).

        Returns
        -------
        None
        """

        if self.normalize_labels:
            self.y_mean = jnp.mean(y, axis = 0)
            self.y_std = jnp.std(y, axis = 0)

            y = (y - self.y_mean) / self.y_std

        if self.optimize_kernel_params:
            kernel_params = self._optimize_kernel_params(X, y, self.alpha)
        self.kernel.set_params(**kernel_params)

        K_XX_plus_alpha_I = self.kernel(X) #+ self.alpha * jnp.eye(X.shape[0])
        diag_indices = jnp.diag_indices_from(K_XX_plus_alpha_I)
        K_XX_plus_alpha_I = K_XX_plus_alpha_I.at[diag_indices].set(K_XX_plus_alpha_I[diag_indices] + self.alpha)
        L = jnp.linalg.cholesky(K_XX_plus_alpha_I)
        coefs = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
        self.coefs = coefs
        self.X_fit_ = X
        self.L = L
        log_likelihood = -0.5 * y.T @ self.coefs - jnp.sum(jnp.log(jnp.diag(self.L))) - (y.shape[0] / 2) * jnp.log(2 * jnp.pi)
        self.log_likelihood = log_likelihood
        return self
    
    def predict(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Predict target values for the given input data.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray, optional
            Ignored parameter.

        Returns
        -------
        jnp.ndarray
            Predicted target values of shape (n_samples,).
        """
        K_Xx = self.kernel(self.X_fit_, X)
        y_pred = K_Xx.T @ self.coefs
        ## The following two lines can be performed more computationally efficiently.
        ## TODO: Check the following two lines and make it more efficient.
        v = np.linalg.solve(self.L, K_Xx)
        pred_var = jnp.diag(self.kernel(X, X)) - jnp.sum(v * v, axis = 0)
        pred_std = jnp.sqrt(pred_var)
        
        return self.y_std * y_pred + self.y_mean, pred_std
    
    def fit_predict(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the Gaussian Process Regression model and make predictions.

        Parameters
        ----------
        X : jnp.ndarray
            Input data of shape (n_samples, n_features).
        y : jnp.ndarray, optional
            Target values of shape (n_samples, 1).

        Returns
        -------
        jnp.ndarray
            Predicted target values of shape (n_samples,).
        """
        self.fit(X, y)
        return self.predict(X)