import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.optimize import minimize as jax_minimize
from jaxopt import OSQP
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

import sys
sys.path.append("..")
from utils.kernel_utils import Kernel, RBF
from utils.linalg_utils import make_psd, cartesian_product, remove_diagonal_elements, columns_mean_excluding_self

from typing import Callable, Tuple, List, Optional, Union, Dict, Any

from jax import config
config.update("jax_enable_x64", True)


class KernelAlternativeProxyATE(BaseEstimator, RegressorMixin):

    def __init__(self,
                 kernel_A: Kernel,
                 kernel_W: Kernel,
                 kernel_Z: Kernel,
                 kernel_X: Kernel = RBF(),
                 lambda_: float = 0.1,
                 eta: float = 0.1, 
                 lambda2_: float = 0.1,
                 optimize_lambda_parameters: bool = True,
                 optimize_eta_parameter: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 eta_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 **kwargs) -> None:
        """
        Initialize the KernelAlternativeProxyATE estimator.

        Parameters:
        - kernel_A (Kernel): Kernel function for variable A.
        - kernel_W (Kernel): Kernel function for variable W.
        - kernel_Z (Kernel): Kernel function for variable Z.
        - kernel_X (Kernel, optional): Kernel function for variable X. Defaults to RBF().
        - lambda_ (float, optional): Regularization parameter. Defaults to 0.1.
        - eta (float, optional): Regularization parameter for structural function prediction. Defaults to 0.1.
        - lambda2_ (float, optional): Second stage regularization parameter. Defaults to 0.1.
        - optimize_lambda_parameters (bool, optional): Flag to optimize lambda regularization parameters. Defaults to True.
        - optimize_eta_parameters (bool, optional): Flag to optimize eta regularization parameter. Defaults to True.
        - lambda_optimization_range (Tuple[float, float], optional): Range for lambda optimization. Defaults to (1e-7, 1.0).
        - eta_optimization_range (Tuple[float, float], optional): Range for eta optimization. Defaults to (1e-7, 1.0).
        - **kwargs: Additional parameters.
        """
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        label_variance_in_lambda_opt = kwargs.pop('label_variance_in_lambda_opt', 0.0)
        label_variance_in_eta_opt = kwargs.pop('label_variance_in_eta_opt', 0.0)
        large_lambda_eta_option = kwargs.pop('large_lambda_eta_option', False)
        selecting_biggest_lambda_tol = kwargs.pop('selecting_biggest_lambda_tol', 1e-9)
        selecting_biggest_eta_tol = kwargs.pop('selecting_biggest_eta_tol', 1e-9)
        selecting_biggest_lambda2_tol = kwargs.pop('selecting_biggest_zeta_tol', 1e-9)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Kernel)):
            raise Exception("Kernel for A must be callable Kernel class")
        if (not isinstance(kernel_W, Kernel)):
            raise Exception("Kernel for W must be callable Kernel class")
        if (not isinstance(kernel_Z, Kernel)):
            raise Exception("Kernel for Z must be callable Kernel class")
        if (not isinstance(kernel_X, Kernel)):
            raise Exception("Kernel for X must be callable Kernel class")
        
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda_, self.eta, self.lambda2_ = lambda_, eta, lambda2_
        self.optimize_lambda_parameters = optimize_lambda_parameters
        self.optimize_eta_parameter = optimize_eta_parameter
        self.lambda_optimization_range = lambda_optimization_range
        self.eta_optimization_range = eta_optimization_range
        self.large_lambda_eta_option = large_lambda_eta_option
        self.selecting_biggest_lambda_tol = selecting_biggest_lambda_tol
        self.selecting_biggest_eta_tol = selecting_biggest_eta_tol
        self.selecting_biggest_lambda2_tol = selecting_biggest_lambda2_tol
        self.label_variance_in_lambda_opt = label_variance_in_lambda_opt
        self.label_variance_in_eta_opt = label_variance_in_eta_opt
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps        

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################        
    @staticmethod
    @jit
    def _lambda_objective(lambda_: float, 
                          K_AWX: jnp.ndarray, 
                          K_ZZ: jnp.ndarray, 
                          identity_matrix: jnp.ndarray,
                          label_variance_in_lambda_opt: float, 
                          make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for lambda optimization.

        Parameters:
        - lambda_ (float): Regularization parameter.
        - K_AWX (jnp.ndarray): Kernel matrix for variable A, W and X.
        - K_ZZ (jnp.ndarray): Kernel matrix for variable Z.
        - identity_matrix (jnp.ndarray): Identity matrix of size n (data size of stage 1 data).
        - make_psd_eps (float, optional): Epsilon value for making matrices positive semi-definite. Defaults to 1e-9.

        Returns:
        - float: Objective value.
        """
        n = K_AWX.shape[0]
        ridge_weights = make_psd(K_AWX + n * lambda_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_AWX).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_ZZ @ H1_tilde_inv_times_H1) 
        objective += label_variance_in_lambda_opt * jnp.trace(R)
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.sum((H1_diag - 1) / H1_diag)
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.trace(R @ H1_tilde_inv @ R.T)
        return objective
    
    @staticmethod
    @jit 
    def _eta_objective(eta, L, L_sub, M, N, L2, M2, stage1_data_size, label_variance_in_eta_opt, make_psd_eps = 1e-9):
        stage2_data_size = L.shape[0] - 1
        alpha = jnp.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)
        cost = ((1 / stage1_data_size) * (alpha.T @ make_psd(L2, make_psd_eps) @ alpha) - 2 * (alpha.T @ M2)) 
        cost += label_variance_in_eta_opt * (2 / stage2_data_size) * jnp.trace(jnp.linalg.solve(make_psd(L + stage2_data_size * eta * N, make_psd_eps), L))
        return cost.reshape(())
    
    @staticmethod
    @jit 
    def _lambda2_objective(lambda2_: float,
                           K_AA: jnp.ndarray,
                           K_ZZ: jnp.ndarray,
                           Y: jnp.ndarray,
                           label_variance_in_lambda_opt: float, 
                           make_psd_eps: float = 1e-9) -> float:
        """
        Computes the objective function for optimization with respect to lambda2_.

        Parameters:
        - lambda2_ (float): Parameter for optimization.
        - K_AA (jnp.ndarray): Kernel matrix for data points in set A.
        - K_ZZ (jnp.ndarray): Kernel matrix for data points in set Z.
        - Y (jnp.ndarray): Target values matrix.
        - make_psd_eps (float, optional): Epsilon value for making a matrix positive semidefinite. Defaults to 1e-9.

        Returns:
        - float: Objective value.
        """
        n = K_AA.shape[0]
        identity_matrix = jnp.eye(n)
        ridge_weights = make_psd(K_AA + n * lambda2_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_AA).T
        H2 = identity_matrix - R
        H2_diag = jnp.diag(H2)
        H2_tilde_inv = jnp.diag(1 / H2_diag)
        kernel_output = K_ZZ * (Y @ Y.T)
        H2_tilde_inv_times_H2 = H2_tilde_inv @ H2
        objective = (1 / n) * jnp.trace(H2_tilde_inv_times_H2 @ kernel_output @ H2_tilde_inv_times_H2.T)
        objective += label_variance_in_lambda_opt * jnp.trace(R) 
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.sum((H2_diag - 1) / H2_diag)
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.trace(R @ H2_tilde_inv @ R.T)
        return objective
    
    @staticmethod
    @jit 
    def _predict_structural_function(alpha: jnp.ndarray,
                                     B: jnp.ndarray,
                                     B_bar: jnp.ndarray,
                                     third_stage_KRR_weights: jnp.ndarray,
                                     K_ATraina: jnp.ndarray,
                                     K_ATildea: jnp.ndarray,
                                     ones_divided_by_m: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the structural function.

        Parameters:
        - alpha (jnp.ndarray): Coefficient array.
        - B (jnp.ndarray): Matrix B from second stage.
        - B_bar (jnp.ndarray): Matrix B_bar from second stage.
        - third_stage_KRR_weights (jnp.ndarray): Weights from third stage kernel ridge regression.
        - K_ATraina (jnp.ndarray): Kernel matrix between training set A and a test point.
        - K_ATildea (jnp.ndarray): Kernel matrix between stage 2 set A and a test point.
        - ones_divided_by_m (jnp.ndarray): Array of ones divided by stage 2 data size.

        Returns:
        - jnp.ndarray: Predicted values.
        """
        pred = (alpha[:-1].T @ ((B.T @ (third_stage_KRR_weights @ K_ATraina)) * K_ATildea))
        pred += (alpha[-1] * ((B_bar.T @ (third_stage_KRR_weights @ K_ATraina)) * K_ATildea) @ ones_divided_by_m)
        return pred

    ########################################################################
    ############## Second Stage Regression Loss for Validation #############
    ########################################################################  
    def _2nd_stage_regression_loss(self, A_test, W_test, X_test = None):
        alpha = self.alpha
        K_ZZ = self.K_ZZ
        val_size = A_test.shape[0]
        val_idx = jnp.arange(val_size)
        stage1_idx, stage2_idx = self.stage1_idx, self.stage2_idx
        stage1_data_size, stage2_data_size = stage1_idx.shape[0], stage2_idx.shape[0]
        train_size = stage1_data_size + stage2_data_size
        B, B_bar = self.B, self.B_bar
        make_psd_eps = self.make_psd_eps

        K_ATrainATest = self.kernel_A(self.ATrain, A_test)
        K_WTrainWTest = self.kernel_W(self.WTrain, W_test)
        if self.XTrain is None:
            K_XtrainXTest = jnp.ones((train_size, val_size))
        else:
            K_XtrainXTest = self.kernel_X(self.XTrain, X_test)

        K_AATest = K_ATrainATest[tuple(cartesian_product(stage1_idx, val_idx).T)].reshape(stage1_data_size, val_size)
        K_WWTest = K_WTrainWTest[tuple(cartesian_product(stage1_idx, val_idx).T)].reshape(stage1_data_size, val_size)
        K_XXTest = K_XtrainXTest[tuple(cartesian_product(stage1_idx, val_idx).T)].reshape(stage1_data_size, val_size)
        K_ATildeAtest = K_ATrainATest[tuple(cartesian_product(stage2_idx, val_idx).T)].reshape(stage2_data_size, val_size)

        B2 = jnp.linalg.solve(make_psd(self.stage1_ridge_weights, make_psd_eps), (K_WWTest * K_XXTest * K_AATest))
        B2_bar = jnp.linalg.solve(make_psd(self.stage1_ridge_weights, make_psd_eps),  (columns_mean_excluding_self(K_WWTest * K_XXTest) * K_AATest))
        ones_divided_by_val_size = jnp.ones((val_size)) / val_size   
        ones_divided_by_m = jnp.ones((stage2_data_size)) / stage2_data_size

        block_component12 = (B2.T @ K_ZZ @ B) * K_ATildeAtest.T
        block_component22 = (B2.T @ K_ZZ @ B_bar) * K_ATildeAtest.T
        block_component32 = (B.T @ K_ZZ @ B2_bar) * K_ATildeAtest
        block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_ATildeAtest

        L2_sub = jnp.vstack((block_component12.T, ones_divided_by_m.T @ block_component22.T))
        L2 = L2_sub @ L2_sub.T
        M2 = jnp.vstack(((block_component32 @ ones_divided_by_val_size).reshape(-1, 1), (ones_divided_by_m.T @ block_component42 @ ones_divided_by_val_size).reshape(-1, 1)))
        cost = ((1 / val_size) * (alpha.T @ make_psd(L2, make_psd_eps) @ alpha) - 2 * (alpha.T @ M2))
        return cost
    
    ########################################################################
    ###################### FIT AND PREDICT FUNCTIONS #######################
    ########################################################################
    def fit(self, 
            AWZX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray,) -> None:
        """
        Fit the KernelAlternativeProxyATE model.

        Parameters:
        - AWZX (Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]): Tuple of data arrays (A, W, Z, X).
        - Y (np.ndarray): Target values.
        """
        kernel_A, kernel_W, kernel_Z, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X
        lambda_, eta, lambda2_ = self.lambda_, self.eta, self.lambda2_
        optimize_lambda_parameters = self.optimize_lambda_parameters
        optimize_eta_parameter = self.optimize_eta_parameter
        lambda_optimization_range = self.lambda_optimization_range
        eta_optimization_range = self.eta_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps
        large_lambda_eta_option = self.large_lambda_eta_option
        selecting_biggest_lambda_tol = self.selecting_biggest_lambda_tol
        selecting_biggest_eta_tol = self.selecting_biggest_eta_tol
        selecting_biggest_lambda2_tol = self.selecting_biggest_lambda2_tol
        label_variance_in_lambda_opt = self.label_variance_in_lambda_opt
        label_variance_in_eta_opt = self.label_variance_in_eta_opt
        
        if len(AWZX) == 4:
            A, W, Z, X = AWZX
        elif len(AWZX) == 3:
            A, W, Z = AWZX
            X = None
        
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_ZTrainZTrain = kernel_Z(Z, Z)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = make_psd(kernel_X(X, X), make_psd_eps)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        ############################# SPLIT DATA IN STAGE 1 AND STAGE 2 #####################################
        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        ################################ KERNEL MATRICES ####################################################
        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, 'use_length_scale_heuristic'):
                kernel_.use_length_scale_heuristic = False

        ########## OPTIMIZE THE LAMBDA REGULARIZATION PARAMETER IF IT IS SPECIFIED ###########################
        I_n = jnp.eye(stage1_data_size)
        K_AWX = K_AA * K_WW * K_XX
        if optimize_lambda_parameters:
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            self.lambda_list = lambda_list
            lambda_objective_list = jnp.array([self._lambda_objective(lambda_, K_AWX, K_ZZ, 
                                                                      I_n, label_variance_in_lambda_opt, make_psd_eps) for lambda_ in lambda_list])
            self.lambda_objective_list = lambda_objective_list
            if large_lambda_eta_option:
                lambda_objective_min = jnp.min(lambda_objective_list).item()
                _lambda_objective_list_minimizer_list = jnp.where(jnp.abs(lambda_objective_list - lambda_objective_min) <= selecting_biggest_lambda_tol)[0]
                lambda_ = jnp.max(lambda_list[_lambda_objective_list_minimizer_list])
            else:
                lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]

        ########### FIRST AND SECOND STAGE REGRESSION ########################################
        stage1_ridge_weights = (K_AWX + stage1_data_size * lambda_ * I_n)
        self.stage1_ridge_weights = stage1_ridge_weights
        B = jnp.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps), (K_WWTilde * K_XXTilde * K_AATilde))
        B_bar = jnp.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps),  (columns_mean_excluding_self(K_WWTilde * K_XXTilde) * K_AATilde))

        block_component1 = (B.T @ K_ZZ @ B) * K_ATildeATilde
        block_component2 = (B.T @ K_ZZ @ B_bar) * K_ATildeATilde
        block_component4 = (B_bar.T @ K_ZZ @ B_bar) * K_ATildeATilde
        ones_divided_by_m = jnp.ones((stage2_data_size)) / stage2_data_size

        L_sub = jnp.vstack((block_component1, ones_divided_by_m.T @ block_component2.T))
        L = L_sub @ L_sub.T
        self.L = L_sub.T
        M = jnp.vstack(((block_component2 @ ones_divided_by_m).reshape(-1, 1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))
        
        P = jnp.hstack((block_component1, (block_component2 @ ones_divided_by_m).reshape(-1, 1)))
        R = jnp.hstack(((ones_divided_by_m.T @ block_component2.T).reshape(1, -1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))
        N = jnp.vstack((P, R))

        if optimize_eta_parameter:
            B2 = jnp.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps), K_AWX)
            B2_bar = jnp.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps),  (columns_mean_excluding_self(K_WW * K_XX)* K_AA))
            ones_divided_by_n = jnp.ones((stage1_data_size)) / stage1_data_size    

            block_component12 = (B2.T @ K_ZZ @ B) * K_AATilde
            block_component22 = (B2.T @ K_ZZ @ B_bar) * K_AATilde
            block_component32 = (B.T @ K_ZZ @ B2_bar) * K_ATildeA
            block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_ATildeA

            L2_sub = jnp.vstack((block_component12.T, ones_divided_by_m.T @ block_component22.T))
            L2 = L2_sub @ L2_sub.T
            M2 = jnp.vstack(((block_component32 @ ones_divided_by_n).reshape(-1, 1), (ones_divided_by_m.T @ block_component42 @ ones_divided_by_n).reshape(-1, 1)))

            eta_list = np.logspace(np.log(eta_optimization_range[0]), np.log(eta_optimization_range[1]), regularization_grid_points, base = np.exp(1))
            self.eta_list = eta_list
            eta_objective_list = jnp.array([self._eta_objective(eta_, L, L_sub, M, N, L2, M2, stage1_data_size, 
                                                                label_variance_in_eta_opt, make_psd_eps) for eta_ in eta_list]).reshape(-1, 1)

            if large_lambda_eta_option:
                eta_objective_min = jnp.min(eta_objective_list).item()
                _eta_objective_list_minimizer_list = jnp.where(jnp.abs(eta_objective_list - eta_objective_min) <= selecting_biggest_eta_tol)[0]
                eta = jnp.max(eta_list[_eta_objective_list_minimizer_list])
            else:
                eta = eta_list[jnp.argmin(eta_objective_list).item()]
            self.eta_objective_list = eta_objective_list
            self.final_second_stage_validation_loss = self._eta_objective(eta, L, L_sub, M, N, L2, M2, stage1_data_size, 
                                                                          0, make_psd_eps)
        alpha = jnp.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)
        ########### THIRD STAGE ########################################
        if optimize_lambda_parameters:
            lambda2_list = lambda_list.copy()
            lambda2_objective_list = jnp.array([self._lambda2_objective(lambda2_, K_ATrainATrain, K_ZTrainZTrain, Y,
                                                                        label_variance_in_lambda_opt, make_psd_eps) for lambda2_ in lambda_list])
            self.lambda2_objective_list = lambda2_objective_list
            if large_lambda_eta_option:
                lambda2_objective_min = jnp.min(lambda2_objective_list).item()
                _lambda2_objective_list_minimizer_list = jnp.where(jnp.abs(lambda2_objective_list - lambda2_objective_min) <= selecting_biggest_lambda2_tol)[0]
                lambda2_ = jnp.max(lambda2_list[_lambda2_objective_list_minimizer_list])
            else:
                lambda2_ = lambda2_list[jnp.argmin(lambda2_objective_list).item()]

        K_ZZTrain = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, train_indices).T)].reshape(stage1_data_size, train_data_size)
        K_ATrainATrain_ = K_ATrainATrain[tuple(cartesian_product(train_indices, train_indices).T)].reshape(train_data_size, train_data_size)
        third_stage_KRR_weights = jnp.linalg.solve(make_psd(K_ATrainATrain_ + train_data_size * lambda2_ * jnp.eye(train_data_size), make_psd_eps), (K_ZZTrain.T * Y[train_indices])).T 

        self.alpha = alpha
        self.B, self.B_bar = B, B_bar
        self.third_stage_KRR_weights = third_stage_KRR_weights
        self.ones_divided_by_m = ones_divided_by_m
        self.ATrain, self.WTrain, self.XTrain, self.ZTrain = A, W, X, Z
        self.K_ZZ = K_ZZ
        self.train_indices = train_indices
        self.stage1_idx, self.stage2_idx = stage1_idx, stage2_idx

        ##### For debugging purpose, I might want to check the regularization values after optimization #######
        self.lambda_ = lambda_
        self.lambda2_ = lambda2_
        self.eta = eta

    def predict(self, A: jnp.ndarray, verbose: str = False) -> jnp.ndarray:
        """
        Predict outcomes for new data points.

        Parameters:
        - A (jnp.ndarray): New data points for variable A.

        Returns:
        - jnp.ndarray: Predicted values.
        """
        if A.ndim != 2:
            A_test = A.reshape(-1, 1)
        else:
            A_test = A
        K_ATrainATest = self.kernel_A(self.ATrain, A_test)

        test_indices = jnp.arange(A_test.shape[0])
        test_shape = test_indices.shape[0]

        K_ATrainATest_ = K_ATrainATest[tuple(cartesian_product(self.train_indices, test_indices).T)].reshape(self.train_indices.shape[0], test_shape)
        K_ATildeATest = K_ATrainATest[tuple(cartesian_product(self.stage2_idx, test_indices).T)].reshape(self.stage2_idx.shape[0], test_shape)

        ones_divided_by_m = self.ones_divided_by_m
        alpha = self.alpha
        B, B_bar = self.B, self.B_bar
        third_stage_KRR_weights = self.third_stage_KRR_weights

        if verbose:
            f_struct_pred = jnp.array([self._predict_structural_function(alpha, B, B_bar, third_stage_KRR_weights, 
                                                                        K_ATrainATest_[:, jj], K_ATildeATest[:, jj], 
                                                                        ones_divided_by_m).item() for jj in tqdm(range(K_ATildeATest.shape[1]))])
        else:
            f_struct_pred = jnp.array([self._predict_structural_function(alpha, B, B_bar, third_stage_KRR_weights, 
                                                                        K_ATrainATest_[:, jj], K_ATildeATest[:, jj], 
                                                                        ones_divided_by_m).item() for jj in range(K_ATildeATest.shape[1])])
        return f_struct_pred

    def _predict_bridge_func(self, A_test : jnp.ndarray, Z_test : jnp.ndarray, X_test = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ZZTest = self.kernel_Z(self.ZTrain[self.stage1_idx, :], Z_test)
        K_ATildeATest = self.kernel_A(self.ATrain[self.stage2_idx, :], A_test)
        alpha, B, B_bar = self.alpha, self.B, self.B_bar
        ones_divided_by_m = self.ones_divided_by_m
        bridge_function = jnp.array([alpha[:-1].T @ ((B.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1, 1)) + alpha[-1] * ones_divided_by_m.T @ ((B_bar.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1,1)) for jj in range(K_ATildeATest.shape[1])])
        return bridge_function[:, 0, :]
    

class KernelAlternativeProxyATT(BaseEstimator, RegressorMixin):

    def __init__(self,
                 kernel_A: Kernel,
                 kernel_W: Kernel,
                 kernel_Z: Kernel,
                 kernel_X: Kernel = RBF(),
                 lambda_: float = 0.1,
                 eta: float = 0.1, 
                 lambda2_: float = 0.1,
                 zeta: float = 0.1,
                 optimize_lambda_parameters: bool = True,
                 optimize_eta_parameter: bool = True,
                 optimize_zeta_parameter: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 eta_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 **kwargs) -> None:
        """
        Initialize the KernelAlternativeProxyATT class.

        Parameters:
        - kernel_A: Callable, kernel function for variable A.
        - kernel_W: Callable, kernel function for variable W.
        - kernel_Z: Callable, kernel function for variable Z.
        - kernel_X: Callable, kernel function for variable X (default is RBF).
        - lambda_: float, regularization parameter for stage 1 (default is 0.1).
        - eta: float, regularization parameter for stage 2 (default is 0.1).
        - lambda2_: float, regularization parameter for stage 3 (default is 0.1).
        - zeta: float, regularization parameter for theta (default is 0.1).
        - optimize_lambda_parameters (bool, optional): Flag to optimize lambda regularization parameters. Defaults to True.
        - optimize_eta_parameters (bool, optional): Flag to optimize eta regularization parameter. Defaults to True.
        - lambda_optimization_range: Tuple[float, float], range for lambda optimization (default is (1e-7, 1.0)).
        - eta_optimization_range: Tuple[float, float], range for lambda optimization (default is (1e-7, 1.0)).
        - zeta_optimization_range: Tuple[float, float], range for zeta optimization (default is (1e-7, 1.0)).
        - **kwargs: additional keyword arguments.
        """
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        label_variance_in_lambda_opt = kwargs.pop('label_variance_in_lambda_opt', 0.0)
        label_variance_in_eta_opt = kwargs.pop('label_variance_in_eta_opt', 0.0)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)
        lambda2_optimization_range = kwargs.pop('lambda2_optimization_range', None)

        if (not isinstance(kernel_A, Kernel)):
            raise Exception("Kernel for A must be callable Kernel class.")
        if (not isinstance(kernel_W, Kernel)):
            raise Exception("Kernel for W must be callable Kernel class.")
        if (not isinstance(kernel_Z, Kernel)):
            raise Exception("Kernel for Z must be callable Kernel class.")
        if (not isinstance(kernel_X, Kernel)):
            raise Exception("Kernel for X must be callable Kernel class.")
        
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda_, self.eta, self.lambda2_, self.zeta = lambda_, eta, lambda2_, zeta
        self.optimize_lambda_parameters = optimize_lambda_parameters
        self.optimize_eta_parameter = optimize_eta_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda_optimization_range = lambda_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range
        self.eta_optimization_range = eta_optimization_range
        self.zeta_optimization_range = zeta_optimization_range
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.label_variance_in_lambda_opt = label_variance_in_lambda_opt
        self.label_variance_in_eta_opt = label_variance_in_eta_opt
        self.make_psd_eps = make_psd_eps        

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################

    @staticmethod
    @jit 
    def _lambda_objective(lambda_: float, 
                          K_AWX: jnp.ndarray, 
                          K_ZZ: jnp.ndarray, 
                          identity_matrix: jnp.ndarray,
                          label_variance_in_lambda_opt: float, 
                          make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for lambda optimization.

        Parameters:
        - lambda_ (float): Regularization parameter.
        - K_AWX (jnp.ndarray): Kernel matrix for variable A, W and X.
        - K_ZZ (jnp.ndarray): Kernel matrix for variable Z.
        - identity_matrix (jnp.ndarray): Identity matrix of size n (data size of stage 1 data).
        - make_psd_eps (float, optional): Epsilon value for making matrices positive semi-definite. Defaults to 1e-9.

        Returns:
        - float: Objective value.
        """
        n = K_AWX.shape[0]
        ridge_weights = make_psd(K_AWX + n * lambda_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_AWX).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_ZZ @ H1_tilde_inv_times_H1) 
        objective += label_variance_in_lambda_opt * jnp.trace(R)
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.sum((H1_diag - 1) / H1_diag)
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.trace(R @ H1_tilde_inv @ R.T)
        return objective
    
    @staticmethod
    @jit 
    def _zeta_objective(zeta: float, 
                        K_ATildeATilde: jnp.ndarray, 
                        K_WXTilde: jnp.ndarray) -> float:
        """
        Computes the objective function for optimization with respect to zeta.

        Parameters:
        - zeta: float, parameter for optimization.
        - K_ATildeATilde: jnp.ndarray, kernel matrix for data points in set A.
        - K_WXTilde: jnp.ndarray, kernel matrix for data points in sets W and X.

        Returns:
        - float: Objective value.
        """
        m = K_ATildeATilde.shape[0]
        R = jnp.linalg.solve(make_psd(K_ATildeATilde + m * zeta * jnp.eye(m)), K_ATildeATilde)
        S = jnp.diag((1 / (1 - jnp.diag(R))) ** 2)
        T = S @ (K_WXTilde - 2 * K_WXTilde @ R.T + R @ K_WXTilde @ R.T)
        zeta_cost = jnp.trace(T)
        return zeta_cost

    @staticmethod
    @jit 
    def _eta_objective(eta, L, M, N, L2, M2, stage1_data_size, label_variance_in_eta_opt, make_psd_eps = 1e-9):
        stage2_data_size = L.shape[0] - 1
        alpha = jnp.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)
        cost = ((1 / stage1_data_size) * (alpha.T @ make_psd(L2, make_psd_eps) @ alpha) - 2 * (alpha.T @ M2))
        cost += label_variance_in_eta_opt * (2 / stage2_data_size) * jnp.trace(jnp.linalg.solve(make_psd(L + stage2_data_size * eta * N, make_psd_eps), L))
        return cost.reshape(())
    
    @staticmethod
    @jit 
    def _lambda2_objective(lambda2_: float,
                           K_AA: jnp.ndarray,
                           K_ZZ: jnp.ndarray,
                           Y: jnp.ndarray,
                           label_variance_in_lambda_opt: float, 
                           make_psd_eps: float = 1e-9) -> float:
        """
        Computes the objective function for optimization with respect to lambda2_.

        Parameters:
        - lambda2_: float, parameter for optimization.
        - K_AA: jnp.ndarray, kernel matrix for data points in set A.
        - K_ZZ: jnp.ndarray, kernel matrix for data points in set Z.
        - Y: jnp.ndarray, target values matrix.
        - make_psd_eps: float, epsilon value for making a matrix positive semi-definite (default is 1e-9).

        Returns:
        - float: Objective value.
        """
        n = K_AA.shape[0]
        identity_matrix = jnp.eye(n)
        ridge_weights = make_psd(K_AA + n * lambda2_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_AA).T
        H2 = identity_matrix - R
        H2_diag = jnp.diag(H2)
        H2_tilde_inv = jnp.diag(1 / H2_diag)
        kernel_output = K_ZZ * (Y @ Y.T)
        H2_tilde_inv_times_H2 = H2_tilde_inv @ H2
        objective = (1 / n) * jnp.trace(H2_tilde_inv_times_H2 @ kernel_output @ H2_tilde_inv_times_H2.T)
        objective += label_variance_in_lambda_opt * jnp.trace(R) 
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.sum((H2_diag - 1) / H2_diag)
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.trace(R @ H2_tilde_inv @ R.T)
        return objective

    @staticmethod
    @jit 
    def _predict_structural_function(alpha: jnp.ndarray, 
                                     B: jnp.ndarray, 
                                     B_bar: jnp.ndarray, 
                                     third_stage_KRR_weights: jnp.ndarray, 
                                     K_ATraina: jnp.ndarray, 
                                     K_ATildea: jnp.ndarray, 
                                     ones_divided_by_m: jnp.ndarray) -> float:
        """
        Predicts the structural function value.

        Parameters:
        - alpha: jnp.ndarray, parameter vector.
        - B: jnp.ndarray, matrix B.
        - B_bar: jnp.ndarray, matrix B_bar.
        - third_stage_KRR_weights: jnp.ndarray, weights from third stage KRR.
        - K_ATraina: jnp.ndarray, kernel matrix between training A and test A.
        - K_ATildea: jnp.ndarray, kernel matrix between stage 2 A and test A.
        - ones_divided_by_m: jnp.ndarray, vector of ones divided by m.

        Returns:
        - float: Predicted value.
        """
        pred = (alpha[:-1].T @ ((B.T @ (third_stage_KRR_weights @ K_ATraina)) * K_ATildea))
        pred += (alpha[-1] * ((B_bar.T @ (third_stage_KRR_weights @ K_ATraina)) * K_ATildea) @ ones_divided_by_m)
        return pred

    ########################################################################
    ###################### FIT AND PREDICT FUNCTIONS #######################
    ########################################################################
    
    def fit(self, 
            AWZX : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]],
            Y : jnp.ndarray,
            aprime : jnp.ndarray):
        """
        Fit the KernelAlternativeProxyATE model.

        Parameters:
        - AWZX (Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]): Tuple of data arrays (A, W, Z, X).
        - Y (jnp.ndarray): Target values.
        - aprime (jnp.ndarray): Historically observed treatment value a'.
        """
        kernel_A, kernel_W, kernel_Z, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X
        lambda_, eta, lambda2_, zeta = self.lambda_, self.eta, self.lambda2_, self.zeta
        optimize_lambda_parameters = self.optimize_lambda_parameters
        optimize_eta_parameter = self.optimize_eta_parameter
        optimize_zeta_parameter = self.optimize_zeta_parameter
        lambda_optimization_range = self.lambda_optimization_range
        eta_optimization_range = self.eta_optimization_range
        zeta_optimization_range = self.zeta_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        label_variance_in_lambda_opt = self.label_variance_in_lambda_opt
        label_variance_in_eta_opt = self.label_variance_in_eta_opt
        make_psd_eps = self.make_psd_eps

        if len(AWZX) == 4:
            A, W, Z, X = AWZX
        elif len(AWZX) == 3:
            A, W, Z = AWZX
            X = None
        
        aprime = jnp.array(aprime).reshape(-1, 1)
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_ZTrainZTrain = kernel_Z(Z, Z)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = kernel_X(X, X)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_WTildeWTilde = K_WTrainWTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_XTildeXTilde = K_XTrainXTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        if hasattr(self.kernel_A, 'use_length_scale_heuristic'):
            self.kernel_A.use_length_scale_heuristic = False

        if hasattr(self.kernel_W, 'use_length_scale_heuristic'):
            self.kernel_W.use_length_scale_heuristic = False

        if hasattr(self.kernel_Z, 'use_length_scale_heuristic'):
            self.kernel_Z.use_length_scale_heuristic = False

        if hasattr(self.kernel_X, 'use_length_scale_heuristic'):
            self.kernel_X.use_length_scale_heuristic = False

        K_ATildeaprime = kernel_A(A[stage2_idx], aprime)
        ########## OPTIMIZE THE REGULARIZATION PARAMETERS IF IT IS SPECIFIED ###########################
        I_n = jnp.eye(stage1_data_size)
        K_AWX = K_AA * K_WW * K_XX
        if optimize_lambda_parameters:
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda_objective_list = jnp.array([self._lambda_objective(lambda_, K_AWX, K_ZZ, I_n, label_variance_in_lambda_opt, make_psd_eps) for lambda_ in lambda_list])
            lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]
            self.lambda_objective_list = lambda_objective_list
            self.lambda_list = lambda_list

        if optimize_zeta_parameter:
            K_WX_Tilde = K_WTildeWTilde * K_XTildeXTilde
            zeta_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta_objective_list = jnp.array([self._zeta_objective(zeta, K_ATildeATilde, K_WX_Tilde) for zeta in zeta_list])
            zeta = zeta_list[jnp.argmin(zeta_objective_list).item()]
            self.zeta_objective_list = zeta_objective_list
            self.zeta_list = zeta_list

        ########### FIRST AND SECOND STAGE ########################################
        stage1_ridge_weights = (K_AWX + stage1_data_size * lambda_ * I_n)
        theta = jnp.linalg.solve(make_psd(K_ATildeATilde) + stage2_data_size * zeta * jnp.eye(stage2_data_size), K_ATildeaprime)
        ones_matrix_m = remove_diagonal_elements(jnp.ones((stage2_data_size, stage2_data_size)))
        B = jnp.linalg.solve(stage1_ridge_weights, (K_WWTilde * K_XXTilde * K_AATilde))
        B_bar = jnp.linalg.solve(stage1_ridge_weights,  (((K_WWTilde * K_XXTilde) @ (theta * ones_matrix_m)) * K_AATilde))

        block_component1 = (B.T @ K_ZZ @ B) * K_ATildeATilde
        block_component2 = (B.T @ K_ZZ @ B_bar) * K_ATildeATilde
        block_component4 = (B_bar.T @ K_ZZ @ B_bar) * K_ATildeATilde
        ones_divided_by_m = jnp.ones((stage2_data_size)) / stage2_data_size

        L = jnp.vstack((block_component1, ones_divided_by_m.T @ block_component2.T)) @ jnp.hstack((block_component1, (block_component2 @ ones_divided_by_m).reshape(-1, 1)))
        M = jnp.vstack(((block_component2 @ ones_divided_by_m).reshape(-1, 1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))

        P = jnp.hstack((block_component1, (block_component2 @ ones_divided_by_m).reshape(-1, 1)))
        R = jnp.hstack(((ones_divided_by_m.T @ block_component2.T).reshape(1, -1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))
        N = jnp.vstack((P, R))

        if optimize_eta_parameter:
            # zeta2 is only required if the parameter eta will be optimized using validation set (stage 1 data as a validation).
            K_WX = K_WW * K_XX
            zeta2_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta2_objective_list = jnp.array([self._zeta_objective(zeta, K_AA, K_WX) for zeta in zeta2_list])
            zeta2 = zeta_list[jnp.argmin(zeta2_objective_list).item()]
            K_Aaprime = kernel_A(A[stage1_idx], aprime)
            theta2 = jnp.linalg.solve(make_psd(K_AA) + stage1_data_size * zeta2 * jnp.eye(stage1_data_size), K_Aaprime)
            ones_matrix_n = remove_diagonal_elements(jnp.ones((stage1_data_size, stage1_data_size)))
            B2 = jnp.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps), K_AWX)
            B2_bar = jnp.linalg.solve(stage1_ridge_weights,  (((K_WX) @ (theta2 * ones_matrix_n)) * K_AA))
            ones_divided_by_n = jnp.ones((stage1_data_size)) / stage1_data_size    

            block_component12 = (B2.T @ K_ZZ @ B) * K_AATilde
            block_component22 = (B2.T @ K_ZZ @ B_bar) * K_AATilde
            block_component32 = (B.T @ K_ZZ @ B2_bar) * K_ATildeA
            block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_ATildeA

            L2_sub = jnp.vstack((block_component12.T, ones_divided_by_m.T @ block_component22.T))
            L2 = L2_sub @ L2_sub.T
            M2 = jnp.vstack(((block_component32 @ ones_divided_by_n).reshape(-1, 1), (ones_divided_by_m.T @ block_component42 @ ones_divided_by_n).reshape(-1, 1)))

            eta_list = np.logspace(np.log(eta_optimization_range[0]), np.log(eta_optimization_range[1]), regularization_grid_points, base = np.exp(1))
            eta_objective_list = jnp.array([self._eta_objective(eta_, L, M, N, L2, M2, stage1_data_size, label_variance_in_eta_opt, make_psd_eps) for eta_ in eta_list]).reshape(-1, 1)

            eta = eta_list[jnp.argmin(eta_objective_list).item()]
            self.eta_objective_list = eta_objective_list
            self.eta_list = eta_list

        alpha = jnp.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)

        ########### THIRD STAGE ########################################
        if optimize_lambda_parameters:
            if self.lambda2_optimization_range is None:
                lambda2_optimization_range = lambda_optimization_range
            else:
                lambda2_optimization_range = self.lambda2_optimization_range
            lambda2_list = jnp.logspace(jnp.log(lambda2_optimization_range[0]), jnp.log(lambda2_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda2_objective_list = jnp.array([self._lambda2_objective(lambda2_, K_ATrainATrain, K_ZTrainZTrain, Y, label_variance_in_lambda_opt, make_psd_eps) for lambda2_ in lambda2_list])
            lambda2_ = lambda2_list[jnp.argmin(lambda2_objective_list).item()]
            self.lambda2_objective_list = lambda2_objective_list

        K_ZZtrain = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, train_indices).T)].reshape(stage1_data_size, train_data_size)
        K_ATrainATrain_ = K_ATrainATrain[tuple(cartesian_product(train_indices, train_indices).T)].reshape(train_data_size, train_data_size)
        third_stage_KRR_weights = jnp.linalg.solve(make_psd(K_ATrainATrain_ + train_data_size * lambda2_ * jnp.eye(train_data_size)), (K_ZZtrain.T * Y[train_indices])).T 

        self.alpha = alpha
        self.B, self.B_bar = B, B_bar
        self.third_stage_KRR_weights = third_stage_KRR_weights
        self.ones_divided_by_m = ones_divided_by_m
        self.ATrain, self.ZTrain = A, Z
        self.train_indices = train_indices
        self.stage1_idx, self.stage2_idx = stage1_idx, stage2_idx

        ##### For debugging purpose, I might want to check the regularization values after optimization #######
        self.lambda_ = lambda_
        self.zeta = zeta
        self.eta = eta
        self.lambda2_ = lambda2_

    def predict(self, A: jnp.ndarray) -> jnp.ndarray:
        """
        Predict outcomes for new data points.

        Parameters:
        - A (jnp.ndarray): New data points for variable A.

        Returns:
        - jnp.ndarray: Predicted values.
        """
        A_test = A.reshape(-1, 1)
        K_ATrainATest = self.kernel_A(self.ATrain, A_test)

        test_indices = jnp.arange(A_test.shape[0])
        test_shape = test_indices.shape[0]

        K_ATrainATest_ = K_ATrainATest[tuple(cartesian_product(self.train_indices, test_indices).T)].reshape(self.train_indices.shape[0], test_shape)
        K_ATildeATest = K_ATrainATest[tuple(cartesian_product(self.stage2_idx, test_indices).T)].reshape(self.stage2_idx.shape[0], test_shape)

        ones_divided_by_m = self.ones_divided_by_m
        alpha = self.alpha
        B, B_bar = self.B, self.B_bar
        third_stage_KRR_weights = self.third_stage_KRR_weights

        f_struct_pred = jnp.array([self._predict_structural_function(alpha, B, B_bar, third_stage_KRR_weights, 
                                                                     K_ATrainATest_[:, jj], K_ATildeATest[:, jj], 
                                                                     ones_divided_by_m).item() for jj in range(K_ATildeATest.shape[1])])
        return f_struct_pred

    def _predict_bridge_func(self, A_test : jnp.ndarray, Z_test : jnp.ndarray, X_test = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ZZTest = self.kernel_Z(self.ZTrain[self.stage1_idx, :], Z_test)
        K_ATildeATest = self.kernel_A(self.ATrain[self.stage2_idx, :], A_test)
        alpha, B, B_bar = self.alpha, self.B, self.B_bar
        ones_divided_by_m = self.ones_divided_by_m
        bridge_function = jnp.array([alpha[:-1].T @ ((B.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1, 1)) + alpha[-1] * ones_divided_by_m.T @ ((B_bar.T @ K_ZZTest) * K_ATildeATest[:, jj].reshape(-1,1)) for jj in range(K_ATildeATest.shape[1])])
        return bridge_function[:, 0, :]


class KernelAlternativeProxyCATE(BaseEstimator, RegressorMixin):

    def __init__(self,
                 kernel_A: Kernel,
                 kernel_W: Kernel,
                 kernel_Z: Kernel,
                 kernel_V: Kernel,
                 kernel_X: Kernel = RBF(),
                 lambda_: float = 0.1,
                 eta: float = 0.1, 
                 lambda2_: float = 0.1,
                 zeta: float = 0.1,
                 optimize_lambda_parameters: bool = True,
                 optimize_eta_parameter: bool = True,
                 optimize_zeta_parameter: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 eta_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 **kwargs) -> None:
        """
        Initialize the KernelAlternativeProxyATE estimator.

        Parameters:
        - kernel_A (Kernel): Kernel function for variable A.
        - kernel_W (Kernel): Kernel function for variable W.
        - kernel_Z (Kernel): Kernel function for variable Z.
        - kernel_X (Kernel, optional): Kernel function for variable X. Defaults to RBF().
        - lambda_ (float, optional): Regularization parameter. Defaults to 0.1.
        - eta (float, optional): Regularization parameter for structural function prediction. Defaults to 0.1.
        - lambda2_ (float, optional): Second stage regularization parameter. Defaults to 0.1.
        - optimize_lambda_parameters (bool, optional): Flag to optimize lambda regularization parameters. Defaults to True.
        - optimize_eta_parameters (bool, optional): Flag to optimize eta regularization parameter. Defaults to True.
        - lambda_optimization_range (Tuple[float, float], optional): Range for lambda optimization. Defaults to (1e-7, 1.0).
        - eta_optimization_range (Tuple[float, float], optional): Range for eta optimization. Defaults to (1e-7, 1.0).
        - **kwargs: Additional parameters.
        """
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        label_variance_in_lambda_opt = kwargs.pop('label_variance_in_lambda_opt', 0.0)
        label_variance_in_eta_opt = kwargs.pop('label_variance_in_eta_opt', 0.0)
        large_lambda_eta_option = kwargs.pop('large_lambda_eta_option', False)
        selecting_biggest_lambda_tol = kwargs.pop('selecting_biggest_lambda_tol', 1e-9)
        selecting_biggest_eta_tol = kwargs.pop('selecting_biggest_eta_tol', 1e-9)
        selecting_biggest_lambda2_tol = kwargs.pop('selecting_biggest_zeta_tol', 1e-9)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_V_params = kwargs.pop('kernel_V_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Kernel)):
            raise Exception("Kernel for A must be callable Kernel class")
        if (not isinstance(kernel_W, Kernel)):
            raise Exception("Kernel for W must be callable Kernel class")
        if (not isinstance(kernel_Z, Kernel)):
            raise Exception("Kernel for Z must be callable Kernel class")
        if (not isinstance(kernel_V, Kernel)):
            raise Exception("Kernel for V must be callable Kernel class")
        if (not isinstance(kernel_X, Kernel)):
            raise Exception("Kernel for X must be callable Kernel class")
        
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_V = kernel_V
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_V_params is not None:
            self.kernel_X.set_params(**kernel_V_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda_, self.eta, self.lambda2_, self.zeta = lambda_, eta, lambda2_, zeta
        self.optimize_lambda_parameters = optimize_lambda_parameters
        self.optimize_eta_parameter = optimize_eta_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda_optimization_range = lambda_optimization_range
        self.eta_optimization_range = eta_optimization_range
        self.zeta_optimization_range = zeta_optimization_range
        self.large_lambda_eta_option = large_lambda_eta_option
        self.selecting_biggest_lambda_tol = selecting_biggest_lambda_tol
        self.selecting_biggest_eta_tol = selecting_biggest_eta_tol
        self.selecting_biggest_lambda2_tol = selecting_biggest_lambda2_tol
        self.label_variance_in_lambda_opt = label_variance_in_lambda_opt
        self.label_variance_in_eta_opt = label_variance_in_eta_opt
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps        

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################        
    @staticmethod
    @jit
    def _lambda_objective(lambda_: float, 
                          K_AWVX: jnp.ndarray, 
                          K_ZZ: jnp.ndarray, 
                          identity_matrix: jnp.ndarray,
                          label_variance_in_lambda_opt: float, 
                          make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for lambda optimization.

        Parameters:
        - lambda_ (float): Regularization parameter.
        - K_AWVX (jnp.ndarray): Kernel matrix for variable A, W, V and X.
        - K_ZZ (jnp.ndarray): Kernel matrix for variable Z.
        - identity_matrix (jnp.ndarray): Identity matrix of size n (data size of stage 1 data).
        - make_psd_eps (float, optional): Epsilon value for making matrices positive semi-definite. Defaults to 1e-9.

        Returns:
        - float: Objective value.
        """
        n = K_AWVX.shape[0]
        ridge_weights = make_psd(K_AWVX + n * lambda_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_AWVX).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_ZZ @ H1_tilde_inv_times_H1) 
        objective += label_variance_in_lambda_opt * jnp.trace(R)
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.sum((H1_diag - 1) / H1_diag)
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.trace(R @ H1_tilde_inv @ R.T)
        return objective
    
    @staticmethod
    @jit 
    def _zeta_objective(zeta: float, 
                        K_VTildeVTilde: jnp.ndarray, 
                        K_WXTilde: jnp.ndarray) -> float:
        """
        Computes the objective function for optimization with respect to zeta.

        Parameters:
        - zeta: float, parameter for optimization.
        - K_VTildeVTilde: jnp.ndarray, kernel matrix for data points in set V.
        - K_WXTilde: jnp.ndarray, kernel matrix for data points in sets W and X.

        Returns:
        - float: Objective value.
        """
        m = K_VTildeVTilde.shape[0]
        R = jnp.linalg.solve(make_psd(K_VTildeVTilde + m * zeta * jnp.eye(m)), K_VTildeVTilde)
        S = jnp.diag((1 / (1 - jnp.diag(R))) ** 2)
        T = S @ (K_WXTilde - 2 * K_WXTilde @ R.T + R @ K_WXTilde @ R.T)
        zeta_cost = jnp.trace(T)
        return zeta_cost

    @staticmethod
    @jit 
    def _eta_objective(eta, L, L_sub, M, N, L2, M2, stage1_data_size, label_variance_in_eta_opt, make_psd_eps = 1e-9):
        stage2_data_size = L.shape[0] - 1
        alpha = jnp.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)
        cost = ((1 / stage1_data_size) * (alpha.T @ make_psd(L2, make_psd_eps) @ alpha) - 2 * (alpha.T @ M2)) 
        cost += label_variance_in_eta_opt * (2 / stage2_data_size) * jnp.trace(jnp.linalg.solve(make_psd(L + stage2_data_size * eta * N, make_psd_eps), L))
        return cost.reshape(())
    
    @staticmethod
    @jit 
    def _lambda2_objective(lambda2_: float,
                           K_AA_VV: jnp.ndarray,
                           K_ZZ: jnp.ndarray,
                           Y: jnp.ndarray,
                           label_variance_in_lambda_opt: float, 
                           make_psd_eps: float = 1e-9) -> float:
        """
        Computes the objective function for optimization with respect to lambda2_.

        Parameters:
        - lambda2_ (float): Parameter for optimization.
        - K_AA_VV (jnp.ndarray): Elementwise multiplication kernel matrices for data points in set A and V.
        - K_ZZ (jnp.ndarray): Kernel matrix for data points in set Z.
        - Y (jnp.ndarray): Target values matrix.
        - make_psd_eps (float, optional): Epsilon value for making a matrix positive semidefinite. Defaults to 1e-9.

        Returns:
        - float: Objective value.
        """
        n = K_AA_VV.shape[0]
        identity_matrix = jnp.eye(n)
        ridge_weights = make_psd(K_AA_VV + n * lambda2_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_AA_VV).T
        H2 = identity_matrix - R
        H2_diag = jnp.diag(H2)
        H2_tilde_inv = jnp.diag(1 / H2_diag)
        kernel_output = K_ZZ * (Y @ Y.T)
        H2_tilde_inv_times_H2 = H2_tilde_inv @ H2
        objective = (1 / n) * jnp.trace(H2_tilde_inv_times_H2 @ kernel_output @ H2_tilde_inv_times_H2.T)
        objective += label_variance_in_lambda_opt * jnp.trace(R) 
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.sum((H2_diag - 1) / H2_diag)
        objective += (1 / n) * label_variance_in_lambda_opt * jnp.trace(R @ H2_tilde_inv @ R.T)
        return objective
    
    @staticmethod
    @jit 
    def _predict_structural_function(alpha: jnp.ndarray,
                                     B: jnp.ndarray,
                                     B_bar: jnp.ndarray,
                                     third_stage_KRR_weights: jnp.ndarray,
                                     K_ATraina: jnp.ndarray,
                                     K_ATildea: jnp.ndarray,
                                     K_VTrainv: jnp.ndarray,
                                     K_VTildev: jnp.ndarray,
                                     ones_divided_by_m: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the structural function.

        Parameters:
        - alpha (jnp.ndarray): Coefficient array.
        - B (jnp.ndarray): Matrix B from second stage.
        - B_bar (jnp.ndarray): Matrix B_bar from second stage.
        - third_stage_KRR_weights (jnp.ndarray): Weights from third stage kernel ridge regression.
        - K_ATraina (jnp.ndarray): Kernel matrix between training set A and a test point.
        - K_ATildea (jnp.ndarray): Kernel matrix between stage 2 set A and a test point.
        - ones_divided_by_m (jnp.ndarray): Array of ones divided by stage 2 data size.

        Returns:
        - jnp.ndarray: Predicted values.
        """
        pred = (alpha[:-1].T @ ((B.T @ (third_stage_KRR_weights @ (K_ATraina * K_VTrainv))) * (K_ATildea * K_VTildev)))
        pred += (alpha[-1] * ((B_bar.T @ (third_stage_KRR_weights @ (K_ATraina * K_VTrainv))) * (K_ATildea * K_VTildev)) @ ones_divided_by_m)
        return pred


    ########################################################################
    ###################### FIT AND PREDICT FUNCTIONS #######################
    ########################################################################
    def fit(self, 
            AWZVX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray,) -> None:
        """
        Fit the KernelAlternativeProxyATE model.

        Parameters:
        - AWZVX (Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]): Tuple of data arrays (A, W, Z, X).
        - Y (np.ndarray): Target values.
        """
        kernel_A, kernel_W, kernel_Z, kernel_V, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_V, self.kernel_X
        lambda_, eta, lambda2_, zeta = self.lambda_, self.eta, self.lambda2_, self.zeta
        optimize_lambda_parameters = self.optimize_lambda_parameters
        optimize_eta_parameter = self.optimize_eta_parameter
        optimize_zeta_parameter = self.optimize_zeta_parameter
        lambda_optimization_range = self.lambda_optimization_range
        eta_optimization_range = self.eta_optimization_range
        zeta_optimization_range = self.zeta_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps
        large_lambda_eta_option = self.large_lambda_eta_option
        selecting_biggest_lambda_tol = self.selecting_biggest_lambda_tol
        selecting_biggest_eta_tol = self.selecting_biggest_eta_tol
        selecting_biggest_lambda2_tol = self.selecting_biggest_lambda2_tol
        label_variance_in_lambda_opt = self.label_variance_in_lambda_opt
        label_variance_in_eta_opt = self.label_variance_in_eta_opt
        
        if len(AWZVX) == 5:
            A, W, Z, V, X = AWZVX
        elif len(AWZVX) == 4:
            A, W, Z, V = AWZVX
            X = None
        
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_ZTrainZTrain = kernel_Z(Z, Z)
        K_VTrainVTrain = kernel_V(V, V)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = make_psd(kernel_X(X, X), make_psd_eps)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_V = kernel_V
        self.kernel_X = kernel_X

        ############################# SPLIT DATA IN STAGE 1 AND STAGE 2 #####################################
        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        ################################ KERNEL MATRICES ####################################################
        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_WTildeWTilde = K_WTrainWTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)
        K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)

        K_VV = K_VTrainVTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_VVTilde = K_VTrainVTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_VTildeV = K_VVTilde.T
        K_VTildeVTilde = K_VTrainVTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XTildeXTilde = K_XTrainXTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_V, self.kernel_X]:
            if hasattr(kernel_, 'use_length_scale_heuristic'):
                kernel_.use_length_scale_heuristic = False

        ########## OPTIMIZE THE LAMBDA REGULARIZATION PARAMETER IF IT IS SPECIFIED ###########################
        I_n = jnp.eye(stage1_data_size)
        K_AWVX = K_AA * K_WW * K_VV * K_XX
        if optimize_lambda_parameters:
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            self.lambda_list = lambda_list
            lambda_objective_list = jnp.array([self._lambda_objective(lambda_, K_AWVX, K_ZZ, 
                                                                      I_n, label_variance_in_lambda_opt, make_psd_eps) for lambda_ in lambda_list])
            self.lambda_objective_list = lambda_objective_list
            if large_lambda_eta_option:
                lambda_objective_min = jnp.min(lambda_objective_list).item()
                _lambda_objective_list_minimizer_list = jnp.where(jnp.abs(lambda_objective_list - lambda_objective_min) <= selecting_biggest_lambda_tol)[0]
                lambda_ = jnp.max(lambda_list[_lambda_objective_list_minimizer_list])
            else:
                lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]

        if optimize_zeta_parameter:
            K_WX_Tilde = K_WTildeWTilde * K_XTildeXTilde
            zeta_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta_objective_list = jnp.array([self._zeta_objective(zeta, K_VTildeVTilde, K_WX_Tilde) for zeta in zeta_list])
            zeta = zeta_list[jnp.argmin(zeta_objective_list).item()]
            self.zeta_objective_list = zeta_objective_list
            self.zeta_list = zeta_list

        ########### FIRST AND SECOND STAGE REGRESSION ########################################
        stage1_ridge_weights = (K_AWVX + stage1_data_size * lambda_ * I_n)
        theta_matrix = jnp.linalg.solve(make_psd(K_VTildeVTilde) + stage2_data_size * zeta * jnp.eye(stage2_data_size), K_VTildeVTilde)
        ones_matrix_m = remove_diagonal_elements(jnp.ones((stage2_data_size, stage2_data_size)))
        self.stage1_ridge_weights = stage1_ridge_weights
        B = jnp.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps), (K_WWTilde * K_XXTilde * K_AATilde * K_VVTilde))
        B_bar = jnp.linalg.solve(stage1_ridge_weights,  (((K_WWTilde * K_XXTilde) @ (theta_matrix * ones_matrix_m)) * (K_AATilde * K_VVTilde)))

        block_component1 = (B.T @ K_ZZ @ B) * K_ATildeATilde * K_VTildeVTilde
        block_component2 = (B.T @ K_ZZ @ B_bar) * K_ATildeATilde * K_VTildeVTilde
        block_component4 = (B_bar.T @ K_ZZ @ B_bar) * K_ATildeATilde * K_VTildeVTilde
        ones_divided_by_m = jnp.ones((stage2_data_size)) / stage2_data_size

        L_sub = jnp.vstack((block_component1, ones_divided_by_m.T @ block_component2.T))
        L = L_sub @ L_sub.T
        self.L = L_sub.T
        M = jnp.vstack(((block_component2 @ ones_divided_by_m).reshape(-1, 1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))
        
        P = jnp.hstack((block_component1, (block_component2 @ ones_divided_by_m).reshape(-1, 1)))
        R = jnp.hstack(((ones_divided_by_m.T @ block_component2.T).reshape(1, -1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))
        N = jnp.vstack((P, R))

        if optimize_eta_parameter:
            theta_matrix2 = jnp.linalg.solve(make_psd(K_VV) + stage1_data_size * zeta * jnp.eye(stage1_data_size), K_VV)
            ones_matrix_n = remove_diagonal_elements(jnp.ones((stage1_data_size, stage1_data_size)))
            B2 = jnp.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps), K_AWVX)
            B2_bar = jnp.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps),  (((K_WW * K_XX) @ (theta_matrix2 * ones_matrix_n)) * (K_AA * K_VV)))
            ones_divided_by_n = jnp.ones((stage1_data_size)) / stage1_data_size    

            block_component12 = (B2.T @ K_ZZ @ B) * K_AATilde * K_VVTilde
            block_component22 = (B2.T @ K_ZZ @ B_bar) * K_AATilde * K_VVTilde
            block_component32 = (B.T @ K_ZZ @ B2_bar) * K_ATildeA * K_VTildeV
            block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_ATildeA * K_VTildeV

            L2_sub = jnp.vstack((block_component12.T, ones_divided_by_m.T @ block_component22.T))
            L2 = L2_sub @ L2_sub.T
            M2 = jnp.vstack(((block_component32 @ ones_divided_by_n).reshape(-1, 1), (ones_divided_by_m.T @ block_component42 @ ones_divided_by_n).reshape(-1, 1)))

            eta_list = np.logspace(np.log(eta_optimization_range[0]), np.log(eta_optimization_range[1]), regularization_grid_points, base = np.exp(1))
            self.eta_list = eta_list
            eta_objective_list = jnp.array([self._eta_objective(eta_, L, L_sub, M, N, L2, M2, stage1_data_size, 
                                                                label_variance_in_eta_opt, make_psd_eps) for eta_ in eta_list]).reshape(-1, 1)

            if large_lambda_eta_option:
                eta_objective_min = jnp.min(eta_objective_list).item()
                _eta_objective_list_minimizer_list = jnp.where(jnp.abs(eta_objective_list - eta_objective_min) <= selecting_biggest_eta_tol)[0]
                eta = jnp.max(eta_list[_eta_objective_list_minimizer_list])
            else:
                eta = eta_list[jnp.argmin(eta_objective_list).item()]
            self.eta_objective_list = eta_objective_list
            self.final_second_stage_validation_loss = self._eta_objective(eta, L, L_sub, M, N, L2, M2, stage1_data_size, 
                                                                          0, make_psd_eps)
        alpha = jnp.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)
        ########### THIRD STAGE ########################################
        if optimize_lambda_parameters:
            lambda2_list = lambda_list.copy()
            lambda2_objective_list = jnp.array([self._lambda2_objective(lambda2_, K_ATrainATrain * K_VTrainVTrain, K_ZTrainZTrain, Y,
                                                                        label_variance_in_lambda_opt, make_psd_eps) for lambda2_ in lambda_list])
            self.lambda2_objective_list = lambda2_objective_list
            if large_lambda_eta_option:
                lambda2_objective_min = jnp.min(lambda2_objective_list).item()
                _lambda2_objective_list_minimizer_list = jnp.where(jnp.abs(lambda2_objective_list - lambda2_objective_min) <= selecting_biggest_lambda2_tol)[0]
                lambda2_ = jnp.max(lambda2_list[_lambda2_objective_list_minimizer_list])
            else:
                lambda2_ = lambda2_list[jnp.argmin(lambda2_objective_list).item()]

        K_ZZTrain = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, train_indices).T)].reshape(stage1_data_size, train_data_size)
        K_ATrainATrain_ = K_ATrainATrain[tuple(cartesian_product(train_indices, train_indices).T)].reshape(train_data_size, train_data_size)
        K_VTrainVTrain_ = K_VTrainVTrain[tuple(cartesian_product(train_indices, train_indices).T)].reshape(train_data_size, train_data_size)
        third_stage_KRR_weights = jnp.linalg.solve(make_psd(K_ATrainATrain_ * K_VTrainVTrain_ + train_data_size * lambda2_ * jnp.eye(train_data_size), make_psd_eps), (K_ZZTrain.T * Y[train_indices])).T 

        self.alpha = alpha
        self.B, self.B_bar = B, B_bar
        self.third_stage_KRR_weights = third_stage_KRR_weights
        self.ones_divided_by_m = ones_divided_by_m
        self.ATrain, self.WTrain, self.XTrain, self.VTrain, self.ZTrain = A, W, X, V, Z
        self.K_ZZ = K_ZZ
        self.train_indices = train_indices
        self.stage1_idx, self.stage2_idx = stage1_idx, stage2_idx

        ##### For debugging purpose, I might want to check the regularization values after optimization #######
        self.lambda_ = lambda_
        self.lambda2_ = lambda2_
        self.eta = eta

    def predict(self, A: jnp.ndarray, V: jnp.array) -> jnp.ndarray:
        """
        Predict outcomes for new data points.

        Parameters:
        - A (jnp.ndarray): New data points for variable A.
        - V (jnp.ndarray): New data points for variable V.

        Returns:
        - jnp.ndarray: Predicted values.
        """
        if A.ndim != 2:
            A_test = A.reshape(-1, 1)
        else:
            A_test = A
        V_test = V
        K_ATrainATest = self.kernel_A(self.ATrain, A_test)
        K_VTrainVTest = self.kernel_V(self.VTrain, V_test)

        test_indices = jnp.arange(A_test.shape[0])
        test_shape = test_indices.shape[0]

        K_ATrainATest_ = K_ATrainATest[tuple(cartesian_product(self.train_indices, test_indices).T)].reshape(self.train_indices.shape[0], test_shape)
        K_ATildeATest = K_ATrainATest[tuple(cartesian_product(self.stage2_idx, test_indices).T)].reshape(self.stage2_idx.shape[0], test_shape)
        K_VTrainVTest_ = K_VTrainVTest[tuple(cartesian_product(self.train_indices, test_indices).T)].reshape(self.train_indices.shape[0], test_shape)
        K_VTildeVTest = K_VTrainVTest[tuple(cartesian_product(self.stage2_idx, test_indices).T)].reshape(self.stage2_idx.shape[0], test_shape)

        ones_divided_by_m = self.ones_divided_by_m
        alpha = self.alpha
        B, B_bar = self.B, self.B_bar
        third_stage_KRR_weights = self.third_stage_KRR_weights

        f_struct_pred = jnp.array([self._predict_structural_function(alpha, B, B_bar, third_stage_KRR_weights, 
                                                                     K_ATrainATest_[:, jj], K_ATildeATest[:, jj],
                                                                     K_VTrainVTest_[:, jj], K_VTildeVTest[:, jj], 
                                                                     ones_divided_by_m).item() for jj in range(K_ATildeATest.shape[1])])
        return f_struct_pred 

    def _predict_bridge_func(self, A_test : jnp.ndarray, Z_test : jnp.ndarray, V_test: jnp.array, X_test = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ZZTest = self.kernel_Z(self.ZTrain[self.stage1_idx, :], Z_test)
        K_ATildeATest = self.kernel_A(self.ATrain[self.stage2_idx, :], A_test)
        K_VTildeVTest = self.kernel_V(self.VTrain[self.stage2_idx, :], V_test)
        alpha, B, B_bar = self.alpha, self.B, self.B_bar
        ones_divided_by_m = self.ones_divided_by_m
        bridge_function = jnp.array([alpha[:-1].T @ ((B.T @ K_ZZTest) * (K_ATildeATest[:, jj] * K_VTildeVTest[:, jj]).reshape(-1, 1)) + alpha[-1] * ones_divided_by_m.T @ ((B_bar.T @ K_ZZTest) * (K_ATildeATest[:, jj] * K_VTildeVTest[:, jj]).reshape(-1,1)) for jj in range(K_ATildeATest.shape[1])])
        return bridge_function[:, 0, :]
    

class KernelProxyVariableATE(BaseEstimator, RegressorMixin):

    def __init__(self, 
                 kernel_A : Kernel,
                 kernel_W : Kernel,
                 kernel_Z : Kernel,
                 kernel_X : Kernel = RBF(),
                 lambda1_ : float = 0.1,
                 lambda2_ : float = 0.1,
                 optimize_lambda1_parameter : bool = True,
                 optimize_lambda2_parameter : bool = True,
                 lambda1_optimization_range : Tuple[float, float] = (1e-5, 1.),
                 lambda2_optimization_range : Tuple[float, float] = (1e-5, 1.),
                 **kwargs) -> None:
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Kernel)):
            raise Exception("Kernel for A must be callable Kernel class")
        if (not isinstance(kernel_W, Kernel)):
            raise Exception("Kernel for W must be callable Kernel class")
        if (not isinstance(kernel_Z, Kernel)):
            raise Exception("Kernel for Z must be callable Kernel class")
        if (not isinstance(kernel_X, Kernel)):
            raise Exception("Kernel for X must be callable Kernel class")
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda1_, self.lambda2_ = lambda1_, lambda2_
        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps 

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################        
    @staticmethod
    @jit
    def _lambda1_objective(lambda_: float, 
                           K_ZAX: jnp.ndarray, 
                           K_WW: jnp.ndarray, 
                           identity_matrix: jnp.ndarray,
                           make_psd_eps: float = 1e-9) -> float:
        n = K_ZAX.shape[0]
        ridge_weights = make_psd(K_ZAX + n * lambda_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_ZAX).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_WW @ H1_tilde_inv_times_H1) 
        return objective
    
    @staticmethod
    @jit
    def _lambda2_objective(lambda_: float, 
                            second_stage_ridge_weights: jnp.ndarray, 
                            K_YTilde: jnp.ndarray, 
                            identity_matrix: jnp.ndarray,
                            make_psd_eps: float = 1e-9) -> float:
        n = second_stage_ridge_weights.shape[0]
        ridge_weights = make_psd(second_stage_ridge_weights + n * lambda_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, second_stage_ridge_weights).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_YTilde @ H1_tilde_inv_times_H1) 
        return objective

    def fit(self,             
            AWZX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray,) -> None:
        kernel_A, kernel_W, kernel_Z, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X
        lambda1_, lambda2_ = self.lambda1_, self.lambda2_
        optimize_lambda1_parameter = self.optimize_lambda1_parameter
        optimize_lambda2_parameter = self.optimize_lambda2_parameter
        lambda1_optimization_range = self.lambda1_optimization_range
        lambda2_optimization_range = self.lambda2_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps

        if len(AWZX) == 4:
            A, W, Z, X = AWZX
        elif len(AWZX) == 3:
            A, W, Z = AWZX
            X = None
        
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_ZTrainZTrain = kernel_Z(Z, Z)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = make_psd(kernel_X(X, X), make_psd_eps)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        ############################# SPLIT DATA IN STAGE 1 AND STAGE 2 #####################################
        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        ################################ KERNEL MATRICES ####################################################
        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        # K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_ZZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_XTildeXTilde = K_XTrainXTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, 'use_length_scale_heuristic'):
                kernel_.use_length_scale_heuristic = False
                
        ########### FIRST STAGE REGRESSION ###########################
        K_ZAX = K_ZZ * K_AA * K_XX
        I_n = jnp.eye(stage1_data_size)
        I_m = jnp.eye(stage2_data_size)
        YTilde = Y[stage2_idx]

        if optimize_lambda1_parameter:
            lambda1_list = jnp.logspace(jnp.log(lambda1_optimization_range[0]), jnp.log(lambda1_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda1_objective_list = jnp.array([self._lambda1_objective(lambda_, K_ZAX, K_WW, 
                                                                            I_n, make_psd_eps) for lambda_ in lambda1_list])
            lambda1_ = lambda1_list[jnp.argmin(lambda1_objective_list).item()]
            self.lambda1_ = lambda1_
        ########### SECOND STAGE REGRESSION ###########################
        stage1_ridge_weights = (K_ZAX + stage1_data_size * lambda1_ * I_n)
        K_ZAX_ZAXTilde = K_ZZTilde * K_AATilde * K_XXTilde

        B = jnp.linalg.solve(make_psd(stage1_ridge_weights), K_ZAX_ZAXTilde)
        self.B = B
        stage2_ridge_weights = K_ATildeATilde * (B.T @ K_WW @ B) * K_XTildeXTilde

        x_mean_vec = jnp.mean(K_XXTilde, axis=0)[:, jnp.newaxis]
        if optimize_lambda2_parameter:
            K_YTilde = YTilde @ YTilde.T
            lambda2_list = jnp.logspace(jnp.log(lambda2_optimization_range[0]), jnp.log(lambda2_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda2_objective_list = jnp.array([self._lambda2_objective(lambda_, stage2_ridge_weights, K_YTilde, 
                                                                            I_m, make_psd_eps) for lambda_ in lambda2_list])
            lambda2_ = lambda2_list[jnp.argmin(lambda2_objective_list).item()]
            self.lambda2_ = lambda2_

        stage2_ridge_weights += stage2_data_size * lambda2_ * I_m
        alpha = jnp.linalg.solve(make_psd(stage2_ridge_weights), YTilde)
        w_mean_vec = jnp.mean(K_WW, axis=0)[:, jnp.newaxis]

        self.alpha = alpha
        self.w_mean_vec = w_mean_vec
        self.x_mean_vec = x_mean_vec
        self.ATilde = A[stage2_idx]
        if X is not None:
            self.XTilde = X[stage2_idx]
        else:
            self.XTilde = None
        self.W = W[stage1_idx]
        self.upweight = jnp.mean((B.T @ K_WW) * K_XXTilde.T, axis = 1).reshape(-1, 1)

    def predict(self, A: jnp.ndarray) -> jnp.ndarray:
        if A.ndim != 2:
            A_test = A.reshape(-1, 1)
        else:
            A_test = A
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)

        pred = (K_ATildeATest * self.upweight).T @ self.alpha
        ## The following line is based on the implementation in https://github.com/liyuan9988/DeepFeatureProxyVariable/blob/master/src/models/kernelPV/model.py
        ## However, it is wrong. THe correct version is given in the line above.
        # pred = (K_ATildeATest * (self.B.T @ self.w_mean_vec) * self.x_mean_vec).T @ self.alpha
        return pred
    
    def _predict_bridge_func(self, A_test : jnp.ndarray, W_test : jnp.ndarray, X_test : jnp.array = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_WWTest = self.kernel_W(self.W, W_test)
        if (X_test is not None) & (self.XTilde is not None):
            K_XTildeXTest = self.kernel_X(self.XTilde, X_test)
        else:
            K_XTildeXTest = jnp.ones((self.ATilde.shape[0], W_test.shape[0]))
        
        # bridge_pred = (K_ATildeATest * (self.B.T @ K_WWTest)).T @ self.alpha 
        bridge_pred = jnp.array([(K_ATildeATest[:, jj].reshape(-1, 1) * ((self.B.T @ K_WWTest) * K_XTildeXTest)).T @ self.alpha for jj in range(A_test.shape[0])])
        return bridge_pred[:, :, 0]


class KernelProxyVariableATT(BaseEstimator, RegressorMixin):

    def __init__(self, 
                 kernel_A : Kernel,
                 kernel_W : Kernel,
                 kernel_Z : Kernel,
                 kernel_X : Kernel = RBF(),
                 lambda1_ : float = 0.1,
                 lambda2_ : float = 0.1,
                 zeta: float = 0.1,
                 optimize_lambda1_parameter : bool = True,
                 optimize_lambda2_parameter : bool = True,
                 optimize_zeta_parameter: bool = True,
                 lambda1_optimization_range : Tuple[float, float] = (1e-5, 1.),
                 lambda2_optimization_range : Tuple[float, float] = (1e-5, 1.),
                 zeta_optimization_range: Tuple[float, float] = (1e-5, 1.),
                 **kwargs) -> None:
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Kernel)):
            raise Exception("Kernel for A must be callable Kernel class")
        if (not isinstance(kernel_W, Kernel)):
            raise Exception("Kernel for W must be callable Kernel class")
        if (not isinstance(kernel_Z, Kernel)):
            raise Exception("Kernel for Z must be callable Kernel class")
        if (not isinstance(kernel_X, Kernel)):
            raise Exception("Kernel for X must be callable Kernel class")
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda1_, self.lambda2_, self.zeta = lambda1_, lambda2_, zeta
        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range
        self.zeta_optimization_range = zeta_optimization_range
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps 

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################        
    @staticmethod
    @jit
    def _lambda1_objective(lambda_: float, 
                           K_ZAX: jnp.ndarray, 
                           K_WW: jnp.ndarray, 
                           identity_matrix: jnp.ndarray,
                           make_psd_eps: float = 1e-9) -> float:
        n = K_ZAX.shape[0]
        ridge_weights = make_psd(K_ZAX + n * lambda_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_ZAX).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_WW @ H1_tilde_inv_times_H1) 
        return objective
    
    @staticmethod
    @jit
    def _lambda2_objective(lambda_: float, 
                            second_stage_ridge_weights: jnp.ndarray, 
                            K_YTilde: jnp.ndarray, 
                            identity_matrix: jnp.ndarray,
                            make_psd_eps: float = 1e-9) -> float:
        n = second_stage_ridge_weights.shape[0]
        ridge_weights = make_psd(second_stage_ridge_weights + n * lambda_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, second_stage_ridge_weights).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_YTilde @ H1_tilde_inv_times_H1) 
        return objective

    @staticmethod
    @jit
    def zeta_objective(zeta: float, K_AA: jnp.ndarray, K_XX_WW: jnp.ndarray, make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for zeta optimization.

        Parameters:
        - zeta: float, regularization parameter.
        - K_AA: jnp.ndarray, kernel matrix for variable A.
        - K_XX_WW: jnp.ndarray, combined kernel matrix for variables X and W.

        Returns:
        - float, objective value.
        """
        n = K_AA.shape[0]
        # R = K_AA @ jnp.linalg.inv(K_AA + n * zeta * jnp.eye(n))
        R = jnp.linalg.solve(make_psd(K_AA + n * zeta * jnp.eye(n), make_psd_eps), K_AA).T
        S = jnp.diag((1 / (1 - jnp.diag(R))) ** 2)
        T = S @ (K_XX_WW - 2 * K_XX_WW @ R.T + R @ K_XX_WW @ R.T)
        zeta_cost = jnp.trace(T)
        return zeta_cost

    def fit(self,             
            AWZX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray,) -> None:
        kernel_A, kernel_W, kernel_Z, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X
        lambda1_, lambda2_ = self.lambda1_, self.lambda2_
        optimize_lambda1_parameter = self.optimize_lambda1_parameter
        optimize_lambda2_parameter = self.optimize_lambda2_parameter
        optimize_zeta_parameter = self.optimize_zeta_parameter
        lambda1_optimization_range = self.lambda1_optimization_range
        lambda2_optimization_range = self.lambda2_optimization_range
        zeta_optimization_range = self.zeta_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps

        if len(AWZX) == 4:
            A, W, Z, X = AWZX
        elif len(AWZX) == 3:
            A, W, Z = AWZX
            X = None
        
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_ZTrainZTrain = kernel_Z(Z, Z)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = make_psd(kernel_X(X, X), make_psd_eps)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        ############################# SPLIT DATA IN STAGE 1 AND STAGE 2 #####################################
        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        ################################ KERNEL MATRICES ####################################################
        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_WTildeWTilde = K_WTrainWTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_ZZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_XTildeXTilde = K_XTrainXTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, 'use_length_scale_heuristic'):
                kernel_.use_length_scale_heuristic = False
                
        ########### FIRST STAGE REGRESSION ###########################
        K_ZAX = K_ZZ * K_AA * K_XX
        I_n = jnp.eye(stage1_data_size)
        I_m = jnp.eye(stage2_data_size)
        YTilde = Y[stage2_idx]

        if optimize_lambda1_parameter:
            lambda1_list = jnp.logspace(jnp.log(lambda1_optimization_range[0]), jnp.log(lambda1_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda1_objective_list = jnp.array([self._lambda1_objective(lambda_, K_ZAX, K_WW, 
                                                                            I_n, make_psd_eps) for lambda_ in lambda1_list])
            lambda1_ = lambda1_list[jnp.argmin(lambda1_objective_list).item()]
            self.lambda1_ = lambda1_
        ########### SECOND STAGE REGRESSION ###########################
        stage1_ridge_weights = (K_ZAX + stage1_data_size * lambda1_ * I_n)
        K_ZAX_ZAXTilde = K_ZZTilde * K_AATilde * K_XXTilde

        B = jnp.linalg.solve(make_psd(stage1_ridge_weights), K_ZAX_ZAXTilde)
        self.B = B
        stage2_ridge_weights = K_ATildeATilde * (B.T @ K_WW @ B) * K_XTildeXTilde

        x_mean_vec = jnp.mean(K_XXTilde, axis=0)[:, jnp.newaxis]
        if optimize_lambda2_parameter:
            K_YTilde = YTilde @ YTilde.T
            lambda2_list = jnp.logspace(jnp.log(lambda2_optimization_range[0]), jnp.log(lambda2_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda2_objective_list = jnp.array([self._lambda2_objective(lambda_, stage2_ridge_weights, K_YTilde, 
                                                                            I_m, make_psd_eps) for lambda_ in lambda2_list])
            lambda2_ = lambda2_list[jnp.argmin(lambda2_objective_list).item()]
            self.lambda2_ = lambda2_

        if optimize_zeta_parameter:
            K_XX_WW_Tilde = K_XTildeXTilde * K_WTildeWTilde
            zeta_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta_objective_list = jnp.array([self.zeta_objective(zeta, K_ATildeATilde, K_XX_WW_Tilde) for zeta in zeta_list])
            zeta = zeta_list[jnp.argmin(zeta_objective_list).item()]
            self.zeta = zeta

        stage2_ridge_weights += stage2_data_size * lambda2_ * I_m
        alpha = jnp.linalg.solve(make_psd(stage2_ridge_weights), YTilde)

        self.alpha = alpha
        self.ATilde = A[stage2_idx]
        self.W = W[stage1_idx]
        self.K_WWTilde = K_WWTilde
        self.K_ATildeATilde = K_ATildeATilde
        self.K_XTildeXTilde = K_XTildeXTilde
        if X is not None:
            self.XTilde = X[stage2_idx]
        else:
            self.XTilde = None

    def predict(self, A: jnp.ndarray, aprime: jnp.array) -> jnp.ndarray:
        if A.ndim != 2:
            A_test = A.reshape(-1, 1)
        else:
            A_test = A
        K_ATildeaprime = self.kernel_A(self.ATilde, aprime)
        m = K_ATildeaprime.shape[0]
        CME_weights_ = jnp.linalg.solve(self.K_ATildeATilde + m * self.zeta * jnp.eye(m), K_ATildeaprime) # Conditinal Mean Embedding Weights
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)

        # pred = self.alpha.T @ (K_ATildeATest[::, jj] * ((self.K_XXTilde.T @ (self.B.T @ self.K_WWTilde)) @ CME_weights_))
        pred = jnp.array([self._predict_structural_function(self.alpha, 
                                                            self.B, 
                                                            K_ATildeATest[:, jj].reshape(-1, 1), 
                                                            self.K_XTildeXTilde, 
                                                            self.K_WWTilde, 
                                                            CME_weights_).item() for jj in range(K_ATildeATest.shape[1])])
        return pred
    
    def _predict_structural_function(self, 
                                     alpha: jnp.array,
                                     B: jnp.array,
                                     KATilde_a: jnp.array,
                                     K_XTildeXTilde: jnp.array,
                                     K_WWTilde: jnp.array,
                                     CME_weights_:jnp.array
                                     ):
        return alpha.T @ (KATilde_a * ((K_XTildeXTilde.T * (B.T @ K_WWTilde)) @ CME_weights_))

    def _predict_bridge_func(self, A_test : jnp.ndarray, W_test : jnp.ndarray, X_test : jnp.array = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_WWTest = self.kernel_W(self.W, W_test)
        if (X_test is not None) & (self.XTilde is not None):
            K_XTildeXTest = self.kernel_X(self.XTilde, X_test)
        else:
            K_XTildeXTest = jnp.ones((self.ATilde.shape[0], W_test.shape[0]))

        bridge_pred = jnp.array([(K_ATildeATest[:, jj].reshape(-1, 1) * ((self.B.T @ K_WWTest) * K_XTildeXTest)).T @ self.alpha for jj in range(A_test.shape[0])])
        return bridge_pred[:, :, 0]
    

class KernelProxyVariableCATE(BaseEstimator, RegressorMixin):

    def __init__(self, 
                 kernel_A : Kernel,
                 kernel_W : Kernel,
                 kernel_Z : Kernel,
                 kernel_V : Kernel,
                 kernel_X : Kernel = RBF(),
                 lambda1_ : float = 0.1,
                 lambda2_ : float = 0.1,
                 zeta: float = 0.1,
                 optimize_lambda1_parameter : bool = True,
                 optimize_lambda2_parameter : bool = True,
                 optimize_zeta_parameter : bool = True,
                 lambda1_optimization_range : Tuple[float, float] = (1e-5, 1.),
                 lambda2_optimization_range : Tuple[float, float] = (1e-5, 1.),
                 zeta_optimization_range : Tuple[float, float] = (1e-5, 1.),
                 **kwargs) -> None:
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_V_params = kwargs.pop('kernel_V_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Kernel)):
            raise Exception("Kernel for A must be callable Kernel class")
        if (not isinstance(kernel_W, Kernel)):
            raise Exception("Kernel for W must be callable Kernel class")
        if (not isinstance(kernel_Z, Kernel)):
            raise Exception("Kernel for Z must be callable Kernel class")
        if (not isinstance(kernel_V, Kernel)):
            raise Exception("Kernel for V must be callable Kernel class")
        if (not isinstance(kernel_X, Kernel)):
            raise Exception("Kernel for X must be callable Kernel class")
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_V = kernel_V
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_V_params is not None:
            self.kernel_V.set_params(**kernel_V_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda1_, self.lambda2_, self.zeta = lambda1_, lambda2_, zeta
        self.optimize_lambda1_parameter = optimize_lambda1_parameter
        self.optimize_lambda2_parameter = optimize_lambda2_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda1_optimization_range = lambda1_optimization_range
        self.lambda2_optimization_range = lambda2_optimization_range
        self.zeta_optimization_range = zeta_optimization_range
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps 

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################        
    @staticmethod
    @jit
    def _lambda1_objective(lambda_: float, 
                           K_ZAX: jnp.ndarray, 
                           K_WW: jnp.ndarray, 
                           identity_matrix: jnp.ndarray,
                           make_psd_eps: float = 1e-9) -> float:
        n = K_ZAX.shape[0]
        ridge_weights = make_psd(K_ZAX + n * lambda_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_ZAX).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_WW @ H1_tilde_inv_times_H1) 
        return objective
    
    @staticmethod
    @jit
    def _lambda2_objective(lambda_: float, 
                            second_stage_ridge_weights: jnp.ndarray, 
                            K_YTilde: jnp.ndarray, 
                            identity_matrix: jnp.ndarray,
                            make_psd_eps: float = 1e-9) -> float:
        n = second_stage_ridge_weights.shape[0]
        ridge_weights = make_psd(second_stage_ridge_weights + n * lambda_ * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, second_stage_ridge_weights).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_YTilde @ H1_tilde_inv_times_H1) 
        return objective

    @staticmethod
    @jit
    def _zeta_objective(zeta: float, K_VV: jnp.ndarray, K_XX_WW: jnp.ndarray) -> float:
        """
        Objective function for zeta optimization.

        Parameters:
        - zeta: float, regularization parameter.
        - K_AA: jnp.ndarray, kernel matrix for variable A.
        - K_XX_WW: jnp.ndarray, combined kernel matrix for variables X and W.

        Returns:
        - float, objective value.
        """
        n = K_VV.shape[0]
        # R = K_VV @ jnp.linalg.inv(K_VV + n * lambda2 * jnp.eye(n))
        R = jnp.linalg.solve(make_psd(K_VV + n * zeta * jnp.eye(n)), K_VV).T
        S = jnp.diag((1 / (1 - jnp.diag(R))) ** 2)
        T = S @ (K_XX_WW - 2 * K_XX_WW @ R.T + R @ K_XX_WW @ R.T)
        zeta_cost = jnp.trace(T)
        return zeta_cost

    @staticmethod
    @jit
    def _predict_structural_function(alpha: jnp.ndarray, 
                                     B: jnp.ndarray, 
                                     K_ATildea: jnp.ndarray, 
                                     K_VTildev: jnp.ndarray,
                                     K_XTildeXTilde: jnp.ndarray, 
                                     K_WWTilde: jnp.ndarray, 
                                     CME_weights_: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the structural function.

        Parameters:
        - alpha: jnp.ndarray, alpha values.
        - B: jnp.ndarray, matrix B.
        - K_ATildea: jnp.ndarray, kernel matrix for A tilde and a.
        - K_XTildeXTilde: jnp.ndarray, kernel matrix for X tilde.
        - K_WW: jnp.ndarray, kernel matrix for W.
        - CME_weights_: jnp.ndarray, conditional mean embedding weights.

        Returns:
        - jnp.ndarray, predicted structural function.
        """
        # n = K_Ww.shape[0]
        pred = alpha.T @ ((K_ATildea * K_VTildev).reshape(-1, 1) * ((K_XTildeXTilde * (B.T @ K_WWTilde)) @ CME_weights_))
        return pred

    def fit(self,             
            AWZVX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray,) -> None:
        kernel_A, kernel_W, kernel_Z, kernel_V, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_V, self.kernel_X
        lambda1_, lambda2_, zeta = self.lambda1_, self.lambda2_, self.zeta
        optimize_lambda1_parameter = self.optimize_lambda1_parameter
        optimize_lambda2_parameter = self.optimize_lambda2_parameter
        optimize_zeta_parameter = self.optimize_zeta_parameter
        lambda1_optimization_range = self.lambda1_optimization_range
        lambda2_optimization_range = self.lambda2_optimization_range
        zeta_optimization_range = self.zeta_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps

        if len(AWZVX) == 5:
            A, W, Z, V, X = AWZVX
        elif len(AWZVX) == 4:
            A, W, Z, V = AWZVX
            X = None
        
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_ZTrainZTrain = kernel_Z(Z, Z)
        K_VTrainVTrain = kernel_V(V, V)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = make_psd(kernel_X(X, X), make_psd_eps)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_V = kernel_V
        self.kernel_X = kernel_X

        ############################# SPLIT DATA IN STAGE 1 AND STAGE 2 #####################################
        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        ################################ KERNEL MATRICES ####################################################
        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_VV = K_VTrainVTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_VVTilde = K_VTrainVTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_VTildeV = K_VVTilde.T
        K_VTildeVTilde = K_VTrainVTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_WTildeWTilde = K_WTrainWTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_ZZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_XTildeXTilde = K_XTrainXTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_V, self.kernel_X]:
            if hasattr(kernel_, 'use_length_scale_heuristic'):
                kernel_.use_length_scale_heuristic = False
                
        ########### FIRST STAGE REGRESSION ###########################
        K_ZAVX = K_ZZ * K_AA * K_VV * K_XX
        I_n = jnp.eye(stage1_data_size)
        I_m = jnp.eye(stage2_data_size)
        YTilde = Y[stage2_idx]

        if optimize_lambda1_parameter:
            lambda1_list = jnp.logspace(jnp.log(lambda1_optimization_range[0]), jnp.log(lambda1_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda1_objective_list = jnp.array([self._lambda1_objective(lambda_, K_ZAVX, K_WW, 
                                                                            I_n, make_psd_eps) for lambda_ in lambda1_list])
            lambda1_ = lambda1_list[jnp.argmin(lambda1_objective_list).item()]
            self.lambda1_ = lambda1_
        ########### SECOND STAGE REGRESSION ###########################
        stage1_ridge_weights = (K_ZAVX + stage1_data_size * lambda1_ * I_n)
        K_ZAVX_ZAVXTilde = K_ZZTilde * K_AATilde * K_VVTilde * K_XXTilde
        self.stage1_ridge_weights = stage1_ridge_weights
        B = jnp.linalg.solve(make_psd(stage1_ridge_weights), K_ZAVX_ZAVXTilde)
        self.B = B
        stage2_ridge_weights = K_ATildeATilde * (B.T @ K_WW @ B) * K_VTildeVTilde * K_XTildeXTilde

        if optimize_lambda2_parameter:
            K_YTilde = YTilde @ YTilde.T
            lambda2_list = jnp.logspace(jnp.log(lambda2_optimization_range[0]), jnp.log(lambda2_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda2_objective_list = jnp.array([self._lambda2_objective(lambda_, stage2_ridge_weights, K_YTilde, 
                                                                            I_m, make_psd_eps) for lambda_ in lambda2_list])
            lambda2_ = lambda2_list[jnp.argmin(lambda2_objective_list).item()]
            self.lambda2_ = lambda2_

        stage2_ridge_weights += stage2_data_size * lambda2_ * I_m
        alpha = jnp.linalg.solve(make_psd(stage2_ridge_weights), YTilde)

        if optimize_zeta_parameter:
            K_XXWW_Tilde = K_XTildeXTilde * K_WTildeWTilde
            zeta_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta_objective_list = jnp.array([self._zeta_objective(zeta, K_VTildeVTilde, K_XXWW_Tilde) for zeta in zeta_list])
            zeta = zeta_list[jnp.argmin(zeta_objective_list).item()]
            self.zeta = zeta

        self.alpha = alpha
        self.ATilde, self.VTilde = A[stage2_idx], V[stage2_idx]
        self.K_XTildeXTilde = K_XTildeXTilde
        self.K_WWTilde = K_WWTilde
        self.K_VTildeVTilde = K_VTildeVTilde
        if X is not None:
            self.XTilde = X[stage2_idx]
        else:
            self.XTilde = None
        self.W = W[stage1_idx]

    def predict(self, A: jnp.ndarray, V: jnp.array) -> jnp.ndarray:
        if A.ndim != 2:
            A_test = A.reshape(-1, 1)
        else:
            A_test = A
        V_test = V
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_VTildeVTest = self.kernel_V(self.VTilde, V_test)
        m = K_ATildeATest.shape[0]
        # CME_weights_ = jnp.linalg.inv(make_psd(self.K_VTildeVTilde) + m * self.zeta * jnp.eye(m)) @ K_VTildeVTest # Conditinal Mean Embedding Weights
        CME_weights_ = jnp.linalg.solve(make_psd(self.K_VTildeVTilde + m * self.zeta * jnp.eye(m)), K_VTildeVTest)
        # self.CME_weights_ = CME_weights_
        pred = jnp.array([self._predict_structural_function(self.alpha, 
                                                            self.B, 
                                                            K_ATildeATest[:, jj].reshape(-1, 1), 
                                                            K_VTildeVTest[:, jj].reshape(-1, 1), 
                                                            self.K_XTildeXTilde, 
                                                            self.K_WWTilde, 
                                                            CME_weights_[:, jj].reshape(-1, 1)).item() for jj in range(K_ATildeATest.shape[1])])
        return pred
    
    def _predict_bridge_func(self, A_test : jnp.ndarray, W_test : jnp.ndarray, V_test: jnp.array, X_test : jnp.array = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_VTildeVTest = self.kernel_V(self.VTilde, V_test)
        K_WWTest = self.kernel_W(self.W, W_test)
        if (X_test is not None) & (self.XTilde is not None):
            K_XTildeXTest = self.kernel_X(self.XTilde, X_test)
        else:
            K_XTildeXTest = jnp.ones((self.ATilde.shape[0], W_test.shape[0]))
        
        # bridge_pred = (K_ATildeATest * (self.B.T @ K_WWTest)).T @ self.alpha 
        bridge_pred = jnp.array([((K_ATildeATest[:, jj] * K_VTildeVTest[:, jj]).reshape(-1, 1) * ((self.B.T @ K_WWTest) * (K_XTildeXTest))).T @ self.alpha for jj in range(A_test.shape[0])])
        return bridge_pred[:, :, 0]


class ProxyMaximumMomentRestrictionATE(BaseEstimator, RegressorMixin):

    def __init__(self,
                 kernel_A : Kernel,
                 kernel_W : Kernel,
                 kernel_Z : Kernel,
                 kernel_X : Kernel = RBF(),
                 lambda_ : float = 0.1,
                 optimize_lambda_parameter: bool = False,
                 validation_percentage: float = 0.1,
                 lambda_optimization_range : Union[List, Tuple[float, float]] = (1e-5, 1.),
                 regularization_grid_points = 15, 
                 **kwargs) -> None:
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Kernel)):
            raise Exception("Kernel for A must be callable Kernel class")
        if (not isinstance(kernel_W, Kernel)):
            raise Exception("Kernel for W must be callable Kernel class")
        if (not isinstance(kernel_Z, Kernel)):
            raise Exception("Kernel for Z must be callable Kernel class")
        if (not isinstance(kernel_X, Kernel)):
            raise Exception("Kernel for X must be callable Kernel class")
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda_ = lambda_
        self.optimize_lambda_parameter = optimize_lambda_parameter
        self.validation_percentage = validation_percentage
        self.lambda_optimization_range = lambda_optimization_range
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps

    @staticmethod
    # @jit
    def _lambda_objective(lambda_, Gsq, L, identity_mat, Y_train, K_ATrainAVal, K_WTrainWVal, K_XTrainXVal, Y_val, make_psd_eps):
        train_data_size = identity_mat.shape[0]
        alpha = Gsq @ jnp.linalg.solve(make_psd(Gsq @ L @ Gsq + lambda_ * train_data_size * identity_mat, make_psd_eps),
                                                Gsq @ Y_train)
        bridge_pred = alpha.T @ (K_ATrainAVal * K_WTrainWVal * K_XTrainXVal)
        cost = jnp.mean((bridge_pred.reshape(-1, 1) - Y_val.reshape(-1, 1)) ** 2)
        return cost.item()

    def fit(self,             
            AWZX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray,) -> None:
        kernel_A, kernel_W, kernel_Z, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X
        lambda_ = self.lambda_
        validation_percentage = self.validation_percentage
        make_psd_eps = self.make_psd_eps

        if len(AWZX) == 4:
            A, W, Z, X = AWZX
        elif len(AWZX) == 3:
            A, W, Z = AWZX
            X = None
        data_size = A.shape[0]
        train_indices = np.random.permutation(data_size)
        validation_size = int(data_size * (validation_percentage)) + 1
        ATrain, WTrain, ZTrain, YTrain = A[train_indices[validation_size:]], W[train_indices[validation_size:]], Z[train_indices[validation_size:]], Y[train_indices[validation_size:]]
        A_val, W_val, Y_val = A[train_indices[:validation_size]], W[train_indices[:validation_size]], Y[train_indices[:validation_size]]
        train_data_size = ATrain.shape[0]
        if X is not None:
            XTrain = X[train_indices[validation_size:]]
            X_val = X[train_indices[:validation_size]]

        K_ATrainATrain = kernel_A(ATrain, ATrain)
        K_WTrainWTrain = kernel_W(WTrain, WTrain)
        K_ZTrainZTrain = kernel_Z(ZTrain, ZTrain)
        if X is None:
            K_XTrainXTrain = jnp.ones((WTrain.shape[0], WTrain.shape[0]))
        else:
            K_XTrainXTrain = kernel_X(XTrain, XTrain)

        for kernel_ in [self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X]:
            if hasattr(kernel_, 'use_length_scale_heuristic'):
                kernel_.use_length_scale_heuristic = False

        self.x_mean_vec = jnp.mean(K_XTrainXTrain, axis=0)[:, jnp.newaxis]
        self.w_mean_vec = jnp.mean(K_WTrainWTrain, axis=0)[:, jnp.newaxis]

        G = K_ATrainATrain * K_ZTrainZTrain * K_XTrainXTrain
        L = K_ATrainATrain * K_WTrainWTrain * K_XTrainXTrain
        D, V = jnp.linalg.eigh(G) 
        Gsq = V @ jnp.diag(jnp.sqrt(jnp.maximum(D, 0.0))) @ V.T

        Identity_mat = jnp.eye(train_data_size)
        if self.optimize_lambda_parameter:
            K_ATrainAVal = kernel_A(ATrain, A_val)
            K_WTrainWVal = kernel_W(WTrain, W_val)
            if X is None:
                K_XTrainXVal = jnp.ones_like(K_WTrainWVal)
            else:
                K_XTrainXVal = kernel_X(XTrain, X_val)

            if isinstance(self.lambda_optimization_range, tuple):
                lambda_list = jnp.logspace(jnp.log(self.lambda_optimization_range[0]), jnp.log(self.lambda_optimization_range[1]), self.regularization_grid_points, base = jnp.exp(1))
            else:
                lambda_list = self.lambda_optimization_range

            lambda_objective_list = jnp.array([self._lambda_objective(lambda_, Gsq, L, Identity_mat, YTrain, K_ATrainAVal, 
                                                                      K_WTrainWVal, K_XTrainXVal, Y_val, make_psd_eps) for lambda_ in lambda_list])
            self.lambda_objective_list = lambda_objective_list
            lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]

        alpha = Gsq @ jnp.linalg.solve(make_psd(Gsq @ L @ Gsq + lambda_ * train_data_size * Identity_mat, make_psd_eps),
                                                Gsq @ YTrain)
        
        self.alpha = alpha
        self.lambda_ = lambda_ 
        self.ATrain, self.WTrain = ATrain, WTrain
        if X is not None:
            self.XTrain = X
        else:
            self.XTrain = None

    def predict(self, A: jnp.ndarray) -> jnp.ndarray:
        if A.ndim != 2:
            A_test = A.reshape(-1, 1)
        else:
            A_test = A
        K_ATrainATest = self.kernel_A(self.ATrain, A_test)

        pred = self.alpha.T @ (K_ATrainATest * self.w_mean_vec * self.x_mean_vec)
        return pred

    def _predict_bridge_func(self, A_test : jnp.ndarray, W_test : jnp.ndarray, X_test : jnp.array = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ATrainATest = self.kernel_A(self.ATrain, A_test)
        K_WTrainWTest = self.kernel_W(self.WTrain, W_test)
        if (X_test is not None) & (self.XTrain is not None):
            K_XTrainXTest = self.kernel_X(self.XTrain, X_test)
        else:
            K_XTrainXTest = jnp.ones((self.ATrain.shape[0], W_test.shape[0]))
        # bridge_pred = self.alpha.T @ (K_ATrainATest * K_WTrainWTest)
        bridge_pred = jnp.array([self.alpha.T @ (K_ATrainATest[:, jj].reshape(-1, 1) * K_WTrainWTest * K_XTrainXTest) for jj in range(A_test.shape[0])])
        return bridge_pred[:, 0, :]
    

class KernelNegativeControlATE(BaseEstimator, RegressorMixin):

    """
    A class to estimate the Average Treatment Effect (ATE) using kernel methods 
    with negative control variables.

    Attributes:
    -----------
    kernel_A : Callable
        Kernel function for the treatment variable A.
    kernel_W : Callable
        Kernel function for the negative control variable W.
    kernel_Z : Callable
        Kernel function for the negative control variable Z.
    kernel_X : Callable
        Kernel function for the covariates X. Default is RBF().
    lambda_ : float
        Regularization parameter for the kernel ridge regression. Default is 0.1.
    zeta : float
        Regularization parameter for the second stage regression. Default is 0.1.
    optimize_regularization_parameters : bool
        Whether to optimize the regularization parameters. Default is True.
    lambda_optimization_range : Tuple[float, float]
        Range for optimizing lambda. Default is (1e-9, 1.0).
    zeta_optimization_range : Tuple[float, float]
        Range for optimizing zeta. Default is (1e-9, 1.0).
    kwargs : dict
        Additional keyword arguments for customization.
    """
    def __init__(self, 
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_X: Callable = RBF(),
                 lambda_: float = 0.1,
                 zeta: float = 0.1, 
                 optimize_regularization_parameters: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 **kwargs) -> None:
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        large_lambda_zeta_option = kwargs.pop('large_lambda_zeta_option', False)
        selecting_biggest_lambda_tol = kwargs.pop('selecting_biggest_lambda_tol', 1e-9)
        selecting_biggest_zeta_tol = kwargs.pop('selecting_biggest_zeta_tol', 1e-9)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Callable)):
            raise Exception("Kernel for A must be callable")
        if (not isinstance(kernel_W, Callable)):
            raise Exception("Kernel for W must be callable")
        if (not isinstance(kernel_Z, Callable)):
            raise Exception("Kernel for Z must be callable")
        if (not isinstance(kernel_X, Callable)):
            raise Exception("Kernel for X must be callable")
        
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda_, self.zeta = lambda_, zeta
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.lambda_optimization_range = lambda_optimization_range
        self.zeta_optimization_range = zeta_optimization_range
        self.large_lambda_zeta_option = large_lambda_zeta_option
        self.selecting_biggest_lambda_tol = selecting_biggest_lambda_tol
        self.selecting_biggest_zeta_tol = selecting_biggest_zeta_tol
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps 

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################
    @staticmethod
    @jit 
    def _lambda_objective(lambda_: float, 
                          K_WW: jnp.ndarray, 
                          K_ZZ: jnp.ndarray, 
                          K_AA: jnp.ndarray, 
                          K_XX: jnp.ndarray,
                          make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for lambda optimization.

        Parameters:
        - lambda_: float, lambda.
        - K_WW: jnp.ndarray, kernel matrix for variable W.
        - K_ZZ: jnp.ndarray, kernel matrix for variable Z.
        - K_AA: jnp.ndarray, kernel matrix for variable A.
        - K_XX: jnp.ndarray, kernel matrix for variable X.
        - make_psd_eps: float, epsilon value for making matrices positive semi-definite.

        Returns:
        - objective: float, objective value.
        """
        n = K_AA.shape[0]
        identity_matrix = jnp.eye(n)
        H1 = identity_matrix - make_psd(K_AA * K_ZZ * K_XX) @ jnp.linalg.inv(make_psd(K_AA * K_ZZ * K_XX + n * lambda_ * identity_matrix, eps = make_psd_eps))
        H1_tilde_inv = jnp.diag(1 / jnp.diag(H1))
        objective = (1 / n) * jnp.trace(H1_tilde_inv @ H1 @ K_WW @ H1 @ H1_tilde_inv)
        return objective

    @staticmethod
    @jit 
    def _zeta_objective(zeta: float,
                        pred_matrix_for_Y_stage_1: jnp.ndarray,
                        M: jnp.ndarray,
                        Y_train_stage1: jnp.ndarray,
                        Y_train_stage2: jnp.ndarray) -> float:
        """
        Objective function for zeta optimization.

        Parameters:
        - zeta: float, regularization parameter.
        - pred_matrix_for_Y_stage_1: jnp.ndarray, prediction matrix for stage 1.
        - M: jnp.ndarray, matrix for second stage regression.
        - Y_train_stage1: jnp.ndarray, stage 1 training labels.
        - Y_train_stage2: jnp.ndarray, stage 2 training labels.

        Returns:
        - mse_loss: float, mean squared error loss.
        """
        # D_transpose_K_WW_D = (D.T @ K_WW @ D)
        # pred_matrix_for_Y_stage_1 = (K_ATildeA * K_XTildeX * D_transpose_K_WW_D) = (K_ATildeA * K_XTildeX * (D.T @ K_WW @ D))
        # alpha = jnp.linalg.inv(M @ M.T + m * zeta * M) @ M @ Y_train_stage2
        
        m = Y_train_stage2.shape[0]
        alpha = jnp.linalg.solve(make_psd(M @ M.T) + m * zeta * make_psd(M), make_psd(M) @ Y_train_stage2)
        Y_pred = pred_matrix_for_Y_stage_1.T @ alpha
        mse_loss = jnp.mean((Y_train_stage1.reshape(-1, 1) - Y_pred.reshape(-1, 1)) ** 2)
        return mse_loss

    @staticmethod
    @jit
    def _predict_structural_function(alpha: jnp.ndarray, 
                                     D: jnp.ndarray, 
                                     K_ATildea: jnp.ndarray, 
                                     K_XTildeX: jnp.ndarray, 
                                     K_WW: jnp.ndarray) -> float:
        """
        Predict the structural function.

        Parameters:
        - alpha: jnp.ndarray, coefficients from the second stage regression.
        - D: jnp.ndarray, matrix D from the first stage regression.
        - K_ATildea: jnp.ndarray, kernel matrix for A at new points.
        - K_XTildeX: jnp.ndarray, kernel matrix for X at new points.
        - K_WW: jnp.ndarray, kernel matrix for W.

        Returns:
        - pred: float, predicted structural function value.
        """
        # n = K_Ww.shape[0]
        pred = alpha.T @ (K_ATildea.reshape(-1, 1) * (K_XTildeX * (D.T @ K_WW)).mean(axis = 1, keepdims = True))
        return pred

    def fit(self, AWZX: Tuple[jnp.ndarray, ...], Y: jnp.ndarray) -> None:
        """
        Fit the model to the data.

        Parameters:
        - AWZX: Tuple[jnp.ndarray, ...], tuple containing arrays for A, W, Z, and optionally X.
        - Y: jnp.ndarray, array of outcome variable.
        """
        kernel_A, kernel_W, kernel_Z, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X
        lambda_, zeta = self.lambda_, self.zeta
        optimize_regularization_parameters = self.optimize_regularization_parameters
        lambda_optimization_range = self.lambda_optimization_range
        zeta_optimization_range = self.zeta_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps
        large_lambda_zeta_option = self.large_lambda_zeta_option
        selecting_biggest_lambda_tol = self.selecting_biggest_lambda_tol
        selecting_biggest_zeta_tol = self.selecting_biggest_zeta_tol

        if len(AWZX) == 4:
            A, W, Z, X = AWZX
        elif len(AWZX) == 3:
            A, W, Z = AWZX
            X = None
        
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = kernel_X(X, X)
        K_ZTrainZTrain = kernel_Z(Z, Z)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        # K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_WTildeW = K_WWTilde.T
        # K_WTildeWTilde = K_WTrainWTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_ZZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_ZTildeZ = K_ZZTilde.T
        # K_ZTildeZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_XTildeX = K_XXTilde.T
        K_XTildeXTilde = K_XTrainXTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        if hasattr(self.kernel_A, 'use_length_scale_heuristic'):
            self.kernel_A.use_length_scale_heuristic = False

        if hasattr(self.kernel_W, 'use_length_scale_heuristic'):
            self.kernel_W.use_length_scale_heuristic = False

        if hasattr(self.kernel_Z, 'use_length_scale_heuristic'):
            self.kernel_Z.use_length_scale_heuristic = False

        if hasattr(self.kernel_X, 'use_length_scale_heuristic'):
            self.kernel_X.use_length_scale_heuristic = False

        ########## OPTIMIZE THE REGULARIZATION PARAMETERS IF IT IS SPECIFIED ###########################
        if optimize_regularization_parameters:
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda_objective_list = jnp.array([self._lambda_objective(lambda_, K_WW, K_ZZ, K_AA, K_XX, make_psd_eps) for lambda_ in lambda_list])
            lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]
            if large_lambda_zeta_option:
                lambda_objective_min = jnp.min(lambda_objective_list).item()
                _lambda_objective_list_minimizer_list = jnp.where(jnp.abs(lambda_objective_list - lambda_objective_min) <= selecting_biggest_lambda_tol)[0]
                lambda_ = jnp.max(lambda_list[_lambda_objective_list_minimizer_list])
            else:
                lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]

            self.lambda_ = lambda_ # For debugging purposes.

        C = K_AA * K_XX * K_ZZ 
        CTilde = K_AATilde * K_XXTilde * K_ZZTilde
        D = jnp.linalg.inv(C + stage1_data_size * lambda_ * jnp.eye(stage1_data_size)) @ CTilde
        M = K_ATildeATilde * K_XTildeXTilde * (D.T @ K_WW @ D)

        if optimize_regularization_parameters:
            pred_matrix_for_Y_stage_1 = (K_ATildeA * K_XTildeX * (D.T @ K_WW))
            zeta_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta_objective_list = jnp.array([self._zeta_objective(zeta_, pred_matrix_for_Y_stage_1, M, Y[stage1_idx], Y[stage2_idx]) for zeta_ in zeta_list])
            if large_lambda_zeta_option:
                zeta_objective_min = jnp.min(zeta_objective_list).item()
                zeta_objective_list_minimizer_list = jnp.where(jnp.abs(zeta_objective_list - zeta_objective_min) <= selecting_biggest_zeta_tol)[0]
                zeta = jnp.max(zeta_list[zeta_objective_list_minimizer_list])
            else:
                zeta = zeta_list[jnp.argmin(zeta_objective_list).item()]
            self.zeta = zeta
            self.zeta_objective_list = zeta_objective_list

        # alpha = jnp.linalg.inv(M @ M.T + stage2_data_size * zeta * M) @ M @ Y[stage2_idx]
        alpha = jnp.linalg.solve(make_psd(M @ M.T) + stage2_data_size * zeta * make_psd(M), M @ Y[stage2_idx])
        self.alpha = alpha
        self.ATilde, self.W = A[stage2_idx], W[stage1_idx]
        if X is not None: 
            self.XTilde = X[stage2_idx]
        self.stage2_idx = stage2_idx
        self.K_XTildeX = K_XXTilde.T
        self.K_WW = K_WW
        self.D = D

    def predict(self, A: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the structural function values for new data.

        Parameters:
        - A: jnp.ndarray, new data for the treatment variable A.

        Returns:
        - f_struct_pred: jnp.ndarray, predicted structural function values.
        """
        alpha = self.alpha
        K_XTildeX = self.K_XTildeX
        K_WW = self.K_WW
        D = self.D
        if A.ndim == 2:
            A_test = A
        else:
            A_test = A.reshape(-1, 1)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        f_struct_pred = np.array([self._predict_structural_function(alpha, D, K_ATildeATest[:, jj], K_XTildeX, K_WW).item() for jj in range(K_ATildeATest.shape[1])])
        return f_struct_pred

    def _predict_bridge_func(self, A_test : jnp.array, W_test : jnp.array, X_test : Optional[jnp.array] = None):
        if A_test.ndim != 2:
            A_test = A_test.reshape(-1, 1)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        if X_test is not None:
            K_XTildeX = self.kernel_X(self.XTilde, X_test)
        else:
            K_XTildeX = jnp.ones((self.ATilde.shape[0], W_test.shape[0]))
        K_WWTest = self.kernel_W(self.W, W_test)
        alpha = self.alpha

        bridge_function = jnp.array([alpha.T @ (K_ATildeATest[:, jj].reshape(-1, 1) * (self.D.T @ K_WWTest)) for jj in range(K_ATildeATest.shape[1])])[:, 0, :]
        return bridge_function
    

class KernelNegativeControlATT(BaseEstimator, RegressorMixin):

    def __init__(self, 
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_X: Callable = RBF(),
                 lambda_: float = 0.1,
                 zeta: float = 0.1, 
                 lambda2: float = 1e-3,
                 optimize_regularization_parameters: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 **kwargs: Any) -> None:
        """
        Initialize the KernelNegativeControlATT model.

        Parameters:
        - kernel_A: Callable, kernel function for variable A.
        - kernel_W: Callable, kernel function for variable W.
        - kernel_Z: Callable, kernel function for variable Z.
        - kernel_X: Callable, kernel function for variable X (default: RBF()).
        - lambda_: float, regularization parameter for lambda (default: 0.1).
        - zeta: float, regularization parameter for zeta (default: 0.1).
        - lambda2: float, regularization parameter for lambda2 (default: 1e-3).
        - optimize_regularization_parameters: bool, whether to optimize regularization parameters (default: True).
        - lambda_optimization_range: Tuple[float, float], range for optimizing lambda (default: (1e-9, 1.0)).
        - zeta_optimization_range: Tuple[float, float], range for optimizing zeta (default: (1e-9, 1.0)).
        - kwargs: Additional keyword arguments.
        """
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        large_lambda_zeta_option = kwargs.pop('large_lambda_zeta_option', False)
        selecting_biggest_lambda_tol = kwargs.pop('selecting_biggest_lambda_tol', 1e-9)
        selecting_biggest_zeta_tol = kwargs.pop('selecting_biggest_zeta_tol', 1e-9)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Callable)):
            raise Exception("Kernel for A must be callable")
        if (not isinstance(kernel_W, Callable)):
            raise Exception("Kernel for W must be callable")
        if (not isinstance(kernel_Z, Callable)):
            raise Exception("Kernel for Z must be callable")
        if (not isinstance(kernel_X, Callable)):
            raise Exception("Kernel for X must be callable")
        
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda_, self.zeta, self.lambda2 = lambda_, zeta, lambda2
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.lambda_optimization_range = lambda_optimization_range
        self.zeta_optimization_range = zeta_optimization_range
        self.large_lambda_zeta_option = large_lambda_zeta_option
        self.selecting_biggest_lambda_tol = selecting_biggest_lambda_tol
        self.selecting_biggest_zeta_tol = selecting_biggest_zeta_tol
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps 

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################
    @staticmethod
    @jit 
    def _lambda_objective(lambda_: float, 
                          K_WW: jnp.ndarray, 
                          K_ZZ: jnp.ndarray, 
                          K_AA: jnp.ndarray, 
                          K_XX: jnp.ndarray,
                          make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for lambda optimization.

        Parameters:
        - lambda_: float, regularization parameter.
        - K_WW: jnp.ndarray, kernel matrix for variable W.
        - K_ZZ: jnp.ndarray, kernel matrix for variable Z.
        - K_AA: jnp.ndarray, kernel matrix for variable A.
        - K_XX: jnp.ndarray, kernel matrix for variable X.
        - make_psd_eps: float, epsilon value for making matrices positive semi-definite.

        Returns:
        - float, objective value.
        """
        n = K_AA.shape[0]
        identity_matrix = jnp.eye(n)
        H1 = identity_matrix - make_psd(K_AA * K_ZZ * K_XX) @ jnp.linalg.inv(make_psd(K_AA * K_ZZ * K_XX + n * lambda_ * identity_matrix, eps = make_psd_eps))
        H1_tilde_inv = jnp.diag(1 / jnp.diag(H1))
        objective = (1 / n) * jnp.trace(H1_tilde_inv @ H1 @ K_WW @ H1 @ H1_tilde_inv)
        return objective

    @staticmethod
    @jit 
    def _zeta_objective(zeta: float,
                        pred_matrix_for_Y_stage_1: jnp.ndarray,
                        M: jnp.ndarray,
                        Y_train_stage1: jnp.ndarray,
                        Y_train_stage2: jnp.ndarray) -> float:
        """
        Objective function for zeta optimization.

        Parameters:
        - zeta: float, regularization parameter.
        - pred_matrix_for_Y_stage_1: jnp.ndarray, prediction matrix for stage 1.
        - M: jnp.ndarray, matrix M.
        - Y_train_stage1: jnp.ndarray, training data for stage 1.
        - Y_train_stage2: jnp.ndarray, training data for stage 2.

        Returns:
        - float, mean squared error loss.
        """
        # D_transpose_K_WW_D = (D.T @ K_WW @ D)
        # pred_matrix_for_Y_stage_1 = (K_ATildeA * K_XTildeX * D_transpose_K_WW_D) = (K_ATildeA * K_XTildeX * (D.T @ K_WW @ D))
        # alpha = jnp.linalg.inv(M @ M.T + m * zeta * M) @ M @ Y_train_stage2

        m = Y_train_stage2.shape[0]
        alpha = jnp.linalg.solve(make_psd(M @ M.T) + m * zeta * make_psd(M), make_psd(M) @ Y_train_stage2)
        Y_pred = pred_matrix_for_Y_stage_1.T @ alpha
        mse_loss = jnp.mean((Y_train_stage1.reshape(-1, 1) - Y_pred.reshape(-1, 1)) ** 2)
        return mse_loss

    @staticmethod
    @jit
    def _lambda2_objective(lambda2: float, K_AA: jnp.ndarray, K_XX_WW: jnp.ndarray) -> float:
        """
        Objective function for lambda2 optimization.

        Parameters:
        - lambda2: float, regularization parameter.
        - K_AA: jnp.ndarray, kernel matrix for variable A.
        - K_XX_WW: jnp.ndarray, combined kernel matrix for variables X and W.

        Returns:
        - float, objective value.
        """
        n = K_AA.shape[0]
        R = K_AA @ jnp.linalg.inv(K_AA + n * lambda2 * jnp.eye(n))
        S = jnp.diag((1 / (1 - jnp.diag(R))) ** 2)
        T = S @ (K_XX_WW - 2 * K_XX_WW @ R.T + R @ K_XX_WW @ R.T)
        lambda2_cost = jnp.trace(T)
        return lambda2_cost

    @staticmethod
    @jit
    def _predict_structural_function(alpha: jnp.ndarray, 
                                     D: jnp.ndarray, 
                                     K_ATildea: jnp.ndarray, 
                                     K_XTildeX: jnp.ndarray, 
                                     K_WW: jnp.ndarray, 
                                     CME_weights_: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the structural function.

        Parameters:
        - alpha: jnp.ndarray, alpha values.
        - D: jnp.ndarray, matrix D.
        - K_ATildea: jnp.ndarray, kernel matrix for A tilde and a.
        - K_XTildeX: jnp.ndarray, kernel matrix for X tilde and X.
        - K_WW: jnp.ndarray, kernel matrix for W.
        - CME_weights_: jnp.ndarray, conditional mean embedding weights.

        Returns:
        - jnp.ndarray, predicted structural function.
        """
        # n = K_Ww.shape[0]
        pred = alpha.T @ (K_ATildea.reshape(-1, 1) * ((K_XTildeX * (D.T @ K_WW)) @ CME_weights_))
        return pred

    def fit(self, 
            AWZX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Union[jnp.ndarray, None]],
            Y: jnp.ndarray):
        """
        Fit the KernelNegativeControlATT model.

        Parameters:
        - AWZX: Tuple containing A, W, Z, and X data.
        - Y: np.ndarray, outcome variable.
        """
        kernel_A, kernel_W, kernel_Z, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_X
        lambda_, zeta = self.lambda_, self.zeta
        optimize_regularization_parameters = self.optimize_regularization_parameters
        lambda_optimization_range = self.lambda_optimization_range
        zeta_optimization_range = self.zeta_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps
        large_lambda_zeta_option = self.large_lambda_zeta_option
        selecting_biggest_lambda_tol = self.selecting_biggest_lambda_tol
        selecting_biggest_zeta_tol = self.selecting_biggest_zeta_tol

        if len(AWZX) == 4:
            A, W, Z, X = AWZX
        elif len(AWZX) == 3:
            A, W, Z = AWZX
            X = None
        
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = kernel_X(X, X)
        K_ZTrainZTrain = kernel_Z(Z, Z)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        # K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_WTildeW = K_WWTilde.T
        # K_WTildeWTilde = K_WTrainWTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_ZZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_ZTildeZ = K_ZZTilde.T
        # K_ZTildeZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_XTildeX = K_XXTilde.T
        K_XTildeXTilde = K_XTrainXTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        if hasattr(self.kernel_A, 'use_length_scale_heuristic'):
            self.kernel_A.use_length_scale_heuristic = False

        if hasattr(self.kernel_W, 'use_length_scale_heuristic'):
            self.kernel_W.use_length_scale_heuristic = False

        if hasattr(self.kernel_Z, 'use_length_scale_heuristic'):
            self.kernel_Z.use_length_scale_heuristic = False

        if hasattr(self.kernel_X, 'use_length_scale_heuristic'):
            self.kernel_X.use_length_scale_heuristic = False

        ########## OPTIMIZE THE REGULARIZATION PARAMETERS IF IT IS SPECIFIED ###########################
        if optimize_regularization_parameters:
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda_objective_list = jnp.array([self._lambda_objective(lambda_, K_WW, K_ZZ, K_AA, K_XX, make_psd_eps) for lambda_ in lambda_list])
            lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]
            if large_lambda_zeta_option:
                lambda_objective_min = jnp.min(lambda_objective_list).item()
                _lambda_objective_list_minimizer_list = jnp.where(jnp.abs(lambda_objective_list - lambda_objective_min) <= selecting_biggest_lambda_tol)[0]
                lambda_ = jnp.max(lambda_list[_lambda_objective_list_minimizer_list])
            else:
                lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]

            self.lambda_ = lambda_ # For debugging purposes.

        C = K_AA * K_XX * K_ZZ 
        CTilde = K_AATilde * K_XXTilde * K_ZZTilde
        D = jnp.linalg.inv(C + stage1_data_size * lambda_ * jnp.eye(stage1_data_size)) @ CTilde
        M = K_ATildeATilde * K_XTildeXTilde * (D.T @ K_WW @ D)

        if optimize_regularization_parameters:
            pred_matrix_for_Y_stage_1 = (K_ATildeA * K_XTildeX * (D.T @ K_WW))
            zeta_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta_objective_list = jnp.array([self._zeta_objective(zeta_, pred_matrix_for_Y_stage_1, M, Y[stage1_idx], Y[stage2_idx]) for zeta_ in zeta_list])
            if large_lambda_zeta_option:
                zeta_objective_min = jnp.min(zeta_objective_list).item()
                zeta_objective_list_minimizer_list = jnp.where(jnp.abs(zeta_objective_list - zeta_objective_min) <= selecting_biggest_zeta_tol)[0]
                zeta = jnp.max(zeta_list[zeta_objective_list_minimizer_list])
            else:
                zeta = zeta_list[jnp.argmin(zeta_objective_list).item()]
            self.zeta = zeta
            self.zeta_objective_list = zeta_objective_list

        # alpha = jnp.linalg.inv(M @ M.T + stage2_data_size * zeta * M) @ M @ Y[stage2_idx]
        alpha = jnp.linalg.solve(make_psd(M @ M.T) + stage2_data_size * zeta * make_psd(M), M @ Y[stage2_idx])
        self.alpha = alpha
        self.ATilde = A[stage2_idx]
        self.A = A[stage1_idx]
        self.K_AA = K_AA
        self.K_XTildeX = K_XXTilde.T
        self.K_WW = K_WW
        self.D = D

        if optimize_regularization_parameters:
            K_XX_WW = K_XX * K_WW
            lambda2_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda2_objective_list = jnp.array([self._lambda2_objective(lambda2, K_AA, K_XX_WW) for lambda2 in lambda2_list])
            lambda2 = lambda2_list[jnp.argmin(lambda2_objective_list).item()]
            self.lambda2 = lambda2

    def predict(self, A: jnp.ndarray, aprime : jnp.ndarray ):
        """
        Predict outcomes for new data points.

        Parameters:
        - A (jnp.ndarray): New data points for variable A.
        - aprime (jnp.ndarray) : Historically observed treatment value a'.
        """
        alpha = self.alpha
        K_XTildeX = self.K_XTildeX
        n = K_XTildeX.shape[1]
        K_WW = self.K_WW
        D = self.D
        A_test = A.reshape(-1, 1)
        aprime = jnp.array(aprime).reshape(-1, 1)
        K_Aaprime = self.kernel_A(self.A, aprime)
        CME_weights_ = jnp.linalg.solve(self.K_AA + n * self.lambda2 * jnp.eye(n), K_Aaprime) # Conditinal Mean Embedding Weights
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        f_struct_pred = np.array([self._predict_structural_function(alpha, D, K_ATildeATest[:, jj], K_XTildeX, K_WW, CME_weights_).item() for jj in range(K_ATildeATest.shape[1])])
        return f_struct_pred


class KernelNegativeControlCATE(BaseEstimator, RegressorMixin):

    """
    A class to estimate the Average Treatment Effect (ATE) using kernel methods 
    with negative control variables.

    Attributes:
    -----------
    kernel_A : Callable
        Kernel function for the treatment variable A.
    kernel_W : Callable
        Kernel function for the negative control variable W.
    kernel_Z : Callable
        Kernel function for the negative control variable Z.
    kernel_V : Callable
        Kernel function for the negative control variable Z.
    kernel_X : Callable
        Kernel function for the covariates X. Default is RBF().
    lambda_ : float
        Regularization parameter for the kernel ridge regression. Default is 0.1.
    zeta : float
        Regularization parameter for the second stage regression. Default is 0.1.
    optimize_regularization_parameters : bool
        Whether to optimize the regularization parameters. Default is True.
    lambda_optimization_range : Tuple[float, float]
        Range for optimizing lambda. Default is (1e-9, 1.0).
    zeta_optimization_range : Tuple[float, float]
        Range for optimizing zeta. Default is (1e-9, 1.0).
    kwargs : dict
        Additional keyword arguments for customization.
    """
    def __init__(self, 
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Z: Callable,
                 kernel_V: Callable,
                 kernel_X: Callable = RBF(),
                 lambda_: float = 0.1,
                 zeta: float = 0.1, 
                 lambda2_: float = 0.1,
                 optimize_regularization_parameters: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 **kwargs) -> None:
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        large_lambda_zeta_option = kwargs.pop('large_lambda_zeta_option', False)
        selecting_biggest_lambda_tol = kwargs.pop('selecting_biggest_lambda_tol', 1e-9)
        selecting_biggest_zeta_tol = kwargs.pop('selecting_biggest_zeta_tol', 1e-9)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_V_params = kwargs.pop('kernel_V_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Callable)):
            raise Exception("Kernel for A must be callable")
        if (not isinstance(kernel_W, Callable)):
            raise Exception("Kernel for W must be callable")
        if (not isinstance(kernel_Z, Callable)):
            raise Exception("Kernel for Z must be callable")
        if (not isinstance(kernel_V, Callable)):
            raise Exception("Kernel for V must be callable")
        if (not isinstance(kernel_X, Callable)):
            raise Exception("Kernel for X must be callable")
        
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_V = kernel_V
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_V_params is not None:
            self.kernel_V.set_params(**kernel_V_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda_, self.zeta, self.lambda2_ = lambda_, zeta, lambda2_
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.lambda_optimization_range = lambda_optimization_range
        self.zeta_optimization_range = zeta_optimization_range
        self.large_lambda_zeta_option = large_lambda_zeta_option
        self.selecting_biggest_lambda_tol = selecting_biggest_lambda_tol
        self.selecting_biggest_zeta_tol = selecting_biggest_zeta_tol
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps 

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################
    @staticmethod
    @jit 
    def _lambda_objective(lambda_: float, 
                          K_WW: jnp.ndarray, 
                          K_ZZ: jnp.ndarray, 
                          K_AA: jnp.ndarray, 
                          K_XX: jnp.ndarray,
                          K_VV: jnp.ndarray,
                          make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for lambda optimization.

        Parameters:
        - lambda_: float, lambda.
        - K_WW: jnp.ndarray, kernel matrix for variable W.
        - K_ZZ: jnp.ndarray, kernel matrix for variable Z.
        - K_AA: jnp.ndarray, kernel matrix for variable A.
        - K_XX: jnp.ndarray, kernel matrix for variable X.
        - make_psd_eps: float, epsilon value for making matrices positive semi-definite.

        Returns:
        - objective: float, objective value.
        """
        n = K_AA.shape[0]
        identity_matrix = jnp.eye(n)
        kernel_hadamard_AZXV = K_AA * K_ZZ * K_XX * K_VV
        # H1 = identity_matrix - make_psd(kernel_hadamard_AZXV) @ jnp.linalg.inv(make_psd(kernel_hadamard_AZXV + n * lambda_ * identity_matrix, eps = make_psd_eps))
        H1 = identity_matrix - jnp.linalg.solve(make_psd(kernel_hadamard_AZXV + n * lambda_ * identity_matrix, eps = make_psd_eps), kernel_hadamard_AZXV).T
        H1_tilde_inv = jnp.diag(1 / jnp.diag(H1))
        objective = jnp.trace(H1_tilde_inv @ H1 @ K_WW @ H1 @ H1_tilde_inv) # * (1 / n) 
        return objective

    @staticmethod
    @jit 
    def _zeta_objective(zeta: float,
                        pred_matrix_for_Y_stage_1: jnp.ndarray,
                        M: jnp.ndarray,
                        Y_train_stage1: jnp.ndarray,
                        Y_train_stage2: jnp.ndarray) -> float:
        """
        Objective function for zeta optimization.

        Parameters:
        - zeta: float, regularization parameter.
        - pred_matrix_for_Y_stage_1: jnp.ndarray, prediction matrix for stage 1.
        - M: jnp.ndarray, matrix for second stage regression.
        - Y_train_stage1: jnp.ndarray, stage 1 training labels.
        - Y_train_stage2: jnp.ndarray, stage 2 training labels.

        Returns:
        - mse_loss: float, mean squared error loss.
        """
        # D_transpose_K_WW_D = (D.T @ K_WW @ D)
        # pred_matrix_for_Y_stage_1 = (K_ATildeA * K_XTildeX * D_transpose_K_WW_D) = (K_ATildeA * K_XTildeX * (D.T @ K_WW @ D))
        # alpha = jnp.linalg.inv(M @ M.T + m * zeta * M) @ M @ Y_train_stage2
        
        m = Y_train_stage2.shape[0]
        alpha = jnp.linalg.solve(make_psd(M @ M.T) + m * zeta * make_psd(M), make_psd(M) @ Y_train_stage2)
        Y_pred = pred_matrix_for_Y_stage_1.T @ alpha
        mse_loss = jnp.mean((Y_train_stage1.reshape(-1, 1) - Y_pred.reshape(-1, 1)) ** 2)
        return mse_loss

    @staticmethod
    @jit
    def _lambda2_objective(lambda2: float, K_VV: jnp.ndarray, K_XX_WW: jnp.ndarray) -> float:
        """
        Objective function for lambda2 optimization.

        Parameters:
        - lambda2: float, regularization parameter.
        - K_AA: jnp.ndarray, kernel matrix for variable A.
        - K_XX_WW: jnp.ndarray, combined kernel matrix for variables X and W.

        Returns:
        - float, objective value.
        """
        n = K_VV.shape[0]
        # R = K_VV @ jnp.linalg.inv(K_VV + n * lambda2 * jnp.eye(n))
        R = jnp.linalg.solve(make_psd(K_VV + n * lambda2 * jnp.eye(n)), K_VV).T
        S = jnp.diag((1 / (1 - jnp.diag(R))) ** 2)
        T = S @ (K_XX_WW - 2 * K_XX_WW @ R.T + R @ K_XX_WW @ R.T)
        lambda2_cost = jnp.trace(T)
        return lambda2_cost
    
    @staticmethod
    @jit
    def _predict_structural_function(alpha: jnp.ndarray, 
                                     D: jnp.ndarray, 
                                     K_ATildea: jnp.ndarray, 
                                     K_VTildev: jnp.ndarray,
                                     K_XTildeX: jnp.ndarray, 
                                     K_WW: jnp.ndarray, 
                                     CME_weights_: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the structural function.

        Parameters:
        - alpha: jnp.ndarray, alpha values.
        - D: jnp.ndarray, matrix D.
        - K_ATildea: jnp.ndarray, kernel matrix for A tilde and a.
        - K_XTildeX: jnp.ndarray, kernel matrix for X tilde and X.
        - K_WW: jnp.ndarray, kernel matrix for W.
        - CME_weights_: jnp.ndarray, conditional mean embedding weights.

        Returns:
        - jnp.ndarray, predicted structural function.
        """
        # n = K_Ww.shape[0]
        pred = alpha.T @ ((K_ATildea * K_VTildev).reshape(-1, 1) * ((K_XTildeX * (D.T @ K_WW)) @ CME_weights_))
        return pred

    def fit(self, AWZVX: Tuple[jnp.ndarray, ...], Y: jnp.ndarray) -> None:
        """
        Fit the model to the data.

        Parameters:
        - AWZX: Tuple[jnp.ndarray, ...], tuple containing arrays for A, W, Z, and optionally X.
        - Y: jnp.ndarray, array of outcome variable.
        """
        kernel_A, kernel_W, kernel_Z, kernel_V, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_V, self.kernel_X
        lambda_, zeta, lambda2_ = self.lambda_, self.zeta, self.lambda2_
        optimize_regularization_parameters = self.optimize_regularization_parameters
        lambda_optimization_range = self.lambda_optimization_range
        zeta_optimization_range = self.zeta_optimization_range
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps
        large_lambda_zeta_option = self.large_lambda_zeta_option
        selecting_biggest_lambda_tol = self.selecting_biggest_lambda_tol
        selecting_biggest_zeta_tol = self.selecting_biggest_zeta_tol

        if len(AWZVX) == 5:
            A, W, Z, V, X = AWZVX
        elif len(AWZVX) == 4:
            A, W, Z, V = AWZVX
            X = None
        
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_VTrainVTrain = kernel_V(V, V)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = kernel_X(X, X)
        K_ZTrainZTrain = kernel_Z(Z, Z)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_X = kernel_X

        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        # K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_WTildeW = K_WWTilde.T
        # K_WTildeWTilde = K_WTrainWTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_ZZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_ZTildeZ = K_ZZTilde.T
        # K_ZTildeZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_VV = K_VTrainVTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_VVTilde = K_VTrainVTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_VTildeV = K_VVTilde.T
        K_VTildeVTilde = K_VTrainVTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_XTildeX = K_XXTilde.T
        K_XTildeXTilde = K_XTrainXTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        if hasattr(self.kernel_A, 'use_length_scale_heuristic'):
            self.kernel_A.use_length_scale_heuristic = False

        if hasattr(self.kernel_W, 'use_length_scale_heuristic'):
            self.kernel_W.use_length_scale_heuristic = False

        if hasattr(self.kernel_Z, 'use_length_scale_heuristic'):
            self.kernel_Z.use_length_scale_heuristic = False

        if hasattr(self.kernel_V, 'use_length_scale_heuristic'):
            self.kernel_V.use_length_scale_heuristic = False

        if hasattr(self.kernel_X, 'use_length_scale_heuristic'):
            self.kernel_X.use_length_scale_heuristic = False

        ########## OPTIMIZE THE REGULARIZATION PARAMETERS IF IT IS SPECIFIED ###########################
        if optimize_regularization_parameters:
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda_objective_list = jnp.array([self._lambda_objective(lambda_, K_WW, K_ZZ, K_AA, K_XX, K_VV, make_psd_eps) for lambda_ in lambda_list])
            lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]
            if large_lambda_zeta_option:
                lambda_objective_min = jnp.min(lambda_objective_list).item()
                _lambda_objective_list_minimizer_list = jnp.where(jnp.abs(lambda_objective_list - lambda_objective_min) <= selecting_biggest_lambda_tol)[0]
                lambda_ = jnp.max(lambda_list[_lambda_objective_list_minimizer_list])
            else:
                lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]

            self.lambda_ = lambda_ # For debugging purposes.

        C = K_AA * K_XX * K_ZZ * K_VV
        CTilde = K_AATilde * K_XXTilde * K_ZZTilde * K_VVTilde
        # D = jnp.linalg.inv(C + stage1_data_size * lambda_ * jnp.eye(stage1_data_size)) @ CTilde
        D = jnp.linalg.solve(C + stage1_data_size * lambda_ * jnp.eye(stage1_data_size), CTilde)
        M = K_ATildeATilde * K_XTildeXTilde * K_VTildeVTilde * (D.T @ K_WW @ D)

        if optimize_regularization_parameters:
            pred_matrix_for_Y_stage_1 = (K_ATildeA * K_XTildeX * K_VTildeV * (D.T @ K_WW))
            zeta_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta_objective_list = jnp.array([self._zeta_objective(zeta_, pred_matrix_for_Y_stage_1, M, Y[stage1_idx], Y[stage2_idx]) for zeta_ in zeta_list])
            if large_lambda_zeta_option:
                zeta_objective_min = jnp.min(zeta_objective_list).item()
                zeta_objective_list_minimizer_list = jnp.where(jnp.abs(zeta_objective_list - zeta_objective_min) <= selecting_biggest_zeta_tol)[0]
                zeta = jnp.max(zeta_list[zeta_objective_list_minimizer_list])
            else:
                zeta = zeta_list[jnp.argmin(zeta_objective_list).item()]
            self.zeta = zeta
            self.zeta_objective_list = zeta_objective_list

        # alpha = jnp.linalg.inv(M @ M.T + stage2_data_size * zeta * M) @ M @ Y[stage2_idx]
        alpha = jnp.linalg.solve(make_psd(M @ M.T) + stage2_data_size * zeta * make_psd(M), M @ Y[stage2_idx])
        self.alpha = alpha
        self.ATilde = A[stage2_idx]
        self.VTilde = V[stage2_idx]
        self.V = V[stage1_idx]
        self.K_XTildeX = K_XXTilde.T
        self.K_VV = K_VV
        self.K_WW = K_WW
        self.D = D

        if optimize_regularization_parameters:
            K_XX_WW = K_XX * K_WW
            lambda2_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda2_objective_list = jnp.array([self._lambda2_objective(lambda2, K_VV, K_XX_WW) for lambda2 in lambda2_list])
            lambda2_ = lambda2_list[jnp.argmin(lambda2_objective_list).item()]
            self.lambda2_ = lambda2_

    def predict(self, A: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the structural function values for new data.

        Parameters:
        - A: jnp.ndarray, new data for the treatment variable A.

        Returns:
        - f_struct_pred: jnp.ndarray, predicted structural function values.
        """
        alpha = self.alpha
        K_XTildeX = self.K_XTildeX
        K_WW = self.K_WW
        n = K_WW.shape[0]
        D = self.D
        A_test = A.reshape(-1, 1)
        v_test = jnp.array(v).reshape(-1, 1)
        K_ATildeATest = self.kernel_A(self.ATilde, A_test)
        K_VTildeVTest = self.kernel_V(self.VTilde, v_test)
        K_VVTest = self.kernel_V(self.V, v_test)
        if v_test.shape[0] == 1:
            K_VTildeVTest = jnp.tile(K_VTildeVTest, A_test.shape[0])
        # CME_weights_ = jnp.linalg.solve(self.K_VV + n * self.lambda2_ * jnp.eye(n), K_VVTest) # Conditinal Mean Embedding Weights
        CME_weights_ = jnp.linalg.inv(make_psd(self.K_VV) + n * self.lambda2_ * jnp.eye(n)) @ K_VVTest # Conditinal Mean Embedding Weights
        self.CME_weights_ = CME_weights_
        f_struct_pred = np.array([self._predict_structural_function(alpha, D, K_ATildeATest[:, jj], 
                                                                    K_VTildeVTest[:, jj], K_XTildeX, 
                                                                    K_WW, CME_weights_[:, jj].reshape(-1, 1)).item() 
                                                                    for jj in range(K_ATildeATest.shape[1])])
        return f_struct_pred


class SingleKernelProxyVariable(BaseEstimator, RegressorMixin):
    """
    Single Kernel Proxy Variable (SKPV) model for regression.
    """

    def __init__(self,
                 kernel_A: Callable,
                 kernel_W: Callable,
                 kernel_Y: Callable,
                 lambda_: float = 0.01,
                 eta: float = 0.01,
                 optimize_regularization_parameters: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 eta_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 **kwargs):
        """
        Initialize the SKPV model.

        Parameters:
        - kernel_A: Callable, kernel function for variable A.
        - kernel_W: Callable, kernel function for variable W.
        - kernel_Y: Callable, kernel function for variable Y.
        - lambda_: float, regularization parameter lambda.
        - eta: float, regularization parameter eta.
        - optimize_regularization_parameters: bool, whether to optimize regularization parameters.
        - lambda_optimization_range: Tuple[float, float], range for lambda optimization.
        - eta_optimization_range: Tuple[float, float], range for eta optimization.
        - kwargs: Additional arguments:
            - stage1_perc: float, percentage of data to use in stage 1 (default: 0.5).
            - large_lambda_eta_option: bool, whether to select the largest lambda/eta within a tolerance (default: False).
            - selecting_biggest_lambda_tol: float, tolerance for selecting the largest lambda (default: 1e-3).
            - selecting_biggest_eta_tol: float, tolerance for selecting the largest eta (default: 2e-2).
            - regularization_grid_points: int, number of points for grid used for optimizing the regularizers (default: 150).
            - make_psd_eps: float, epsilon value for making matrices positive semi-definite (default: 1e-7).
            - kernel_A_params: dict, additional parameters for kernel_A (default: None).
            - kernel_W_params: dict, additional parameters for kernel_W (default: None).
            - kernel_Y_params: dict, additional parameters for kernel_Y (default: None).
        """

        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        large_lambda_eta_option = kwargs.pop('large_lambda_eta_option', False)
        selecting_biggest_lambda_tol = kwargs.pop('selecting_biggest_lambda_tol', 1e-3)
        selecting_biggest_eta_tol = kwargs.pop('selecting_biggest_eta_tol', 2e-2)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Y_params = kwargs.pop('kernel_Y_params', None)

        if (not isinstance(kernel_A, Callable)):
            raise Exception("Kernel for A must be callable")
        if (not isinstance(kernel_W, Callable)):
            raise Exception("Kernel for W must be callable")
        if (not isinstance(kernel_Y, Callable)):
            raise Exception("Kernel for Y must be callable")
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Y = kernel_Y

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Y_params is not None:
            self.kernel_Y.set_params(**kernel_Y_params)

        self.lambda_, self.eta = lambda_, eta
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.lambda_optimization_range = lambda_optimization_range
        self.eta_optimization_range  = eta_optimization_range 
        self.stage1_perc = stage1_perc
        self.large_lambda_eta_option = large_lambda_eta_option
        self.selecting_biggest_lambda_tol = selecting_biggest_lambda_tol
        self.selecting_biggest_eta_tol = selecting_biggest_eta_tol
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps        

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################
        
    @staticmethod
    @jit 
    def SKPV_lambda_objective(lambda_: float, 
                              K_WW: jnp.ndarray, 
                              K_AA: jnp.ndarray, 
                              K_YY: jnp.ndarray, 
                              make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for lambda optimization.

        Parameters:
        - lambda_: float, lambda.
        - K_WW: jnp.ndarray, kernel matrix for variable W.
        - K_AA: jnp.ndarray, kernel matrix for variable A.
        - K_YY: jnp.ndarray, kernel matrix for variable Y.
        - make_psd_eps: float, epsilon value for making matrices positive semi-definite.

        Returns:
        - objective: float, objective value.
        """
        n = K_AA.shape[0]
        identity_matrix = jnp.eye(n)
        H1 = identity_matrix - (K_AA * K_YY) @ jnp.linalg.inv(make_psd(K_AA * K_YY, eps = make_psd_eps) + n * lambda_ * identity_matrix)
        # H1 = identity_matrix - jnp.linalg.solve((K_AA * K_YY + n * lambda_ * identity_matrix).T, (K_AA * K_YY).T).T 
        H1_tilde_inv = jnp.diag(1 / jnp.diag(H1))
        objective = (1 / n) * jnp.trace(H1_tilde_inv @ H1 @ K_WW @ H1 @ H1_tilde_inv)
        return objective

    @staticmethod
    @jit 
    def SKPV_eta_objective(eta: float, 
                           K_ATildeA: jnp.ndarray, 
                           K_WW: jnp.ndarray, 
                           M: jnp.ndarray, 
                           B: jnp.ndarray, 
                           Y_train_stage1: jnp.ndarray, 
                           Y_train_stage2: jnp.ndarray, 
                           make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for eta optimization.

        Parameters:
        - eta: float, eta.
        - K_ATildeA: jnp.ndarray, kernel matrix product of A and its transpose.
        - K_WW: jnp.ndarray, kernel matrix for variable W.
        - M: jnp.ndarray, intermediate matrix.
        - B: jnp.ndarray, intermediate matrix.
        - Y_train_stage1: jnp.ndarray, training labels for stage 1.
        - Y_train_stage2: jnp.ndarray, training labels for stage 2.
        - make_psd_eps: float, epsilon value for making matrices positive semi-definite.

        Returns:
        - objective: float, objective value.
        """
        m = K_ATildeA.shape[0]
        alpha = jnp.linalg.inv(make_psd(M, eps = make_psd_eps) + m * eta * jnp.eye(m)) @ Y_train_stage2
        Y_train_stage1_pred = (alpha.T @ (K_ATildeA * (B.T @ K_WW))).T
        objective = ((Y_train_stage1 - Y_train_stage1_pred) ** 2).mean() 
        return objective
    
    @staticmethod
    @jit
    def predict_structured_func(alpha: jnp.ndarray, B: jnp.ndarray, K_ATilde_a: jnp.ndarray, K_WWTrain: jnp.ndarray) -> float:
        """
        Predict structured function.

        Parameters:
        - alpha: jnp.ndarray, 
        - B: jnp.ndarray, 
        - K_ATilde_a: jnp.ndarray, 
        - K_WWTrain: jnp.ndarray, 

        Returns:
        - prediction: float, predicted value.
        """
        return (alpha.T @ (K_ATilde_a.reshape(-1, 1) * (B.T @ K_WWTrain))).mean()
    
    ########################################################################
    ##################### FIT AND PREDICT FUNCTIONS ########################
    ########################################################################

    def fit(self, AW: Tuple[jnp.ndarray, jnp.ndarray], Y: jnp.ndarray) -> 'SingleKernelProxyVariable':
        """
        Fit the SKPV model.

        Parameters:
        - AW: tuple, (A, W) where A and W are feature matrices.
        - Y: jnp.ndarray, target values.

        Returns:
        - self: SingleKernelProxyVariable, fitted model instance.
        """

        kernel_A, kernel_W, kernel_Y = self.kernel_A, self.kernel_W, self.kernel_Y
        lambda_, eta = self.lambda_, self.eta
        optimize_regularization_parameters = self.optimize_regularization_parameters
        lambda_optimization_range = self.lambda_optimization_range
        eta_optimization_range = self.eta_optimization_range
        stage1_perc = self.stage1_perc
        large_lambda_eta_option = self.large_lambda_eta_option
        selecting_biggest_lambda_tol = self.selecting_biggest_lambda_tol
        selecting_biggest_eta_tol = self.selecting_biggest_eta_tol
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps

        A, W = AW

        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_YTrainYTrain = kernel_Y(Y, Y)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Y = kernel_Y

        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        Y_train_stage1, Y_train_stage2 = Y[stage1_idx].reshape(-1, 1), Y[stage2_idx].reshape(-1, 1)
        A_train_stage2 = A[stage2_idx].reshape(-1, 1)
        W_train_stage1 = W[stage1_idx].reshape(-1, 1)

        self.A_train_stage2 = A_train_stage2
        self.W_train_stage1 = W_train_stage1

        ########### CONSTRUCT THE KERNEL MATRICES ##########################
        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_YY = K_YTrainYTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_YYTilde = K_YTrainYTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_WWTrain = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, train_indices).T)].reshape(stage1_data_size, train_data_size)
        self.K_WWTrain = K_WWTrain

        if hasattr(self.kernel_A, 'use_length_scale_heuristic'):
            self.kernel_A.use_length_scale_heuristic = False
        if hasattr(self.kernel_W, 'use_length_scale_heuristic'):
            self.kernel_W.use_length_scale_heuristic = False
        if hasattr(self.kernel_Y, 'use_length_scale_heuristic'):
            self.kernel_Y.use_length_scale_heuristic = False

        ### For debugging, I need the following for now
        self.K_WW = K_WW
        self.K_AA = K_AA
        self.K_YY = K_YY
        ########## OPTIMIZE THE REGULARIZATION PARAMETERS IF IT IS SPECIFIED ###########################
        ### The optimization can also be done by jax.scipy.optimize.minimize function as we did in Kernel Instrumental Variable Regression
        ### TODO: Check if grid search is more effective than jax.scipy.optimize.minimize. If jax.scipy.optimize.minimize is better, change the optimization procedure.     
        if optimize_regularization_parameters:
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            eta_list = jnp.logspace(jnp.log(eta_optimization_range[0]), jnp.log(eta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))

            SKPV_lambda_objective_list = jnp.array([self.SKPV_lambda_objective(lambda_, K_WW, K_AA, K_YY) for lambda_ in lambda_list])
            if large_lambda_eta_option:
                SKPV_lambda_objective_min = jnp.min(SKPV_lambda_objective_list).item()
                SKPV_lambda_objective_list_minimizer_list = jnp.where(jnp.abs(SKPV_lambda_objective_list - SKPV_lambda_objective_min) <= selecting_biggest_lambda_tol)[0]
                lambda_ = jnp.max(lambda_list[SKPV_lambda_objective_list_minimizer_list])
            else:
                lambda_ = lambda_list[jnp.argmin(SKPV_lambda_objective_list).item()]

            B = jnp.linalg.inv(make_psd(K_AA * K_YY, eps = make_psd_eps) + stage1_data_size * lambda_ * jnp.eye(stage1_data_size)) @ (K_AATilde * K_YYTilde)
            M = K_ATildeATilde * (B.T @ K_WW @ B)

            SKPV_eta_objective_list = jnp.array([self.SKPV_eta_objective(eta, K_ATildeA, K_WW, M, B, Y_train_stage1, Y_train_stage2, make_psd_eps) for eta in eta_list])
            if large_lambda_eta_option:
                SKPV_eta_objective_min = jnp.min(SKPV_eta_objective_list).item()
                SKPV_eta_objective_list_minimizer_list = jnp.where(jnp.abs(SKPV_eta_objective_list - SKPV_eta_objective_min) <= selecting_biggest_eta_tol)[0]
                eta = jnp.max(eta_list[SKPV_eta_objective_list_minimizer_list])
            else:
                eta = eta_list[jnp.argmin(SKPV_eta_objective_list).item()]

            self.lambda_,self.eta = lambda_, eta

            alpha = jnp.linalg.inv(make_psd(M, eps = make_psd_eps) + stage2_data_size * eta * jnp.eye(stage2_data_size)) @ Y_train_stage2

        ######### IF REGULARIZATION PARAMETER OPTIMIZATION IS NOT REQUIRED, DIRECTLY CONSTRUCT THE ESTIMATOR MATRICES ################
        else:
            B = jnp.linalg.inv(make_psd(K_AA * K_YY, eps = make_psd_eps) + stage1_data_size * lambda_ * jnp.eye(stage1_data_size)) @ (K_AATilde * K_YYTilde)
            M = K_ATildeATilde * (B.T @ K_WW @ B)
            alpha = jnp.linalg.inv(make_psd(M, eps = make_psd_eps) + stage2_data_size * eta * jnp.eye(stage2_data_size)) @ Y_train_stage2

        self.B, self.M, self.alpha = B, M, alpha

    def predict(self, AW: Tuple[jnp.ndarray, jnp.ndarray], mode: str = "structured_func_prediction") -> jnp.ndarray:
        """
        Predict using the SKPV model.

        Parameters:
        - AW: tuple, (A, W) where A and W are feature matrices.
        - mode: str, prediction mode ("structured_func_prediction" or "y_prediction").

        Returns:
        - prediction: np.ndarray, predicted values.
        """
        if isinstance(AW, tuple) and (len(AW) == 2):
            A, W = AW
        elif isinstance(AW, np.ndarray) or isinstance(AW, jnp.ndarray):
            A = AW
            if mode == "y_prediction":
                 raise Exception("For predicting y values, the proxy variables W must be supplied")
        else:
            raise Exception("For inference, the parameter AW must be either a tuple consisting both A and W or a (j)np.ndarray consisting only A.")
        
        K_ATildeATest = self.kernel_A(self.A_train_stage2, A)
        if mode == "structured_func_prediction":
            pred = jnp.array([self.predict_structured_func(self.alpha, self.B, K_ATildeATest[:, jj], self.K_WWTrain).item() for jj in range(K_ATildeATest.shape[1])])
        elif mode == "y_prediction":
            K_WWTest = self.kernel_W(self.W_train_stage1, W)
            k_aw = K_ATildeATest * (self.B.T @ K_WWTest)
            pred = (self.alpha.T @ k_aw).T
        else:
            raise Exception("There are only two prediction modes: 1) structured_func_prediction, 2) y_prediction. You should use either of them.")
        return pred
    
    def fit_predict(self, AW: Tuple[jnp.ndarray, jnp.ndarray], Y: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the SKPV model and make predictions.

        Parameters:
        - AW: tuple, (A, W) where A and W are feature matrices.
        - Y: jnp.ndarray, target values.

        Returns:
        - predictions: jnp.ndarray, predicted values.
        """
        self.fit(AW, Y)
        return self.predict(AW)


##### DEBUG ALGORITHMS
class KernelAlternativeProxyCATE_OLD(BaseEstimator, RegressorMixin):

    def __init__(self,
                 kernel_A: Kernel,
                 kernel_W: Kernel,
                 kernel_Z: Kernel,
                 kernel_V: Kernel,
                 kernel_X: Kernel = RBF(),
                 lambda_: float = 0.1,
                 eta: float = 0.1, 
                 zeta: float = 0.1,
                 lambda2_: float = 0.1,
                 optimize_lambda_parameters: bool = True,
                 optimize_eta_parameter: bool = True,
                 optimize_zeta_parameter: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 eta_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 zeta_optimization_range: Tuple[float, float] = (1e-7, 1.0),
                 **kwargs) -> None:
        """
        Initialize the KernelAlternativeProxyATE estimator.

        Parameters:
        - kernel_A (Kernel): Kernel function for variable A.
        - kernel_W (Kernel): Kernel function for variable W.
        - kernel_Z (Kernel): Kernel function for variable Z.
        - kernel_X (Kernel, optional): Kernel function for variable X. Defaults to RBF().
        - lambda_ (float, optional): Regularization parameter. Defaults to 0.1.
        - eta (float, optional): Regularization parameter for structural function prediction. Defaults to 0.1.
        - lambda2_ (float, optional): Second stage regularization parameter. Defaults to 0.1.
        - optimize_regularization_parameters (bool, optional): Flag to optimize regularization parameters. Defaults to True.
        - lambda_optimization_range (Tuple[float, float], optional): Range for lambda optimization. Defaults to (1e-9, 1.0).
        - **kwargs: Additional parameters.
        """
        stage1_perc = kwargs.pop('stage1_perc', 0.5)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)
        label_variance_in_eta_opt = kwargs.pop('label_variance_in_eta_opt', 0.0)
        make_psd_eps = kwargs.pop('make_psd_eps', 1e-9)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        kernel_W_params = kwargs.pop('kernel_W_params', None)
        kernel_Z_params = kwargs.pop('kernel_Y_params', None)
        kernel_V_params = kwargs.pop('kernel_V_params', None)
        kernel_X_params = kwargs.pop('kernel_X_params', None)

        if (not isinstance(kernel_A, Kernel)):
            raise Exception("Kernel for A must be callable Kernel class.")
        if (not isinstance(kernel_W, Kernel)):
            raise Exception("Kernel for W must be callable Kernel class.")
        if (not isinstance(kernel_Z, Kernel)):
            raise Exception("Kernel for Z must be callable Kernel class.")
        if (not isinstance(kernel_V, Kernel)):
            raise Exception("Kernel for V must be callable Kernel class.")
        if (not isinstance(kernel_X, Kernel)):
            raise Exception("Kernel for X must be callable Kernel class.")
        
        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_V = kernel_V
        self.kernel_X = kernel_X

        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)
        if kernel_W_params is not None:
            self.kernel_W.set_params(**kernel_W_params)
        if kernel_Z_params is not None:
            self.kernel_Z.set_params(**kernel_Z_params)
        if kernel_V_params is not None:
            self.kernel_V.set_params(**kernel_V_params)
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)

        self.lambda_, self.eta, self.zeta, self.lambda2_ = lambda_, eta, zeta, lambda2_
        self.optimize_lambda_parameters = optimize_lambda_parameters
        self.optimize_eta_parameter = optimize_eta_parameter
        self.optimize_zeta_parameter = optimize_zeta_parameter
        self.lambda_optimization_range = lambda_optimization_range
        self.eta_optimization_range = eta_optimization_range
        self.zeta_optimization_range = zeta_optimization_range
        self.label_variance_in_eta_opt = label_variance_in_eta_opt
        self.stage1_perc = stage1_perc
        self.regularization_grid_points = regularization_grid_points
        self.make_psd_eps = make_psd_eps        

    ########################################################################
    ###################### STATIC JIT FUNCTIONS ############################
    ########################################################################

    @staticmethod
    @jit 
    def _lambda_objective(lambda_: float, 
                          K_AWXV: jnp.ndarray, 
                          K_ZZ: jnp.ndarray, 
                          identity_matrix: jnp.ndarray,
                          make_psd_eps: float = 1e-9) -> float:
        """
        Objective function for lambda optimization.

        Parameters:
        - lambda_ (float): Regularization parameter.
        - K_WW (jnp.ndarray): Kernel matrix for variable W.
        - K_ZZ (jnp.ndarray): Kernel matrix for variable Z.
        - K_AA (jnp.ndarray): Kernel matrix for variable A.
        - K_XX (jnp.ndarray): Kernel matrix for variable X.
        - make_psd_eps (float, optional): Epsilon value for making matrices positive semi-definite. Defaults to 1e-9.

        Returns:
        - float: Objective value.
        """
        n = K_AWXV.shape[0]
        H1 = identity_matrix - jnp.linalg.solve(make_psd(K_AWXV + n * lambda_ * identity_matrix, eps = make_psd_eps), K_AWXV).T 
        H1_tilde_inv = jnp.diag(1 / jnp.diag(H1))
        objective = jnp.trace(H1_tilde_inv @ H1 @ K_ZZ @ H1 @ H1_tilde_inv)
        return objective
    
    @staticmethod
    @jit 
    def _eta_objective(eta, L, M, N, L2, M2, stage1_data_size, label_variance_in_eta_opt, make_psd_eps = 1e-9):
        stage2_data_size = L.shape[0] - 1
        alpha = jnp.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)
        cost = ((1 / stage1_data_size) * (alpha.T @ make_psd(L2, make_psd_eps) @ alpha) - 2 * (alpha.T @ M2))
        cost += label_variance_in_eta_opt * (2 / stage2_data_size) * jnp.trace(jnp.linalg.solve(make_psd(L + stage2_data_size * eta * N, make_psd_eps), L))
        return cost.reshape(())
    
    @staticmethod
    @jit 
    def _zeta_objective(zeta: float, 
                        K_VTildeVTilde: jnp.ndarray, 
                        K_ATildeATilde: jnp.ndarray) -> float:
        """
        Computes the objective function for optimization with respect to zeta.

        Parameters:
        - zeta: float, parameter for optimization.
        - K_ATildeATilde: jnp.ndarray, kernel matrix for data points in set A.
        - K_WXTilde: jnp.ndarray, kernel matrix for data points in sets W and X.

        Returns:
        - float: Objective value.
        """
        m = K_VTildeVTilde.shape[0]
        K_ATildeATilde_square = K_ATildeATilde * K_ATildeATilde
        R = jnp.linalg.solve(make_psd(K_VTildeVTilde + m * zeta * jnp.eye(m)), K_VTildeVTilde).T
        S = jnp.diag((1 / (1 - jnp.diag(R))) ** 2)
        T = S @ (K_ATildeATilde_square - 2 * K_ATildeATilde_square @ R.T + R @ K_ATildeATilde_square @ R.T)
        zeta_cost = jnp.trace(T)
        return zeta_cost
    
    @staticmethod
    @jit 
    def _lambda2_objective(lambda2_: float,
                           K_AA_VV: jnp.ndarray,
                           K_ZZ: jnp.ndarray,
                           Y: jnp.ndarray,
                           make_psd_eps: float = 1e-9) -> float:
        """
        Computes the objective function for optimization with respect to lambda2_.

        Parameters:
        - lambda2_ (float): Parameter for optimization.
        - K_AA_VV (jnp.ndarray): Combined kernel matrix for data points in set A and V.
        - K_ZZ (jnp.ndarray): Kernel matrix for data points in set Z.
        - Y (jnp.ndarray): Target values matrix.
        - make_psd_eps (float, optional): Epsilon value for making a matrix positive semidefinite. Defaults to 1e-9.

        Returns:
        - float: Objective value.
        """
        n = K_AA_VV.shape[0]
        identity_matrix = jnp.eye(n)
        H2 = identity_matrix - jnp.linalg.solve(make_psd(K_AA_VV + n * lambda2_ * identity_matrix, eps = make_psd_eps), K_AA_VV).T 
        H2_tilde_inv = jnp.diag(1 / jnp.diag(H2))
        objective = (1 / n) * jnp.trace(H2_tilde_inv @ H2 @ (K_ZZ * (Y @ Y.T)) @ H2 @ H2_tilde_inv)
        return objective

    @staticmethod
    @jit 
    def _predict_structural_function(alpha: jnp.ndarray,
                                     B: jnp.ndarray,
                                     B_bar: jnp.ndarray,
                                     third_stage_KRR_weights: jnp.ndarray,
                                     K_ATraina: jnp.ndarray,
                                     K_ATildea: jnp.ndarray,
                                     K_VTrainv: jnp.ndarray,
                                     ones_divided_by_m: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the structural function.

        Parameters:
        - alpha (jnp.ndarray): Coefficient array.
        - B (jnp.ndarray): Matrix B from second stage.
        - B_bar (jnp.ndarray): Matrix B_bar from second stage.
        - third_stage_KRR_weights (jnp.ndarray): Weights from third stage kernel ridge regression.
        - K_ATraina (jnp.ndarray): Kernel matrix between training set A and a test point.
        - K_ATildea (jnp.ndarray): Kernel matrix between stage 2 set A and a test point.
        - ones_divided_by_m (jnp.ndarray): Array of ones divided by stage 2 data size.

        Returns:
        - jnp.ndarray: Predicted values.
        """
        pred = (alpha[:-1].T @ ((B.T @ (third_stage_KRR_weights @ (K_ATraina * K_VTrainv))) * (K_ATildea)))
        pred += (alpha[-1] * ((B_bar.T @ (third_stage_KRR_weights @ (K_ATraina * K_VTrainv))) * (K_ATildea)) @ ones_divided_by_m)
        return pred

    ########################################################################
    ###################### FIT AND PREDICT FUNCTIONS #######################
    ########################################################################

    def fit(self, 
            AWZVX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray) -> None:
        """
        Fit the KernelAlternativeProxyATE model.

        Parameters:
        - AWZX (Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]): Tuple of data arrays (A, W, Z, X).
        - Y (np.ndarray): Target values.
        """
        kernel_A, kernel_W, kernel_Z, kernel_V, kernel_X = self.kernel_A, self.kernel_W, self.kernel_Z, self.kernel_V, self.kernel_X
        lambda_, eta, lambda2_, zeta = self.lambda_, self.eta, self.lambda2_, self.zeta
        optimize_lambda_parameters = self.optimize_lambda_parameters
        optimize_eta_parameter = self.optimize_eta_parameter
        optimize_zeta_parameter = self.optimize_zeta_parameter
        lambda_optimization_range = self.lambda_optimization_range
        eta_optimization_range = self.eta_optimization_range
        zeta_optimization_range = self.zeta_optimization_range
        label_variance_in_eta_opt = self.label_variance_in_eta_opt
        stage1_perc = self.stage1_perc
        regularization_grid_points = self.regularization_grid_points
        make_psd_eps = self.make_psd_eps

        if len(AWZVX) == 5:
            A, W, Z, V, X = AWZVX
        elif len(AWZVX) == 4:
            A, W, Z, V = AWZVX
            X = None
        
        K_ATrainATrain = kernel_A(A, A)
        K_WTrainWTrain = kernel_W(W, W)
        K_ZTrainZTrain = kernel_Z(Z, Z)
        K_VTrainVTrain = kernel_V(V, V)
        if X is None:
            K_XTrainXTrain = jnp.ones((W.shape[0], W.shape[0]))
        else:
            K_XTrainXTrain = kernel_X(X, X)

        self.kernel_A = kernel_A
        self.kernel_W = kernel_W
        self.kernel_Z = kernel_Z
        self.kernel_V = kernel_V
        self.kernel_X = kernel_X

        train_data_size = A.shape[0]
        train_indices = np.random.permutation(train_data_size)

        if (stage1_perc > 0.) & (stage1_perc < 1.):
            stage1_data_size = int(train_data_size * stage1_perc)
            stage2_data_size = train_data_size - stage1_data_size
            stage1_idx, stage2_idx = train_indices[:stage1_data_size], train_indices[stage1_data_size:]
        else:
            stage1_data_size, stage2_data_size = train_data_size, train_data_size
            stage1_idx, stage2_idx = train_indices, train_indices

        K_AA = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_AATilde = K_ATrainATrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        K_ATildeA = K_AATilde.T
        K_ATildeATilde = K_ATrainATrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_WW = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_WWTilde = K_WTrainWTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_WTildeW = K_WWTilde.T
        # K_WTildeWTilde = K_WTrainWTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_ZZ = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        # K_ZZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_ZTildeZ = K_ZZTilde.T
        # K_ZTildeZTilde = K_ZTrainZTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_VV = K_VTrainVTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_VVTilde = K_VTrainVTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_VTildeV = K_VVTilde.T
        K_VTildeVTilde = K_VTrainVTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        K_XX = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage1_idx).T)].reshape(stage1_data_size, stage1_data_size)
        K_XXTilde = K_XTrainXTrain[tuple(cartesian_product(stage1_idx, stage2_idx).T)].reshape(stage1_data_size, stage2_data_size)
        # K_XTildeX = K_XXTilde.T
        # K_XTildeXTilde = K_XTrainXTrain[tuple(cartesian_product(stage2_idx, stage2_idx).T)].reshape(stage2_data_size, stage2_data_size)

        if hasattr(self.kernel_A, 'use_length_scale_heuristic'):
            self.kernel_A.use_length_scale_heuristic = False

        if hasattr(self.kernel_W, 'use_length_scale_heuristic'):
            self.kernel_W.use_length_scale_heuristic = False

        if hasattr(self.kernel_Z, 'use_length_scale_heuristic'):
            self.kernel_Z.use_length_scale_heuristic = False

        if hasattr(self.kernel_V, 'use_length_scale_heuristic'):
            self.kernel_V.use_length_scale_heuristic = False

        if hasattr(self.kernel_X, 'use_length_scale_heuristic'):
            self.kernel_X.use_length_scale_heuristic = False

        ########## OPTIMIZE THE REGULARIZATION PARAMETERS IF IT IS SPECIFIED ###########################
        I_n = jnp.eye(stage1_data_size)
        I_m = jnp.eye(stage2_data_size)
        K_WAXV = K_WW * K_AA * K_XX * K_VV
        if optimize_lambda_parameters:
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda_objective_list = jnp.array([self._lambda_objective(lambda_, K_WAXV, K_ZZ, I_n, make_psd_eps) for lambda_ in lambda_list])
            lambda_ = lambda_list[jnp.argmin(lambda_objective_list).item()]
            self.lambda_list = lambda_list
            self.lambda_objective_list = lambda_objective_list

        if optimize_zeta_parameter:
            zeta_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta_objective_list = jnp.array([self._zeta_objective(zeta, K_VTildeVTilde, K_ATildeATilde) for zeta in zeta_list])
            zeta = zeta_list[jnp.argmin(zeta_objective_list).item()]
            self.zeta_list = zeta_list
            self.zeta_objective_list = zeta_objective_list

        ########### FIRST AND SECOND STAGE ########################################
        theta = jnp.linalg.solve(make_psd(K_VTildeVTilde) + stage2_data_size * zeta * I_m, K_VTildeVTilde)
        self.theta = theta
        ones_matrix_m = remove_diagonal_elements(jnp.ones((stage2_data_size, stage2_data_size)))
        stage1_ridge_weights = K_WAXV + stage1_data_size * lambda_ * I_n
        B = jnp.linalg.solve(stage1_ridge_weights, (K_WWTilde * K_XXTilde * K_VVTilde * K_AATilde))
        B_bar = jnp.linalg.solve(stage1_ridge_weights,  (((K_WWTilde * K_XXTilde * K_VVTilde) @ (theta * ones_matrix_m)) * K_AATilde))

        block_component1 = (B.T @ K_ZZ @ B) * K_ATildeATilde
        block_component2 = (B.T @ K_ZZ @ B_bar) * K_ATildeATilde
        block_component4 = (B_bar.T @ K_ZZ @ B_bar) * K_ATildeATilde
        ones_divided_by_m = jnp.ones((stage2_data_size)) / stage2_data_size

        L = jnp.vstack((block_component1, ones_divided_by_m.T @ block_component2.T)) @ jnp.hstack((block_component1, (block_component2 @ ones_divided_by_m).reshape(-1, 1)))
        M = jnp.vstack(((block_component2 @ ones_divided_by_m).reshape(-1, 1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))

        P = jnp.hstack((block_component1, (block_component2 @ ones_divided_by_m).reshape(-1, 1)))
        R = jnp.hstack(((ones_divided_by_m.T @ block_component2.T).reshape(1, -1), (ones_divided_by_m.T @ block_component4 @ ones_divided_by_m).reshape(-1, 1)))
        N = jnp.vstack((P, R))

        if optimize_eta_parameter:
            # zeta2 is only required if the parameter eta will be optimized using validation set (stage 1 data as a validation).
            zeta2_list = jnp.logspace(jnp.log(zeta_optimization_range[0]), jnp.log(zeta_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            zeta2_objective_list = jnp.array([self._zeta_objective(zeta, K_VV, K_AA) for zeta in zeta2_list])
            zeta2 = zeta_list[jnp.argmin(zeta2_objective_list).item()]

            theta2 = jnp.linalg.solve(make_psd(K_VV) + stage1_data_size * zeta2 * I_n, K_VV)
            ones_matrix_n = remove_diagonal_elements(jnp.ones((stage1_data_size, stage1_data_size)))
            B2 = jnp.linalg.solve(make_psd(stage1_ridge_weights, make_psd_eps), K_WAXV)
            B2_bar = jnp.linalg.solve(stage1_ridge_weights,  (((K_WW * K_XX * K_VV) @ (theta2 * ones_matrix_n)) * K_AA))
            ones_divided_by_n = jnp.ones((stage1_data_size)) / stage1_data_size    

            block_component12 = (B2.T @ K_ZZ @ B) * K_AATilde
            block_component22 = (B2.T @ K_ZZ @ B_bar) * K_AATilde
            block_component32 = (B.T @ K_ZZ @ B2_bar) * K_ATildeA
            block_component42 = (B_bar.T @ K_ZZ @ B2_bar) * K_ATildeA

            L2_sub = jnp.vstack((block_component12.T, ones_divided_by_m.T @ block_component22.T))
            L2 = L2_sub @ L2_sub.T
            M2 = jnp.vstack(((block_component32 @ ones_divided_by_n).reshape(-1, 1), (ones_divided_by_m.T @ block_component42 @ ones_divided_by_n).reshape(-1, 1)))

            eta_list = np.logspace(np.log(eta_optimization_range[0]), np.log(eta_optimization_range[1]), regularization_grid_points, base = np.exp(1))
            eta_objective_list = jnp.array([self._eta_objective(eta_, L, M, N, L2, M2, stage1_data_size, label_variance_in_eta_opt, make_psd_eps) for eta_ in eta_list]).reshape(-1, 1)

            eta = eta_list[jnp.argmin(eta_objective_list).item()]
            self.eta_objective_list = eta_objective_list
            self.eta_list = eta_list

        alpha = jnp.linalg.solve(make_psd(L / stage2_data_size + eta * N, make_psd_eps), M)

        ########### THIRD STAGE ########################################
        if optimize_lambda_parameters:
            lambda2_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            lambda2_objective_list = jnp.array([self._lambda2_objective(lambda2_, K_ATrainATrain * K_VTrainVTrain, K_ZTrainZTrain, Y, make_psd_eps) for lambda2_ in lambda2_list])
            lambda2_ = lambda2_list[jnp.argmin(lambda2_objective_list).item()]
            self.lambda2_objective_list = lambda2_objective_list

        K_ZZtrain = K_ZTrainZTrain[tuple(cartesian_product(stage1_idx, train_indices).T)].reshape(stage1_data_size, train_data_size)
        K_ATrainATrain_ = K_ATrainATrain[tuple(cartesian_product(train_indices, train_indices).T)].reshape(train_data_size, train_data_size)
        K_VTrainVTrain_ = K_VTrainVTrain[tuple(cartesian_product(train_indices, train_indices).T)].reshape(train_data_size, train_data_size)
        third_stage_KRR_weights = jnp.linalg.solve(make_psd(K_ATrainATrain_ * K_VTrainVTrain_ + train_data_size * lambda2_ * jnp.eye(train_data_size)), (K_ZZtrain.T * Y[train_indices])).T 
        self.alpha = alpha
        self.B, self.B_bar = B, B_bar
        self.third_stage_KRR_weights = third_stage_KRR_weights
        self.ones_divided_by_m = ones_divided_by_m
        self.ATrain = A
        self.VTrain = V
        self.train_indices = train_indices
        self.stage2_idx = stage2_idx

        ##### For debugging purpose, I might want to check the regularization values after optimization #######
        self.lambda_ = lambda_
        self.lambda2_ = lambda2_
        self.eta = eta
        self.zeta = zeta

    def predict(self, A: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Predict outcomes for new data points.

        Parameters:
        - A (jnp.ndarray): New data points for variable A.
        - v (jnp.ndarray): New data points for variable V.
        Returns:
        - jnp.ndarray: Predicted values of conditional treatment effect.
        """
        A_test = A.reshape(-1, 1)
        v_test = jnp.array(v).reshape(-1, 1)
        
        K_ATrainATest = self.kernel_A(self.ATrain, A_test)
        K_VTrainv = self.kernel_V(self.VTrain, v_test)

        test_indices = jnp.arange(A_test.shape[0])
        test_shape = test_indices.shape[0]
        if v_test.shape[0] == 1:
            K_VTrainv = jnp.tile(K_VTrainv, test_shape)

        K_ATrainATest_ = K_ATrainATest[tuple(cartesian_product(self.train_indices, test_indices).T)].reshape(self.train_indices.shape[0], test_shape)
        K_ATildeATest = K_ATrainATest[tuple(cartesian_product(self.stage2_idx, test_indices).T)].reshape(self.stage2_idx.shape[0], test_shape)

        K_VTrainVTest_ = K_VTrainv[tuple(cartesian_product(self.train_indices, test_indices).T)].reshape(self.train_indices.shape[0], test_shape)
        # K_VTildeVTest = K_VTrainv[tuple(cartesian_product(self.stage2_idx, test_indices).T)].reshape(self.stage2_idx.shape[0], test_shape)

        ones_divided_by_m = self.ones_divided_by_m
        alpha = self.alpha
        B, B_bar = self.B, self.B_bar
        third_stage_KRR_weights = self.third_stage_KRR_weights

        f_struct_pred = jnp.array([self._predict_structural_function(alpha, B, B_bar, 
                                                                     third_stage_KRR_weights, 
                                                                     K_ATrainATest_[:, jj], 
                                                                     K_ATildeATest[:, jj], 
                                                                     K_VTrainVTest_[:, jj],
                                                                     ones_divided_by_m).item() for jj in range(K_ATildeATest.shape[1])])
        return f_struct_pred
