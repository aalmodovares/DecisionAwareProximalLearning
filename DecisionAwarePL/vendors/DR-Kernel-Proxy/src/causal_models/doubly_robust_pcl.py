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
from utils.kernel_utils import RBF
from utils.linalg_utils import make_psd
from causal_models.proxy_causal_learning import KernelAlternativeProxyATE, KernelNegativeControlATE, KernelProxyVariableATE, ProxyMaximumMomentRestrictionATE
from causal_models.proxy_causal_learning import KernelAlternativeProxyATT, KernelProxyVariableATT, KernelAlternativeProxyCATE, KernelProxyVariableCATE
from typing import Callable, Tuple, Optional, Union, Dict

from jax import config
config.update("jax_enable_x64", True)

treatment_bridge_algo_param_dict_default = {
                                            "kernel_A" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                            "kernel_W" : RBF(use_length_scale_heuristic = True, use_jit_call = True), 
                                            "kernel_Z" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                            "kernel_V" : RBF(use_length_scale_heuristic = True, use_jit_call = True), # Only required for CATE algorithm
                                            "kernel_X" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                            "lambda_" : 1e-3,
                                            "eta" : 1e-3,
                                            "lambda2_" : 1e-3,
                                            "zeta" : 1e-3, # Only required for ATT or CATE algorithm
                                            "optimize_lambda_parameters" : True,
                                            "optimize_eta_parameter" : True, 
                                            "optimize_zeta_parameter" : True, # Only required for ATT or CATE algorithm
                                            "lambda_optimization_range" : (1e-5, 1.0),
                                            "eta_optimization_range" : (1e-5, 1.0),
                                            "zeta_optimization_range" : (1e-5, 1.0), # Only required for ATT or CATE algorithm
                                            "stage1_perc" : 0.5,
                                            "regularization_grid_points" : 50, 
                                            "make_psd_eps" : 1e-9,
                                            "label_variance_in_lambda_opt" : 0.,
                                            "label_variance_in_eta_opt" : 0.0,
                                            }

outcome_bridge_knc_algo_param_dict_default = {
                                            "algorithm_name" : "Kernel_Negative_Control",
                                            "kernel_A" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                            "kernel_W" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                            "kernel_Z" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                            "kernel_X" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                            "lambda_" : 1e-3,
                                            "zeta" : 1e-3, 
                                            "optimize_regularization_parameters" : True,
                                            "lambda_optimization_range" : (1e-5, 1.0),
                                            "zeta_optimization_range" : (1e-5, 1.0),
                                            "stage1_perc" : 0.5,
                                            "regularization_grid_points" : 25, 
                                            "make_psd_eps" : 1e-9,
                                            }

outcome_bridge_kpv_algo_param_dict_default = {
                                             "algorithm_name" : "Kernel_Proxy_Variable",
                                             "kernel_A" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                             "kernel_W" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                             "kernel_Z" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                             "kernel_V" : RBF(use_length_scale_heuristic = True, use_jit_call = True), # Only required for CATE algorithm
                                             "kernel_X" : RBF(use_length_scale_heuristic = True, use_jit_call = True),      
                                             "lambda1_" : 0.1,
                                             "lambda2_" : 0.1,
                                             "zeta" : 1e-3, # Only required for ATT or CATE algorithm
                                             "optimize_lambda1_parameter" : True,
                                             "optimize_lambda2_parameter" : True,
                                             "optimize_zeta_parameter" : True, # Only required for ATT or CATE algorithm
                                             "lambda1_optimization_range" : (1e-5, 1.0),
                                             "lambda2_optimization_range" : (1e-5, 1.0),
                                             "zeta_optimization_range" : (1e-5, 1.0), # Only required for ATT or CATE algorithm
                                             "stage1_perc" : 0.5,
                                             "regularization_grid_points" : 25, 
                                             "make_psd_eps" : 1e-9,
                                                }

outcome_bridge_pmmr_algo_param_dict_default = {
                                             "algorithm_name" : "Proximal_Maximum_Moment_Restriction",
                                             "kernel_A" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                             "kernel_W" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                             "kernel_Z" : RBF(use_length_scale_heuristic = True, use_jit_call = True),
                                             "kernel_X" : RBF(use_length_scale_heuristic = True, use_jit_call = True),      
                                             "lambda_" : 1e-3,
                                             "optimize_lambda_parameter" : True,
                                             "validation_percentage" : 0.1,
                                             "lambda_optimization_range" : (5*1e-5, 1e-3),
                                             "regularization_grid_points" : 25,
                                             "make_psd_eps" : 1e-9,
                                                }

class DoublyRobustKernelProxyATE(BaseEstimator, RegressorMixin):

    def __init__(self,
                 treatment_bridge_algo_param_dict : Dict = treatment_bridge_algo_param_dict_default,
                 outcome_bridge_algo_param_dict : Dict = outcome_bridge_kpv_algo_param_dict_default,
                 lambda_DR : float = 1e-3,
                 optimize_lambda_DR_parameter: bool = True,
                 lambda_DR_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 **kwargs,
                 ):
        self.treatment_bridge_algo_param_dict = treatment_bridge_algo_param_dict
        self.outcome_bridge_algo_param_dict = outcome_bridge_algo_param_dict
        self.lambda_DR = lambda_DR
        self.optimize_lambda_DR_parameter = optimize_lambda_DR_parameter
        self.lambda_DR_optimization_range = lambda_DR_optimization_range
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.label_variance_in_lambda_DR_opt = kwargs.pop('label_variance_in_lambda_DR_opt', 0)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 5e-9)

        self.treatment_bridge_algo = KernelAlternativeProxyATE(**treatment_bridge_algo_param_dict)
        if outcome_bridge_algo_param_dict["algorithm_name"] == "Kernel_Proxy_Variable":
            self.outcome_bridge_algo = KernelProxyVariableATE(**outcome_bridge_algo_param_dict)
        elif outcome_bridge_algo_param_dict["algorithm_name"] == "Proxy_Maximum_Moment_Restriction":
             self.outcome_bridge_algo = ProxyMaximumMomentRestrictionATE(**outcome_bridge_algo_param_dict)
        elif outcome_bridge_algo_param_dict["algorithm_name"] == "Kernel_Negative_Control":
            self.outcome_bridge_algo = KernelNegativeControlATE(**outcome_bridge_algo_param_dict)

    @staticmethod
    @jit
    def _lambda_DR_objective(lambda_DR: float,
                             K_ZW, K_AA, 
                             identity_matrix, 
                             label_variance_in_lambda_DR_opt = 0,
                             make_psd_eps = 1e-9):
        n = K_ZW.shape[0]
        ridge_weights = make_psd(K_ZW + n * lambda_DR * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_ZW).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_AA @ H1_tilde_inv_times_H1) 
        objective += label_variance_in_lambda_DR_opt * jnp.trace(R)
        objective += (1 / n) * label_variance_in_lambda_DR_opt * jnp.sum((H1_diag - 1) / H1_diag)
        objective += (1 / n) * label_variance_in_lambda_DR_opt * jnp.trace(R @ H1_tilde_inv @ R.T)
        return objective
    
    def fit(self, 
            AWZX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray,) -> None:
        if len(AWZX) == 4:
            A, W, Z, X = AWZX
        elif len(AWZX) == 3:
            A, W, Z = AWZX
            X = None
        
        self.treatment_bridge_algo.fit(AWZX, Y)
        self.outcome_bridge_algo.fit(AWZX, Y)
        outcome_kernel_A = self.outcome_bridge_algo.kernel_A
        K_AA = outcome_kernel_A(A, A)

        if self.optimize_lambda_DR_parameter:
            lambda_DR_optimization_range = self.lambda_DR_optimization_range
            regularization_grid_points = self.regularization_grid_points
            lambda_list = jnp.logspace(jnp.log(lambda_DR_optimization_range[0]), jnp.log(lambda_DR_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            
            treatment_kernel_Z = self.treatment_bridge_algo.kernel_Z
            outcome_kernel_W = self.outcome_bridge_algo.kernel_W
            
            treatment_K_ZZ = treatment_kernel_Z(Z, Z)
            outcome_kernel_WW = outcome_kernel_W(W, W)

            K_ZW = treatment_K_ZZ * outcome_kernel_WW
            Identity_mat = jnp.eye(K_AA.shape[0])
            lambda_objective_list = jnp.array([self._lambda_DR_objective(lambda_, K_ZW, K_AA, 
                                                                         Identity_mat, 
                                                                         self.label_variance_in_lambda_DR_opt,
                                                                         self.make_psd_eps) for lambda_ in lambda_list])

            lambda_DR = lambda_list[jnp.argmin(lambda_objective_list).item()]
            self.lambda_objective_list = lambda_objective_list
            self.lambda_DR = lambda_DR
            
        self.W = W
        self.Z = Z
        self.X = X
        self.A = A
        self.K_AA = K_AA
        self.kernel_A = self.treatment_bridge_algo.kernel_A

    def predict(self, A: jnp.ndarray):
        if A.ndim == 2:
            A = A
        else:
            A = A.reshape(-1, 1)
        
        treatment_bridge_algo = self.treatment_bridge_algo
        outcome_bridge_algo = self.outcome_bridge_algo
        treatment_bridge_algo_pred = treatment_bridge_algo.predict(A).reshape(-1)
        outcome_bridge_algo_pred = outcome_bridge_algo.predict(A).reshape(-1)
        self.treatment_bridge_algo_pred, self.outcome_bridge_algo_pred = treatment_bridge_algo_pred, outcome_bridge_algo_pred
        treatment_bridge_algo_bridge_pred = treatment_bridge_algo._predict_bridge_func(A, self.Z, self.X)
        outcome_bridge_algo_bridge_pred = outcome_bridge_algo._predict_bridge_func(A, self.W, self.X)
        lambda_DR = self.lambda_DR
        identity_matrix = jnp.eye(self.A.shape[0])
        DR_KRR_Weights = jnp.linalg.solve(make_psd(self.K_AA + self.A.shape[0] * lambda_DR * identity_matrix), outcome_bridge_algo.kernel_A(self.A, A))
        slack_prediction = ((treatment_bridge_algo_bridge_pred * outcome_bridge_algo_bridge_pred) * DR_KRR_Weights.T).sum(1)
        self.slack_prediction = slack_prediction
        f_struct_pred = treatment_bridge_algo_pred + outcome_bridge_algo_pred - slack_prediction
        return f_struct_pred


class DoublyRobustKernelProxyATT(BaseEstimator, RegressorMixin):

    def __init__(self,
                 treatment_bridge_algo_param_dict : Dict = treatment_bridge_algo_param_dict_default,
                 outcome_bridge_algo_param_dict : Dict = outcome_bridge_kpv_algo_param_dict_default,
                 lambda_DR : float = 1e-3,
                 optimize_lambda_DR_parameter: bool = True,
                 lambda_DR_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 **kwargs,
                 ):
        self.treatment_bridge_algo_param_dict = treatment_bridge_algo_param_dict
        self.outcome_bridge_algo_param_dict = outcome_bridge_algo_param_dict
        self.lambda_DR = lambda_DR
        self.optimize_lambda_DR_parameter = optimize_lambda_DR_parameter
        self.lambda_DR_optimization_range = lambda_DR_optimization_range
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.label_variance_in_lambda_DR_opt = kwargs.pop('label_variance_in_lambda_DR_opt', 0)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 5e-9)

        self.treatment_bridge_algo = KernelAlternativeProxyATT(**treatment_bridge_algo_param_dict)
        if outcome_bridge_algo_param_dict["algorithm_name"] == "Kernel_Proxy_Variable":
            self.outcome_bridge_algo = KernelProxyVariableATT(**outcome_bridge_algo_param_dict)
        elif outcome_bridge_algo_param_dict["algorithm_name"] == "Proxy_Maximum_Moment_Restriction":
            raise Exception("Not implemented yet!")
        elif outcome_bridge_algo_param_dict["algorithm_name"] == "Kernel_Negative_Control":
            raise Exception("Not implemented yet!")

    @staticmethod
    @jit
    def _lambda_DR_objective(lambda_DR: float,
                             K_ZW, K_AA, 
                             identity_matrix, 
                             label_variance_in_lambda_DR_opt = 0,
                             make_psd_eps = 1e-9):
        n = K_ZW.shape[0]
        ridge_weights = make_psd(K_ZW + n * lambda_DR * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_ZW).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_AA @ H1_tilde_inv_times_H1) 
        objective += label_variance_in_lambda_DR_opt * jnp.trace(R)
        objective += (1 / n) * label_variance_in_lambda_DR_opt * jnp.sum((H1_diag - 1) / H1_diag)
        objective += (1 / n) * label_variance_in_lambda_DR_opt * jnp.trace(R @ H1_tilde_inv @ R.T)
        return objective
    
    def fit(self, 
            AWZX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray,
            a_prime: jnp.array,) -> None:
        if len(AWZX) == 4:
            A, W, Z, X = AWZX
        elif len(AWZX) == 3:
            A, W, Z = AWZX
            X = None
        self.a_prime = a_prime
        self.treatment_bridge_algo.fit(AWZX, Y, a_prime)
        self.outcome_bridge_algo.fit(AWZX, Y)
        outcome_kernel_A = self.outcome_bridge_algo.kernel_A
        K_AA = outcome_kernel_A(A, A)

        if self.optimize_lambda_DR_parameter:
            lambda_DR_optimization_range = self.lambda_DR_optimization_range
            regularization_grid_points = self.regularization_grid_points
            lambda_list = jnp.logspace(jnp.log(lambda_DR_optimization_range[0]), jnp.log(lambda_DR_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            
            treatment_kernel_Z = self.treatment_bridge_algo.kernel_Z
            outcome_kernel_W = self.outcome_bridge_algo.kernel_W
            
            treatment_K_ZZ = treatment_kernel_Z(Z, Z)
            outcome_kernel_WW = outcome_kernel_W(W, W)

            K_ZW = treatment_K_ZZ * outcome_kernel_WW
            Identity_mat = jnp.eye(K_AA.shape[0])
            lambda_objective_list = jnp.array([self._lambda_DR_objective(lambda_, K_ZW, K_AA, 
                                                                         Identity_mat, 
                                                                         self.label_variance_in_lambda_DR_opt,
                                                                         self.make_psd_eps) for lambda_ in lambda_list])

            lambda_DR = lambda_list[jnp.argmin(lambda_objective_list).item()]
            self.lambda_objective_list = lambda_objective_list
            self.lambda_DR = lambda_DR
            
        self.W = W
        self.Z = Z
        self.X = X
        self.A = A
        self.K_AA = K_AA
        self.kernel_A = self.treatment_bridge_algo.kernel_A

    def predict(self, A: jnp.ndarray,):
        if A.ndim == 2:
            A = A
        else:
            A = A.reshape(-1, 1)
        a_prime = self.a_prime
        treatment_bridge_algo = self.treatment_bridge_algo
        outcome_bridge_algo = self.outcome_bridge_algo
        treatment_bridge_algo_pred = treatment_bridge_algo.predict(A).reshape(-1)
        outcome_bridge_algo_pred = outcome_bridge_algo.predict(A, a_prime).reshape(-1)
        self.treatment_bridge_algo_pred, self.outcome_bridge_algo_pred = treatment_bridge_algo_pred, outcome_bridge_algo_pred
        treatment_bridge_algo_bridge_pred = treatment_bridge_algo._predict_bridge_func(A, self.Z, self.X)
        outcome_bridge_algo_bridge_pred = outcome_bridge_algo._predict_bridge_func(A, self.W, self.X)
        lambda_DR = self.lambda_DR
        identity_matrix = jnp.eye(self.A.shape[0])
        DR_KRR_Weights = jnp.linalg.solve(make_psd(self.K_AA + self.A.shape[0] * lambda_DR * identity_matrix), outcome_bridge_algo.kernel_A(self.A, A))
        slack_prediction = ((treatment_bridge_algo_bridge_pred * outcome_bridge_algo_bridge_pred) * DR_KRR_Weights.T).sum(1)
        self.slack_prediction = slack_prediction
        f_struct_pred = treatment_bridge_algo_pred + outcome_bridge_algo_pred - slack_prediction
        return f_struct_pred


class DoublyRobustKernelProxyCATE(BaseEstimator, RegressorMixin):

    def __init__(self,
                 treatment_bridge_algo_param_dict : Dict = treatment_bridge_algo_param_dict_default,
                 outcome_bridge_algo_param_dict : Dict = outcome_bridge_kpv_algo_param_dict_default,
                 lambda_DR : float = 1e-3,
                 optimize_lambda_DR_parameter: bool = True,
                 lambda_DR_optimization_range: Tuple[float, float] = (1e-5, 1.0),
                 **kwargs,
                 ):
        self.treatment_bridge_algo_param_dict = treatment_bridge_algo_param_dict
        self.outcome_bridge_algo_param_dict = outcome_bridge_algo_param_dict
        self.lambda_DR = lambda_DR
        self.optimize_lambda_DR_parameter = optimize_lambda_DR_parameter
        self.lambda_DR_optimization_range = lambda_DR_optimization_range
        self.regularization_grid_points = kwargs.pop('regularization_grid_points', 25)
        self.label_variance_in_lambda_DR_opt = kwargs.pop('label_variance_in_lambda_DR_opt', 0)
        self.make_psd_eps = kwargs.pop('make_psd_eps', 5e-9)

        self.treatment_bridge_algo = KernelAlternativeProxyCATE(**treatment_bridge_algo_param_dict)
        if outcome_bridge_algo_param_dict["algorithm_name"] == "Kernel_Proxy_Variable":
            self.outcome_bridge_algo = KernelProxyVariableCATE(**outcome_bridge_algo_param_dict)
        elif outcome_bridge_algo_param_dict["algorithm_name"] == "Proxy_Maximum_Moment_Restriction":
            raise Exception("Not implemented yet!")
        elif outcome_bridge_algo_param_dict["algorithm_name"] == "Kernel_Negative_Control":
            raise Exception("Not implemented yet!")

    @staticmethod
    @jit
    def _lambda_DR_objective(lambda_DR: float,
                             K_ZW, K_AA_VV, 
                             identity_matrix, 
                             label_variance_in_lambda_DR_opt = 0,
                             make_psd_eps = 1e-9):
        n = K_ZW.shape[0]
        ridge_weights = make_psd(K_ZW + n * lambda_DR * identity_matrix, eps = make_psd_eps)
        R = jnp.linalg.solve(ridge_weights, K_ZW).T
        H1 = identity_matrix - R
        H1_diag = jnp.diag(H1)
        H1_tilde_inv = jnp.diag(1 / H1_diag)
        H1_tilde_inv_times_H1 = H1_tilde_inv @ H1
        objective = (1 / n) * jnp.trace(H1_tilde_inv_times_H1 @ K_AA_VV @ H1_tilde_inv_times_H1) 
        objective += label_variance_in_lambda_DR_opt * jnp.trace(R)
        objective += (1 / n) * label_variance_in_lambda_DR_opt * jnp.sum((H1_diag - 1) / H1_diag)
        objective += (1 / n) * label_variance_in_lambda_DR_opt * jnp.trace(R @ H1_tilde_inv @ R.T)
        return objective
    
    def fit(self, 
            AWZVX: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], 
            Y: jnp.ndarray,) -> None:
        if len(AWZVX) == 5:
            A, W, Z, V, X = AWZVX
        elif len(AWZVX) == 4:
            A, W, Z, V = AWZVX
            X = None
        
        self.treatment_bridge_algo.fit(AWZVX, Y)
        self.outcome_bridge_algo.fit(AWZVX, Y)
        outcome_kernel_A = self.outcome_bridge_algo.kernel_A
        outcome_kernel_V = self.outcome_bridge_algo.kernel_V
        K_AA = outcome_kernel_A(A, A)
        K_VV = outcome_kernel_V(V, V)

        if self.optimize_lambda_DR_parameter:
            lambda_DR_optimization_range = self.lambda_DR_optimization_range
            regularization_grid_points = self.regularization_grid_points
            lambda_list = jnp.logspace(jnp.log(lambda_DR_optimization_range[0]), jnp.log(lambda_DR_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            
            treatment_kernel_Z = self.treatment_bridge_algo.kernel_Z
            outcome_kernel_W = self.outcome_bridge_algo.kernel_W
            
            treatment_K_ZZ = treatment_kernel_Z(Z, Z)
            outcome_kernel_WW = outcome_kernel_W(W, W)

            K_ZW = treatment_K_ZZ * outcome_kernel_WW
            Identity_mat = jnp.eye(K_AA.shape[0])
            lambda_objective_list = jnp.array([self._lambda_DR_objective(lambda_, K_ZW, K_AA * K_VV, 
                                                                         Identity_mat, 
                                                                         self.label_variance_in_lambda_DR_opt,
                                                                         self.make_psd_eps) for lambda_ in lambda_list])

            lambda_DR = lambda_list[jnp.argmin(lambda_objective_list).item()]
            self.lambda_objective_list = lambda_objective_list
            self.lambda_DR = lambda_DR
            
        self.W = W
        self.Z = Z
        self.X = X
        self.A = A
        self.V = V
        self.K_AA = K_AA
        self.K_VV = K_VV
        # self.kernel_A = self.treatment_bridge_algo.kernel_A
        # self.kernel_V = self.treatment_bridge_algo.kernel_V

    def predict(self, A: jnp.ndarray, V: jnp.ndarray):
        if A.ndim == 2:
            A = A
        else:
            A = A.reshape(-1, 1)
        
        treatment_bridge_algo = self.treatment_bridge_algo
        outcome_bridge_algo = self.outcome_bridge_algo
        treatment_bridge_algo_pred = treatment_bridge_algo.predict(A, V).reshape(-1)
        outcome_bridge_algo_pred = outcome_bridge_algo.predict(A, V).reshape(-1)
        self.treatment_bridge_algo_pred, self.outcome_bridge_algo_pred = treatment_bridge_algo_pred, outcome_bridge_algo_pred
        treatment_bridge_algo_bridge_pred = treatment_bridge_algo._predict_bridge_func(A, self.Z, V, self.X)
        outcome_bridge_algo_bridge_pred = outcome_bridge_algo._predict_bridge_func(A, self.W, V, self.X)
        lambda_DR = self.lambda_DR
        identity_matrix = jnp.eye(self.A.shape[0])
        DR_KRR_Weights = jnp.linalg.solve(make_psd(self.K_AA * self.K_VV + self.A.shape[0] * lambda_DR * identity_matrix), outcome_bridge_algo.kernel_A(self.A, A) * outcome_bridge_algo.kernel_V(self.V, V))
        slack_prediction = ((treatment_bridge_algo_bridge_pred * outcome_bridge_algo_bridge_pred) * DR_KRR_Weights.T).sum(1)
        self.slack_prediction = slack_prediction
        f_struct_pred = treatment_bridge_algo_pred + outcome_bridge_algo_pred - slack_prediction
        return f_struct_pred
