import numpy as np

import gurobipy as gp
from gurobipy import GRB

import sys
import os

from src.util import Util
from src.recovery_template import RecoveryTemplate

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import Parameters

class MLE(RecoveryTemplate):
    """MLE class is used to test the maximum likelihood estimation (MLE) model.

    X set: adversary's random input set
    Y set: target, victims
    """ 

    def __init__(self, params: Parameters, use_noise: bool = False):
        """Initialize the CompressedSensing class with given parameters."""
        super().__init__(params)
        self.A = None
        self.use_noise = use_noise
        self.noise_lambda = self.params.noise_magnitude

    def setup(self):
        """Generate the input set X
        """
        n = self.num_targets
        m = self.num_invocations
        size = (m, n)
        self.A = Util.gen_normal_data(self.params.mean_val_X, self.params.scale_val_X, self.params.min_val_X, self.params.max_val_X, size, self.seed)
    
    def set_inner_products(self, inner_product_l):
        """Set inner products vecb
        """
        self.output_l = inner_product_l
        
    def get_A(self):
        return self.A
    
    def recover(self):
        """Use maximum likelihood to estimate the solution in the matrix form,
        assuming that the solution follows multivariate Gaussian distribution.
        """
        A = self.A
        b = self.output_l
        mean = np.array([self.mean] * self.num_targets)
        scale = np.array([self.scale] * self.num_targets)
        
        try:
            # Create a new model
            model = gp.Model("maximize_likelihood")
            model.setParam('OutputFlag', 0)  # Suppress log output
            
            # Create variable, A * x = b
            x = model.addMVar(shape=A.shape[1], vtype=GRB.CONTINUOUS, lb=self.min_val, ub=self.max_val, name="x")

            if self.use_noise:
                model.addConstr(A @ x - b <= self.noise_lambda, "Ax_b_lambda")
                # Second inequality: Ax - b >= -lambda
                model.addConstr(A @ x - b >= -self.noise_lambda, "Ax_b_neg_lambda")
            else:
                # Add constraint: x + 2 y + 3 z <= 4
                model.addConstr(A @ x == b, "Ax_b")

            # Set objective
            # Maximize likelihood (equivalent to minimizing the quadratic form for Gaussian prior)
            # '@' is the matrix multiplication operator, handles dimension alignment automatically, objective is a scalar
            # objective = (x - mean) @ cov_matrix_inv @ (x - mean)
            objective = (x - mean) @ (x - mean)
            model.setObjective(objective, GRB.MINIMIZE)

            # Optimize model
            model.optimize()

            if model.status == GRB.OPTIMAL:
                # print(f"Optimal value of x: {x.X}")
                # print(f"Optimal value of L1 norm: {sum(t.X)}")
                y_sols = [round(y) for y in x.X]
                # TODO: y_sols has been rounded to the nearest integer, but the solution may not be an integer
                return y_sols
            else:
                print("No optimal solution found")

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")