import math
import numpy as np

from ortools.linear_solver import pywraplp

import gurobipy as gp
from gurobipy import GRB

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import Parameters
from src.recovery_template import RecoveryTemplate
from src.util import Util

class CompressedSensing(RecoveryTemplate):
    """CompressedSensing class is used to test the compressed sensing model."""

    def __init__(self, params: Parameters, type: str, use_noise: bool = False, use_ortools: bool = False):
        """Initialize the CompressedSensing class with given parameters."""
        super().__init__(params)
        self.params = params
        self.type = type
        print(f"Compressed sensing type: {self.type}")
        self.A = None
        self.use_noise = use_noise
        self.intersection_size = self.params.intersection_size

        self.noise_lambda = self.params.noise_magnitude
        self.noise_scale = self.params.noise_scale
        self.noise_loc = self.params.noise_loc
        self.use_ortools = use_ortools

    def setup(self):
        n = self.num_targets
        k = self.intersection_size
        multi_factor = math.log(n / k, 2)
                
        """setup the matrix A
        """
        if self.type == "l1":
            multi_factor = multi_factor * self.params.cs_l1_multi_factor
            print(f"multi_factor: {multi_factor}")
            m = round(k * multi_factor)
            print(f"m={m}")            
            self.num_invocations = m
            self.gen_binary_RIP_l1_A()
        elif self.type == "l2" or self.type == "lasso":
            multi_factor = multi_factor * self.params.cs_l2_multi_factor
            print(f"multi_factor: {multi_factor}")
            m = round(k * multi_factor)
            self.num_invocations = m
            self.gen_RIP_l2_A()
        else:
            print("Invalid norm type")

    def set_inner_products(self, inner_product_l):
        """Set inner products vecb
        """
        self.output_l = inner_product_l

    def get_A(self):
        """Get the matrix A
        """
        # print(f"Matrix A: {self.A}")  
        return self.A
    
    def recover(self):
        if self.type == "l1":
            return self.minimize_l1_norm()
        elif self.type == "l2":
            if self.use_noise:
                # return self.dantzig_selector()
                if self.use_ortools:
                    return self.dantzig_selector_ortools()
                else:
                    return self.dantzig_selector()                
            else:
                return self.minimize_l1_norm()
        elif self.type == "lasso":
            if self.use_noise:
                return self.lasso()
            else:
                return self.minimize_l1_norm()
        else:
            print("Invalid norm type")
            return None

    @staticmethod
    def compute_theoretical_expansion_fail_prob(self, n: int, k: int, epsilon: float):
        """Compute the upperbound on the failure probability of the expander graph not satisfying the expansion property
        Using the theorem listed in the paper
        """
        d = 50 #TODO: depend on n

        m = math.ceil(d * k * (math.e ** (1/epsilon - 1)))

        
        assert(m <= n and k <= n and d <= m)

        first_term = (n * math.e) / k
    
        # Compute the second term: ((1-epsilon) * d * k * e^(1/epsilon - 1) / m)^(epsilon * d)
        second_term = ((1 - epsilon) * d * k * math.exp(1/epsilon - 1) / m) ** (epsilon * d)
    
        # Compute the final bound x
        x = first_term * second_term

        failure_prob = x / (1 - x)
        print(f"failure_prob: {failure_prob}")
        return failure_prob

    def compute_expansion_fail_prob(self, n: int, k: int, epsilon: float):
        """Compute the upperbound on the failure probability of the expander graph not satisfying the expansion property
        Using the theorem listed in the paper
        """
        m_min = math.ceil(k * math.log(n / k))
        m_max = 15 * m_min
        failure_prob_min = 1
        print(f"m_min: {m_min}, m_max: {m_max}")
        m_result = m_min
        d_result = 1
        x_log_min = 1
        for m in range(m_min, m_max, 10):
            for d in range(1, m, 10):
                assert(m <= n and k <= n and d <= m)
                first_term_log = math.log(n * math.e) - math.log(k)
                
                # Compute the second term: ((1-epsilon) * d * k * e^(1/epsilon - 1) / m)^(epsilon * d)
                second_term_log = (epsilon * d) * (math.log(1 - epsilon) + math.log(d) + math.log(k) + (1/epsilon - 1) - math.log(m))
    
                # Compute the final bound x
                x_log = first_term_log + second_term_log

                # failure_prob = x / (1 - x)
                if x_log < x_log_min:
                    x_log_min = x_log
                    d_result = d
                    m_result = m

        x = math.exp(x_log_min)
        failure_prob = x / (1 - x)
        print(f"failure_prob: {failure_prob}")
        print(f"x_log_min: {x_log_min}")
        print(f"m: {m_result}, d: {d_result}")

    def gen_randomized_unbalanced_expander_graph(self, n: int, m: int, d: int):
        """Generate a randomized unbalanced expander graph
        U: Left set of the bipartite graph, |U| = n
        V: Right set of the bipartite graph, |V| = m
        d: Left degree
        Under condition m <= n
        """
        assert(m <= n)
        U = np.array(range(n))
        V = np.array(range(m))

        # Generate the adjacency list sample without replacement so that left d degree is satisfied
        adj_l = [np.random.choice(V, size=d, replace=False) for _ in range(n)]
        # print(adj_l)

        # Convert the adjacency list to a matrix representation without explicit loops
        rows = np.concatenate(adj_l)
        cols = np.repeat(np.arange(n), d)

        # Create the adjacency matrix using sparse matrix logic
        adj_matrix = np.zeros((m, n), dtype=int)
        adj_matrix[rows, cols] = 1

        # print(adj_matrix)  
        return adj_matrix

    def gen_binary_RIP_l1_A(self):
        """Implement RIP matrix using binary matrix (adjacency matrix) of an randomized unbalanced expander graph
        Adaptive attack
        """
        # print("Binary RIP using randomized unbalanced expander graph")
        n = self.num_targets
        m = self.num_invocations
        # d = round(self.num_invocations / 2) ## TODO: Need to update this
        # d = 50 ## TODO: depend on n
        d = math.ceil(m / 2)
        print(f"n={n}, m={m}, d={d}")
        self.A = self.gen_randomized_unbalanced_expander_graph(n, m, d)

    def gen_RIP_l2_A(self):
        """Implement RIP matrix using elements drawn from a normal distribution
        Adaptive and passive attack
        """
        n = self.num_targets
        m = self.num_invocations

        size = (m, n)

        print(f"self.params.mean_val_X: {self.params.mean_val_X}, self.params.scale_val_X: {self.params.scale_val_X}, self.params.min_val_X: {self.params.min_val_X}, self.params.max_val_X: {self.params.max_val_X}")
    
        self.A = Util.gen_normal_data(self.params.mean_val_X, self.params.scale_val_X, self.params.min_val_X, self.params.max_val_X, size, self.seed, round_to_int=False)

        # round it to integer
        # self.A = self.A.round().astype(int)

    def minimize_l1_norm(self):
        """Use Gurobi to minimize the l1 norm of x.

        Raises:
            gp.GurobiError: If there is an error in the Gurobi model.
            AttributeError: If there is an attribute error.
        """
        A = self.A
        b = np.array(self.output_l)

        print(self.noise_lambda)
        
        try:
            # Create a new model
            model = gp.Model("minimize_l1_norm")
            model.setParam('OutputFlag', 0)  # Suppress log output
            
            # Create variable, A * x = b
            x = model.addMVar(shape=A.shape[1], vtype=GRB.CONTINUOUS, lb=self.min_val, ub=self.max_val, name="x")

            # auxiliary variables for L1 norm (one for each element of x

            t = model.addMVar(shape=A.shape[1], vtype=GRB.CONTINUOUS, lb=self.min_val, ub=self.max_val, name="t")

            # Add constraint: x + 2 y + 3 z <= 4

            if self.use_noise:
                # First inequality: Ax - b <= lambda
                model.addConstr(A @ x - b <= self.noise_lambda, "Ax_b_lambda")
                # Second inequality: Ax - b >= -lambda
                model.addConstr(A @ x - b >= -self.noise_lambda, "Ax_b_neg_lambda")
            else:
                model.addConstr(A @ x == b, "Ax_b")

            # Add constraints for the auxiliary variables to represent the L1 norm
            model.addConstr(x <= t, name="x_leq_t") 
            model.addConstr(-x <= t, name="neg_x_leq_t")

            # Set objective to minimize the sum of auxiliary variables t
            model.setObjective(gp.quicksum(t), GRB.MINIMIZE)

            # Optimize model
            model.optimize()

            if model.status == GRB.OPTIMAL:
                # print(f"Optimal value of x: {x.X}")
                # print(f"Optimal value of L1 norm: {sum(t.X)}")
                y_sols = [round(y) for y in x.X]
                return y_sols
            else:
                print("No optimal solution found")


        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")

        except AttributeError:
            print("Encountered an attribute error")

    def dantzig_selector_ortools(self):
        try:
            # Create the solver using the GLOP backend for linear programming
            solver = pywraplp.Solver.CreateSolver('GLOP')

            if not solver:
                raise Exception("Solver creation failed.")

            # Variables
            s_l = [solver.NumVar(self.min_val, self.max_val, f's[{i}]') for i in range(self.num_targets)]
            u_l = [solver.NumVar(0.0, self.max_val, f'u[{i}]') for i in range(self.num_targets)]

            # Constants
            sigma_a = self.params.scale_val_X
            sigma_e = self.noise_scale
            m = self.num_invocations
            n = self.num_targets

            b = np.array(self.output_l)

            # Convert to numpy arrays for easier matrix operations
            A = np.array(self.A)

            # Constraint: A.T @ (b - A @ s_l)
            s_l_array = np.array(s_l)
            temp = A.T @ (b - A @ s_l_array)

            left_bound = -sigma_a * sigma_e * np.sqrt(2 * m * np.log2(n))
            right_bound = -left_bound

            # Add constraints for temp
            for i in range(temp.shape[0]):
                solver.Add(temp[i] <= right_bound)
                solver.Add(temp[i] >= left_bound)

            # Add constraints for u_l and s_l
            for i in range(self.num_targets):
                solver.Add(u_l[i] >= s_l[i])
                solver.Add(u_l[i] >= -s_l[i])

            # Objective: Minimize the sum of u_l
            solver.Minimize(solver.Sum(u_l))

            # Solve the problem
            status = solver.Solve()

            # Check the result
            if status == pywraplp.Solver.OPTIMAL:
                y_sols = [round(s_l[i].solution_value()) for i in range(self.num_targets)]
                return y_sols
            else:
                print('The problem does not have an optimal solution.')

        except Exception as e:
            print(f'Error: {e}')

    def dantzig_selector(self):
        """Implement compressed sensing for sparse secret
        Using Dantzig selector minimizes the l1 norm of the solution
        Raises:
            gp.GurobiError: If there is an error in the Gurobi model.
            AttributeError: If there is an attribute error.
        """
        try:
            model = gp.Model("DantzigSelector")
            model.setParam('OutputFlag', 0)  # Suppress log output

            # Variables
            s_l = model.addVars(self.num_targets, lb=self.min_val, ub=self.max_val, name="s")
            u_l = model.addVars(self.num_targets, lb=0.0, ub=self.max_val, name="u")

            # Constants
            sigma_a = self.params.scale_val_X
            sigma_e = self.noise_scale

            m = self.num_invocations
            n = self.num_targets

            A = self.A
            b = np.array(self.output_l)

            # Constraint: A.T @ (b - A @ s_l)
            s_l_array = np.array([s_l[i] for i in range(self.num_targets)])
            temp = A.T @ (b - A @ s_l_array)

            left_bound = -sigma_a * sigma_e * np.sqrt(2 * m * np.log2(n))
            right_bound = -left_bound

            # Add constraints for temp
            for i in range(len(temp)):
                model.addConstr(temp[i] <= right_bound)
                model.addConstr(temp[i] >= left_bound)

            # Add constraints for u_l and s_l
            for i in range(self.num_targets):
                model.addConstr(u_l[i] >= s_l[i])
                model.addConstr(u_l[i] >= -s_l[i])

            # Objective: Minimize the sum of u_l
            
            model.setObjective(gp.quicksum(u_l[i] for i in range(self.num_targets)), GRB.MINIMIZE)

            # Optimize the model
            model.optimize()

            # Check the result
            if model.status == GRB.OPTIMAL:
                y_sols = [round(s_l[i].x) for i in range(self.num_targets)]
                return y_sols
            else:
                print('The problem does not have an optimal solution.')
        except gp.GurobiError as e:
            print(f'Gurobi Error: {e}')
        except AttributeError as e:
            print(f'Attribute Error: {e}')

    def lasso(self):
        """Lasso for sparse recovery
        Raises:
            gp.GurobiError: If there is an error in the Gurobi model.
            AttributeError: If there is an attribute error.
        """
        try:
            model = gp.Model("Lasso")
            model.setParam('OutputFlag', 0)  # Suppress log output

            # Variables
            s_l = model.addVars(self.num_targets, lb=self.min_val, ub=self.max_val, name="s")
            u_l = model.addVars(self.num_targets, lb=0.0, ub=self.max_val, name="u")

            # Constants
            A = self.A
            b = np.array(self.output_l)

            # Add constraints for u_l and s_l
            for i in range(self.num_targets):
                model.addConstr(u_l[i] >= s_l[i])
                model.addConstr(u_l[i] >= -s_l[i])

            # Objective: Minimize the sum of squared residuals + lambda * sum of u_l
            residuals = b - A @ np.array([s_l[i] for i in range(self.num_targets)])
            lambda_param = 1
            model.setObjective(gp.quicksum(residuals[i] * residuals[i] for i in range(self.num_invocations)) + lambda_param * gp.quicksum(u_l[i] for i in range(self.num_targets)), GRB.MINIMIZE)

            # Optimize the model
            model.optimize()

            # Check the result
            if model.status == GRB.OPTIMAL:
                y_sols = [round(s_l[i].x) for i in range(self.num_targets)]
                return y_sols
            else:
                print('The problem does not have an optimal solution.')
        except gp.GurobiError as e:
            print(f'Gurobi Error: {e}')
        except AttributeError as e:
            print(f'Attribute Error: {e}')

    def basic_linear_regression(self):
        """Linear regression method, OLS
        """
        # self.gen_input_sets()
        # print(self.X)
        A = self.A
        b = np.array(self.output_l)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        # print(f"U: {U}, S: {S}, Vt: {Vt}")

        y_hat = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

        # print(f"y_hat: {y_hat}")

        y_sols = [round(y) for y in y_hat]
        print(f"Solution by basic linear regression: {y_sols}")
        return y_sols