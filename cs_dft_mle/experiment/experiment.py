import numpy as np
import tqdm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import CompressedSensing
from src import DiscreteFourier
from src import MLE
from src import Util
from config import Parameters
import time

class Experiment: 
    def __init__(self, params: Parameters, recovery_type: str, use_noise: bool = False, use_ortools: bool = False):
        # Fix the random seed for testing
        # self.seed = 1
        # np.random.seed(self.seed)

        current_time_ns = time.time_ns()
        self.seed = current_time_ns % (2**32)
        np.random.seed(self.seed)

        self.params = params
        self.min_val = params.min_val_Y
        self.max_val = params.max_val_Y
        # min_val inclusive, max_val exclusive
        self.min_val_range = params.min_val_Y
        self.max_val_range = params.max_val_Y + 1
        self.num_targets = params.num_targets
        self.intersection_size = params.intersection_size
        self.num_invocations = params.num_invocations
        self.intersection_ratio = params.intersection_ratio
        self.recovery_type = recovery_type

        self.output_l = []
        self.use_noise = use_noise
        self.noise_lambda = self.params.noise_magnitude
        self.noise_scale = self.params.noise_scale

        self.use_ortools = use_ortools

    def compute_inner_products(self, A):
        """
        Compute the inner product of matrix A and Y_victim_vals taking into account the condition
        where Y_victim_keys == 0, the corresponding Y_victim_vals should be treated as 0.
        
        Args:
        - A: A NumPy array of shape (m, self.num_targets), representing the associated values of the keys for m runs.
        
        Returns:
        - inner_products: A NumPy array of shape (m,), where each element is the inner product for a particular run.
        """
        # Create a mask where Y_victim_keys == 0
        self.masked_Y_victim_vals = np.where(self.Y_victim_keys != 0, self.Y_victim_vals, 0)
        
        # Compute the inner product between A and the masked Y_victim_vals
        inner_products = np.dot(A, self.masked_Y_victim_vals)

        if self.use_noise:
            # Add noise to the inner products
            # print(f"Adding noise!!!!!!!!!!!noise_scale{self.noise_scale}")
            noise = Util.gen_normal_data(0, self.noise_scale, -self.noise_lambda, self.noise_lambda, inner_products.shape[0], self.seed)
            inner_products += noise
        
        return inner_products
        
    def get_intersection_size(self):
        return self.intersection_size

    def gen_uniform_set_X(self):
        # X is the adversary's input
        # In each iteration, selects an input unfomly at random

        for i in range(self.num_invocations):
            if self.use_normal_data:
                X = self.truncnorm_dist.rvs(size=self.num_targets)
                X = X.round().astype(int)
                
            else:
                X = [np.random.choice(range(self.min_val_range, self.max_val_range)) for _ in range(self.num_targets)]
    
    def gen_target_set_X_keys(self):
        """For simplicity, we use a list of integers as the keys (e.g., identifiers), starting from 1.
        """
        self.X_target_keys = np.array(range(1, self.num_targets + 1))


    def gen_victim_set_Y(self):
        """Generate a random target set (associated values), can be zero or non-zero. 
        Since we only consider recovering the associated values of the intersection set, to simplify the problem, 
        we only generate the victim set of size the same as the target set.
        If zero, then it is not in the intersection or the associated value is zero
        Else, it is in the intersection. 
        The aim is to recover the associated values.
        """
        Y_victim_keys = np.array(range(1, self.num_targets + 1))

        if self.recovery_type == 'mle':
            self.mean_val_Y = self.params.mean_val_Y
            self.scale_val_Y = self.params.scale_val_Y
            Y_victim_vals = Util.gen_normal_data(self.mean_val_Y, self.scale_val_Y, self.min_val, self.max_val, self.num_targets, self.seed)
            # print(f"Y_victim_vals = {Y_victim_vals}")
        else:
            Y_victim_vals = np.array([np.random.choice(range(self.min_val_range, self.max_val_range)) for _ in range(self.num_targets)])
            # print(Y_victim_vals)
            # zero_ratio = 1 - self.intersection_ratio
            # num_zeros = round(zero_ratio * self.num_targets)

            num_zeros = self.num_targets - self.intersection_size
            zero_indices = np.random.choice(self.num_targets, num_zeros, replace=False)
            Y_victim_keys[zero_indices] = 0

        self.Y_victim_vals = Y_victim_vals
        self.Y_victim_keys = Y_victim_keys

    @staticmethod
    def compute_l1_loss(Y_true, Y_pred) -> float:
        return np.abs(Y_true - Y_pred).mean()

    @staticmethod
    def compute_mean_squared_error(Y_true, Y_pred) -> float:
        # Auxiliary function to compute the mean squared error
        MSE = np.square(np.subtract(Y_true, Y_pred)).mean()
        return MSE

    def compute_solution_loss(self, Y_pred) -> None:
        # MSE = self.compute_mean_squared_error(Y_pred, self.masked_Y_victim_vals)
        # print(f"MSE = {MSE}")
        l1_loss = self.compute_l1_loss(Y_pred, self.masked_Y_victim_vals)
        print(f"l1_loss = {l1_loss}")
        return l1_loss

    def run(self) -> None:
        self.gen_victim_set_Y()
        recovery = None
        if self.recovery_type == 'compressed_sensing_l1':
            recovery = CompressedSensing(self.params, 'l1', self.use_noise)
        elif self.recovery_type == 'compressed_sensing_l2':
            recovery = CompressedSensing(self.params, 'l2', self.use_noise, self.use_ortools)
        elif self.recovery_type == 'lasso':
            recovery = CompressedSensing(self.params, 'lasso', self.use_noise)            
        elif self.recovery_type == 'discrete_fourier':
            recovery = DiscreteFourier(self.params, self.use_noise)
        elif self.recovery_type == 'mle':
            recovery = MLE(self.params, self.use_noise)
        else:
            print(f"Unknown recovery type: {self.recovery_type}")
            return
        
        if self.recovery_type == 'discrete_fourier':
            intersection_size = self.get_intersection_size()
            recovery.setup(intersection_size)
        else:
            recovery.setup()  

        A = recovery.get_A()
        self.output_l = self.compute_inner_products(A)
        recovery.set_inner_products(self.output_l)

        start_time = time.time_ns()
        sols = recovery.recover()
        end_time = time.time_ns()
        recovery_time = end_time - start_time

        l1_loss = self.compute_solution_loss(sols)
        return l1_loss, recovery_time

if __name__ == "__main__":
    # Choose the recovery type: 
    # recovery_type = "compressed_sensing_l1"
    # recovery_type = "compressed_sensing_l2"
    # recovery_type = "discrete_fourier"
    # recovery_type = "mle"
    # recovery_type = "lasso"

    use_noise_l = [True]
    use_ortools = True # Need to set to false to use Gurobi
    num_experiments = 5 # default 10, 5 is for testing

    for use_noise in use_noise_l:
        if use_noise:
            recovery_type_l = ["discrete_fourier", "compressed_sensing_l2"]
        else:
            recovery_type_l = ["mle", "discrete_fourier", "compressed_sensing_l1", "compressed_sensing_l2"]

        for recovery_type in recovery_type_l:
            if recovery_type == "mle":
                params_list = Parameters.load_multiple(f"../config/generated_params_non_sparse.json")
            else:
                params_list = Parameters.load_multiple('../config/generated_params_sparse.json')
            
            for params in params_list:
                print(params)
                # Write results to CSV
                field_names = ["min_val_Y", "max_val_Y", "num_targets", "intersection_ratio", "intersection_size", "num_invocations", "mean_val_Y", "scale_val_Y", 
                                "min_val_X", "max_val_X", "mean_val_X", "scale_val_X", "noise_loc", "noise_magnitude", "noise_scale", "recovery_type", "num_experiments", "avg_l1_loss", "avg_recovery_time"]        
                
                print(f"Is valid: {params.is_valid()}")

                l1_loss_total = 0
                recovery_time_total = 0

                for _ in tqdm.tqdm(range(num_experiments)):
                    experiment = Experiment(params, recovery_type, use_noise, use_ortools)
                    l1_loss, recovery_time = experiment.run()
                    l1_loss_total += l1_loss
                    recovery_time_total += recovery_time

                avg_recovery_time = round(recovery_time_total / num_experiments) # in ns
                avg_l1_loss = round(l1_loss_total / num_experiments, 2) # round to precision 2
                
                results = [[params.min_val_Y, params.max_val_Y, params.num_targets, params.intersection_ratio, params.intersection_size, params.num_invocations, params.mean_val_Y, params.scale_val_Y,
                        params.min_val_X, params.max_val_X, params.mean_val_X, params.scale_val_X, params.noise_loc, params.noise_magnitude, params.noise_scale,recovery_type, 
                        num_experiments, avg_l1_loss, avg_recovery_time]]
          
                result_dir = "results"
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                if use_noise:
                    result_file_name = f"{result_dir}/results_{recovery_type}_noise.csv"
                else:
                    result_file_name = f"{result_dir}/results_{recovery_type}.csv"
                    
                Util.write_results_to_csv(result_file_name, field_names, results)
