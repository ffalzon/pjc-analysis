from abc import ABC, abstractmethod
import numpy as np

class RecoveryTemplate(ABC):
    """
    A template class for recovery with virtual methods.
    This class defines the structure for any specific optimization problem.
    """
    
    def __init__(self, params, type = None):
        """Initialize the OptimizationTemplate class with given parameters."""
        self.seed = params.seed if hasattr(params, 'seed') else 1
        np.random.seed(self.seed)
        self.params = params
        self.min_val = params.min_val_Y
        self.max_val = params.max_val_Y
        self.num_targets = params.num_targets
        self.num_invocations = params.num_invocations
        self.mean = params.mean_val_Y
        self.scale = params.scale_val_Y
        self.output_l = []

    @abstractmethod
    def setup(self):
        """
        Abstract method to generate the input set X or perform other setup tasks.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def set_inner_products(self, inner_product_l):
        """
        Abstract method to set the output vector b (inner products or similar).
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def recover(self):
        """
        Abstract method to define the optimization process (e.g., MLE).
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_A(self):
        pass