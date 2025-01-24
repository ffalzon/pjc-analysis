import json

class Parameters:
    def __init__(self, min_val_Y, max_val_Y, num_targets, num_invocations=None, 
                 intersection_ratio=None, intersection_size=None, mean_val_Y=None, 
                 scale_val_Y=None, min_val_X=None, max_val_X=None, 
                 mean_val_X=None, scale_val_X=None, cs_l1_multi_factor=None, 
                 cs_l2_multi_factor=None, noise_loc=None, noise_scale=None, 
                 noise_magnitude=None):
        self.min_val_Y = min_val_Y
        self.max_val_Y = max_val_Y
        self.num_targets = num_targets
        self.num_invocations = num_invocations
        self.intersection_ratio = intersection_ratio
        self.intersection_size = intersection_size  # Added parameter
        self.mean_val_Y = mean_val_Y
        self.scale_val_Y = scale_val_Y
        self.min_val_X = min_val_X
        self.max_val_X = max_val_X
        self.mean_val_X = mean_val_X
        self.scale_val_X = scale_val_X
        self.cs_l1_multi_factor = cs_l1_multi_factor
        self.cs_l2_multi_factor = cs_l2_multi_factor
        self.noise_loc = noise_loc
        self.noise_scale = noise_scale
        self.noise_magnitude = noise_magnitude

    def __repr__(self):
        return (f"Parameters(min_val_Y={self.min_val_Y}, max_val_Y={self.max_val_Y}, "
                f"min_val_X={self.min_val_X}, max_val_X={self.max_val_X}, "
                f"num_targets={self.num_targets}, num_invocations={self.num_invocations}, "
                f"intersection_ratio={self.intersection_ratio}, intersection_size={self.intersection_size}, "  # Updated __repr__
                f"mean_val_Y={self.mean_val_Y}, scale_val_Y={self.scale_val_Y}, "
                f"mean_val_X={self.mean_val_X}, scale_val_X={self.scale_val_X}, "
                f"cs_l1_multi_factor={self.cs_l1_multi_factor}, "
                f"cs_l2_multi_factor={self.cs_l2_multi_factor}, noise_loc={self.noise_loc}, "
                f"noise_scale={self.noise_scale}, noise_magnitude={self.noise_magnitude})")

    @classmethod
    def from_dict(cls, data):
        return cls(
            data['min_val_Y'],
            data['max_val_Y'],
            data['num_targets'],
            data.get('num_invocations'),
            data.get('intersection_ratio'),
            data.get('intersection_size'),  # Added parameter in from_dict
            data.get('mean_val_Y'),  
            data.get('scale_val_Y'),  
            data.get('min_val_X'),  
            data.get('max_val_X'),  
            data.get('mean_val_X'),  
            data.get('scale_val_X'),
            data.get('cs_l1_multi_factor'), 
            data.get('cs_l2_multi_factor'), 
            data.get('noise_loc'), 
            data.get('noise_scale'), 
            data.get('noise_magnitude')
        )

    @classmethod
    def load_multiple(cls, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return [cls.from_dict(d) for d in data['param_sets']]

    def is_valid(self):
        return (self.min_val_Y < self.max_val_Y and 
                (self.min_val_X is None or self.max_val_X is None or self.min_val_X < self.max_val_X) and 
                self.num_targets > 0)

    def scale_targets(self, factor):
        self.num_targets *= factor

# Usage example
if __name__ == "__main__":
    params_list = Parameters.load_multiple('params.json')
    for params in params_list:
        print(params)
        print(f"Is valid: {params.is_valid()}")
