import json
import itertools
import math

def generate_parameters_sparse(min_val_Y_l, max_val_Y_l, scale_val_Y_l, num_targets_intersection_pairs, min_val_X_l, max_val_X_l, scale_val_X_l, cs_l1_multi_factor_l, cs_l2_multi_factor_l, noise_loc_l, noise_scale_l, noise_magnitude_l):
    param_sets = []

    # Ensure that lists of Y, X parameters, noise, and (num_targets, intersection_size) pairs are of the same length
    for (min_val_Y, max_val_Y, scale_val_Y), (min_val_X, max_val_X, scale_val_X), (noise_loc, noise_scale, noise_magnitude) in zip(
        zip(min_val_Y_l, max_val_Y_l, scale_val_Y_l), 
        zip(min_val_X_l, max_val_X_l, scale_val_X_l), 
        zip(noise_loc_l, noise_scale_l, noise_magnitude_l)
    ):
        for num_targets, intersection_size in num_targets_intersection_pairs:
            if min_val_Y < max_val_Y and min_val_X < max_val_X:
                param_set = {
                    "min_val_Y": min_val_Y,
                    "max_val_Y": max_val_Y,
                    "scale_val_Y": scale_val_Y,
                    "num_targets": num_targets,
                    "intersection_size": intersection_size,
                    "min_val_X": min_val_X,
                    "max_val_X": max_val_X,
                    "mean_val_X": round((min_val_X + max_val_X) / 2, 2),
                    "scale_val_X": scale_val_X,
                    "cs_l1_multi_factor": cs_l1_multi_factor_l[0],  # Assuming cs_l1_multi_factor and cs_l2_multi_factor are constants
                    "cs_l2_multi_factor": cs_l2_multi_factor_l[0],
                    "noise_loc": noise_loc,
                    "noise_scale": noise_scale,
                    "noise_magnitude": noise_magnitude   
                }
                param_sets.append(param_set)
    
    return param_sets


def generate_parameters_mle(min_val_Y_l, max_val_Y_l, scale_val_Y_l, num_targets_l, num_invocations_ratio_l, min_val_X_l, max_val_X_l, scale_val_X_l, noise_loc_l, noise_scale_l, noise_magnitude_l):
    """Generate parameter sets for the MLE model.
    Also the auxiliary information for the data distribution.
    """

    param_sets = []
    intersection_ratio = 1.0  # MLE does not use intersection_ratio

    # Ensure that lists of Y, X parameters, and noise are of the same length
    for (min_val_Y, max_val_Y, scale_val_Y), (min_val_X, max_val_X, scale_val_X), (noise_loc, noise_scale, noise_magnitude) in zip(
        zip(min_val_Y_l, max_val_Y_l, scale_val_Y_l), 
        zip(min_val_X_l, max_val_X_l, scale_val_X_l), 
        zip(noise_loc_l, noise_scale_l, noise_magnitude_l)
    ):
        for num_targets in num_targets_l:
            for num_invocations_ratio in num_invocations_ratio_l:
                if min_val_Y < max_val_Y and min_val_X < max_val_X:
                    num_invocations = math.ceil(num_targets * num_invocations_ratio)
                    param_set = {
                        "min_val_Y": min_val_Y,
                        "max_val_Y": max_val_Y,
                        "scale_val_Y": scale_val_Y,
                        "num_targets": num_targets,
                        "num_invocations": num_invocations,
                        "mean_val_Y": round((min_val_Y + max_val_Y) / 2, 2),
                        "min_val_X": min_val_X,
                        "max_val_X": max_val_X,
                        "mean_val_X": round((min_val_X + max_val_X) / 2, 2),
                        "scale_val_X": scale_val_X,
                        "noise_loc": noise_loc,
                        "noise_scale": noise_scale,
                        "noise_magnitude": noise_magnitude
                    }
                    param_sets.append(param_set)
            
    return param_sets

def frange_inclusive(start, stop, step):
    """A range function that accepts float step values."""
    while start <= stop:
        yield start
        start += step

def save_parameters_to_json(param_sets, filename):
    data = {"param_sets": param_sets}
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    num_targets_intersection_pairs = [[100, 10]]
    num_invocations_ratio_l = [0.95]

    # Define ranges and steps for parameter values
    min_val_Y_l = [-100]
    max_val_Y_l = [100]
    scale_val_Y_l = [500]
    
    min_val_X_l = [-100]
    max_val_X_l = [100]
    scale_val_X_l = [50]

    noise_magnitude_l = [10]
    noise_loc_l = [0]
    noise_scale_l = [0.25] 

    cs_l1_multi_factor_l = [2]
    cs_l2_multi_factor_l = [2]

    type_l = ["sparse", "non_sparse"]
    
    # Generate a list of parameter sets
    for type in type_l:
        if type == "sparse":
            param_sets = generate_parameters_sparse(min_val_Y_l, max_val_Y_l, scale_val_Y_l, num_targets_intersection_pairs,
                                                    min_val_X_l, max_val_X_l, scale_val_X_l, cs_l1_multi_factor_l, cs_l2_multi_factor_l, noise_loc_l, noise_scale_l, noise_magnitude_l)
        elif type == "non_sparse":
            param_sets = generate_parameters_mle(min_val_Y_l, max_val_Y_l, scale_val_Y_l, [pair[0] for pair in num_targets_intersection_pairs], num_invocations_ratio_l, 
                                                min_val_X_l, max_val_X_l, scale_val_X_l, noise_loc_l, noise_scale_l, noise_magnitude_l)
        else:
            raise ValueError(f"Invalid type: {type}")
        
        # Save the parameter sets to a JSON file
        file_json_name = f"generated_params_{type}.json"
        save_parameters_to_json(param_sets, file_json_name)
        print(f"Generated {len(param_sets)} parameter sets and saved to {file_json_name}")
