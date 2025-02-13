import tqdm, csv, os
from math import log2
import sys, time

from attacks import TreeAttacks

if __name__ == "__main__":
    
    num_runs=1
    max_val=1000
    secret_set_size=1000

    target_set_size = 100
    arity = 2**4
    sparsity_ratio = 0.1
    
    print('Target set size:', target_set_size)
    print('Tree arity:', arity)
    print('Sparsity:', sparsity_ratio)

    # Run Search Tree Attack
    start_time = time.time_ns()
    TA = TreeAttacks(arity)
    Y ,target_keys = TA.gen_inputs(max_val, sparsity_ratio, target_set_size, secret_set_size)
    num_queries, total_time = TA.tree_attack(max_val, target_keys, Y)
    end_time = time.time_ns()

    print("Search tree attack test complete!")
    print("Number of queries:", num_queries)
    print("Total runtime:", end_time - start_time, "ns")
