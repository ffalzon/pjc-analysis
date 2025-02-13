import tqdm, csv, os
from math import log2
import sys, time

from attacks import TreeAttacks

if __name__ == "__main__":
    
    num_runs=int(sys.argv[1])
    params_flag=str(sys.argv[2])

    max_val=1000
    secret_set_size=100000
    
    # Set Parameter Options
    if params_flag == "part":
        T_Sizes = [1000, 10000, 100000]
        Arity_Vals = [2**1, 2**2, 2**4,  2**6, 2**8, 2**10]
        Sparsity_Vals_small = [0.2, 0.4, 0.6,  0.8, 1.0]
    elif params_flag == "full":
        T_Sizes = [1000, 10000, 100000, 1000000]
        Arity_Vals = [2**1, 2**2, 2**4,  2**6, 2**8, 2**10]
        Sparsity_Vals_small = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        Sparsity_Vals_10_6 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    else:
        raise Exception("Params type not valid.") 

    start_time = time.time_ns()
    for target_set_size in T_Sizes:
        
        for arity in Arity_Vals:
        
            TA = TreeAttacks(arity)

            if target_set_size == 1000000:
                Sparsity_Vals = Sparsity_Vals_10_6
            else:
                Sparsity_Vals = Sparsity_Vals_small

            for sparsity_ratio in Sparsity_Vals:
                
                total_queries, total_time = 0, 0
                sum_queries, sum_time = 0, 0
                
                print('Target set size:', target_set_size)
                print('Tree arity:', arity)
                print('Sparsity:', sparsity_ratio)

                # Run Search Tree Attack
                for _ in tqdm.tqdm(range(num_runs)):
                    Y ,target_keys = TA.gen_inputs(max_val, sparsity_ratio, target_set_size, 
                                                    secret_set_size)
                    num_queries, total_time = TA.tree_attack(max_val, target_keys, Y)
                    sum_queries += num_queries
                    sum_time += total_time

                # Write results to CSV
                file_path = "results/tree" + "-" +str(log2(arity)) + "-" + str(target_set_size) + "-results.csv"
                file_empty = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
                with open(file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if file_empty:
                        fields = ["max_val", "secret_set_size", "target_set_size", 
                                "sparsity", "num_runs", "arity",  "avg_time", "avg_num_queries"]
                        writer.writerow(fields)
                    results = [[max_val, secret_set_size, target_set_size, 
                                sparsity_ratio, num_runs, arity, 
                                round(sum_time / num_runs), round(sum_queries / num_runs)]]
                    writer.writerows(results)
    
    end_time = time.time_ns()
    print("Total runtime:", end_time - start_time)
