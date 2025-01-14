import tqdm, csv, os
from math import log2
import sys

from attacks import TreeAttacks


if __name__ == "__main__":
    
    attack=sys.argv[1]
    max_val=int(sys.argv[2])    
    secret_set_size=int(sys.argv[3])
    num_runs=int(sys.argv[4])
    
    for target_set_size in [10000]:
        
        for arity in [2**6, 2**7,  2**8, 2**9, 2**10]:
        
            TA = TreeAttacks(arity)
            
            for sparsity_ratio in [0.01, 0.02]:
                
                total_queries, total_time = 0, 0
                sum_queries, sum_time = 0, 0
                
                print('Target set size:', target_set_size)
                if attack == "tree":
                    print('Tree arity:', arity)
                print('Sparsity:', sparsity_ratio)


                for _ in tqdm.tqdm(range(num_runs)):
                    Y ,target_keys = TA.gen_inputs(max_val, sparsity_ratio, target_set_size, 
                                                    secret_set_size)
                    if attack == "basic":
                        num_queries, total_time = TA.basic_attack(max_val, target_keys, Y)
                    elif attack == "tree":
                        num_queries, total_time = TA.tree_attack(max_val, target_keys, Y)
                    sum_queries += num_queries
                    sum_time += total_time


                # Write results to CSV
                if attack == "basic":
                    file_path = "results/basic" + "-" + str(target_set_size) + "-results.csv"
                elif attack == "tree":
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
                
