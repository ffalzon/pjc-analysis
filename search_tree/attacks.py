import random 
from helper import HelperFunctions
import time
import sys

sys.set_int_max_str_digits(0)

class TreeAttacks():
    
    def __init__(self, arity) -> None:
        self.arity = arity
        self.helper = HelperFunctions(self.arity)

    def gen_inputs(self, max_val, sparsity_ratio, target_set_size, secret_set_size):
        """
        Generates the victim's input to PJC, Y, and the Adversary's target set, T.
        -----
        Inputs: Parameters for inputs, including the maximum value of Y, the size of the 
            intersection, the size of the target set, and the size of the victim's set
        Returns: The victim's secret key-value set Y and the adversary's target set of keys T.
        """
        if not (0 <= sparsity_ratio <= 1):
            raise ValueError("Sparsity ratio must be between 0 and 1.")
        if target_set_size > secret_set_size and sparsity_ratio > secret_set_size/target_set_size:
            raise ValueError("Sparsity ratio not feasible.")
        
        random.seed()
        key_universe_size = secret_set_size + target_set_size
        
        # Initialize secret set Y
        Y_keys = random.sample(range(key_universe_size), secret_set_size)
        Y_vals = [random.choice(range(1, max_val + 1)) for _ in range(secret_set_size)]
        Y = dict(zip(Y_keys, Y_vals))

        # Initialize target set T
        intersection_size =  round(target_set_size * sparsity_ratio)
        shared_keys = random.sample(Y_keys, intersection_size)
        remaining_items = list(set(range(key_universe_size)) - set(Y_keys))
        target_keys = shared_keys + random.sample(remaining_items, target_set_size - intersection_size)
        target_keys = random.sample(target_keys, target_set_size)
        return Y, target_keys


    def basic_attack(self, max_val, target_keys, Y):
        """
        Implements our basic attack which recovers Y in one query.
        -----
        Inputs: The maximum value of the victim's set, the set of target keys, 
            and the secret set Y. N.B. The set Y is only used as as input to the 
            PJC functionality and not used directly in the attack.
        Returns: The number of queries needed to complete attack (always one) 
            and the attack run time, OR raises an error.
        """
        PJC_time = 0
        start_time = time.time_ns()
        max_digits = len(str(max_val))

        # Compute adversarial input set X
        target_set_size = len(target_keys)
        target_vals = [10**(i*max_digits) for i in range(target_set_size)]
        X = dict(zip(target_keys, target_vals))
        
        # Execute PJC Protocol
        t0 = time.time_ns()
        ans = self.helper.inner_product(X,Y)
        t1 = time.time_ns()
        PJC_time += t1 - t0 
        
        # Carry out basic attack
        ans = str(ans)
        padding_needed = (max_digits - len(ans) % max_digits) % max_digits
        ans = '0' * padding_needed + ans
        solns = [int(ans[i:i+max_digits]) for i in range(0, len(ans), max_digits)]

        # Process solution
        Solution = {} 
        sols_len = len(solns)
        for i in range(sols_len):
            val = solns[i]
            if val != 0:
                adv_val = 10**((sols_len-i-1)*max_digits)
                Solution.update({(key, val) for key in X if X[key] == adv_val})
        end_time = time.time_ns()

        # Check answer
        common_keys = [value for value in target_keys if value in Y.keys()]
        Check = ({key: Y[key] for key in common_keys})
        if Solution != Check:
            raise ValueError("Incorrect Solution.")

        total_time = end_time - start_time
        return 1, total_time - PJC_time


    def tree_attack(self, max_val, target_keys, Y):
        """
        Implements our search tree attack.
        -----
        Inputs: The maximum value of the victim's set, the set of target keys, 
            and the secret set Y. N.B. The set Y is only used as as input to the 
            PJC functionality and not used directly in the attack.
        Returns: The number of queries needed to complete attack and the attack 
            run time, OR raises an error.
        """
        PJC_time = 0
        start_time = time.time_ns()
        Solution = {} # Stores solution
        
        # Initialize queue
        target_set_size = len(target_keys)
        indexes = [i for i in range(target_set_size)] 
        queue = [indexes]

        num_queries = 0
        while queue != []: 
            A = queue.pop()
            max_coeff = self.helper.compute_max_coeff(1, self.arity, A)
            max_digits = len(str(max_val * max_coeff))
            
            # Define adversarial input set X
            X = {}
            for j, subarray in enumerate(self.helper.compute_chunks(self.arity, A)):
                if len(subarray) == 0:
                    break
                # Shifting i because python indexing starts at 0
                target_vals = [10 ** (j * max_digits) for i in subarray]
                X_vals = [target_keys[i] for i in subarray]
                X.update(dict(zip(X_vals, target_vals)))
                
            # Execute PJC protocol
            t0 = time.time_ns()
            ans = self.helper.inner_product(X,Y)
            t1 = time.time_ns()
            PJC_time += t1 - t0 
            num_queries += 1

            # Carry out basic attack
            padding_needed = (max_digits - len(str(ans)) % max_digits) % max_digits
            ans = '0' * padding_needed + str(ans)
            solns = [int(ans[i: i + max_digits]) for i in range(0, len(str(ans)), max_digits)]
            solns = [0] * (self.arity - len(solns)) + solns

            for j, subarray in enumerate(self.helper.compute_chunks(self.arity, A)):
                sol = solns[-(j+1)]
                if len(subarray) == 1 and sol != 0:
                    for i in subarray:
                        Solution[target_keys[i]] = sol
                elif len(subarray) > 1 and sol != 0:
                    queue.append(subarray)


        end_time = time.time_ns()

        # Check answer
        common_keys = [value for value in target_keys if value in Y.keys()]
        Check = {key: Y[key] for key in common_keys}
        if Solution != Check:
            print(Solution)
            print(Check)
            # Extract (key, value) pairs from Check where the value is not in Solution
            different_items = [(key, value) for key, value in Check.items() if value not in Solution.values()]
            print(different_items)
            different_items2 = [(key, value) for key, value in Solution.items() if value not in Check.values()]
            print(different_items2)
            raise ValueError("Incorrect Solution.")
        
        total_time = (end_time - start_time) - PJC_time
        return num_queries, total_time