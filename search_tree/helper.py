class HelperFunctions:

    def __init__(self, arity) -> None:
        self.arity = arity

    def inner_product(self, dict1, dict2):
        """
        Compute inner product of values whose keys are in the intersection.
        -----
        Inputs: Two dictionaries mapping labels to values (int).
        Returns: The inner product of values whose labels are in the intersection.
        """
        common_keys = set(dict1.keys()).intersection(dict2.keys())
        return sum(dict1[key] * dict2[key] for key in common_keys)


    def compute_chunks(self, num_parts, A):
        """
        Yield 'num_parts' striped chunks from array A.
                -----
        Inputs: Number of paritions (int) and array of integers A.
        Returns: Subarrays corresponding to a partition of A. 
        """
        k, m = divmod(len(A), num_parts)
        for i in range(num_parts):
            start_index = i * k + min(i, m)
            end_index = start_index + k + (1 if i < m else 0)
            yield A[start_index:end_index]


    def compute_max_coeff(self, height, num_parts, A):
        """
        Compute the maximum possible coefficient of the linear eq.
                -----
        Inputs: Height of query in the search tree (int), the number of partitions 
            (int) and the array of index values (list[int]).
        Returns: The maximum possible coefficient (int) over all sets in the parition.
        """
        max_coeff = 0
        for subarray in self.compute_chunks(num_parts, A):
            # Shift by len(subarray) since python indexes starting at 0.
            max_coeff = max(max_coeff, height * (sum(subarray)+len(subarray))) 
        return max_coeff
     