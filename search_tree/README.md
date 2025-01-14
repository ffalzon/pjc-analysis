# README for the Basic and Search Tree Attacks

## Detailed Usage

These experiments assume prior installation of Python 3 via the python3 command. First clone the repository. This code
requires installing the tqdm library, the details of which can be found in requirements.txt.

Before running the code, create the directory in which the results will be stored:
```
mkdir results
```

To run the experiments for the search tree attack, run the following commmand from the `search_tree` directory of the repository.

```
python3 benchmark.py ATTACK MAX_VAL SECRET_SET_SIZE NUM-RUNS
```

where
`ATTACK` specifies either the `basic` attack or the `tree` attack,
`MAX_VAL` (int) specifies the maximum possible value of the secret set,
`SECRET_SET_SIZE` (int) specifies the size of the victim's set, Y, and
`NUM-RUNS` (int) specifies the number of times you wish to repeat the attack for any set of parameters.

An example of the command is below:

```
python3 benchmark.py tree 100 1000 10
```

The results are printed out to csv files contained in the `/results` directory. 
For the basic attack, the results are output to a file named `basic-SECRET_SET_SIZE-results.csv` and
for the search-tree attack, the results are output to `tree-LOG_OF_ARITY-SECRET_SET_SIZE-results.csv`.
