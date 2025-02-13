# README

## Setup

### Download and install Gurobi Optimizer

To evaluate the attacks in the `cs_dft_mle` directory, you need to install the Gurobi Optimizer with a (free) academic license.

Make sure you have a Gurobi license:
[https://www.gurobi.com/academia/academic-program-and-licenses/]
The default configuration in `experiment.py` uses ortools instead so that it can
be run even if the gurobi is not installed (for the default recovery
methods). However the evaluation is performed using Gurobi.

### Use Python virtual environment

Tested under `python 3.10.15`.

1. Install `virtualenv`

   If you don't have `virtualenv` installed, you can install it with the
   following command:
   
   `pip install virtualenv`

2. Create a virtual environment

   To create a new virtual environment, run:
   
   `virtualenv myenv`

3. Activate the virtual environment

   *Windows*: `myenv\Scripts\activate`
   
   *macOS/Linux*: `source myenv/bin/activate`

4. Install packages listed in `requirements.txt`

   `pip install -r requirements.txt`

## Work in the `search_tree` directory

To run a simple functionality test for the search tree attack, run the following
command from the top level directory:

`python3 test.py`

Before running the benchmark code, create the directory in which the results will be stored:

`mkdir results`

To run the experiments for the search tree attack, run the following commmand from the `search_tree` directory of the repository:

`python3 benchmark.py NUM-RUNS PARAMS`

where 
`NUM-RUNS` (int) denotes how many times to run the attack for each parameter setting and 
`PARAMS` (str) that takes on the value of `part` (which runs an abbreviatedset of parameters 
and which can run in <1hr on a commercial laptop) or `full` 
(which runs the full set of parameters reported in our paper). 

For example, the command

`python3 benchmark.py 1 part`

specifies running the benchmarks for the abbreciated
set of parameters for 1 run per parameter setting.

## Work in the `cs_dft_mle` directory

1. Open the `config` folder and run `python3 gen_parameters.py`, it will output two json files
`generated_params_non_sparse.json` (for compressed sensing l1, l2, and discrete
fourier) and `generated_params_sparse.json` (for maximum likelihood estimation).
You can edit the `gen_parameters.py` to generate different parameters for experiments.

2. Open the `experiment` folder and run `python3 experiment.py`, it will output a `results` folder that contains a list of csv files reporting the results.
You can edit the list of recover methods (with and without noise) that you would like to use in the main function of the  `experiment.py` file.

When you're done, you can deactivate the environment with: `deactivate`
