# Setup Gurobi license
Make sure you have a Gurobi license:
https://www.gurobi.com/academia/academic-program-and-licenses/
The default configuration in `experiment.py` uses ortools instead so that it can
be run even if the gurobi is not installed (for the default recovery
methods). However the evaluation is performed using Gurobi.

# Using Python Virtual Environment

## 1. Install `virtualenv`

If you don't have `virtualenv` installed, you can install it with the following
command:
`pip install virtualenv`

## 2. Create a Virtual Environment

To create a new virtual environment, run:
TODO: specify the python version
`virtualenv myenv`

## 3. Activate the Virtual Environment

### Windows:

`myenv\Scripts\activate`

### macOS/Linux:

`source myenv/bin/activate`

## 4. Install Packages Listed in `requirements.txt`

`pip install -r requirements.txt`

## 5. Use the Environment

You can now use this environment for your project.

1. Open the `config` folder and run `python3 gen_parameters.py`, it will output two json files
`generated_params_non_sparse.json` (for compressed sensing l1, l2, and discrete
fourier) and `generated_params_sparse.json` (for maximum likelihood estimation).
You can edit the `gen_parameters.py` to generate different parameters for experiments.

2. Open the `experiment` folder and run `python3 experiment.py`, it will output a `results` folder that contains a list of csv files reporting the results.
You can edit the list of recover methods (with and without noise) that you would like to use in the main function of the  'experiment.py' file.

When you're done, you can deactivate the environment with:
`deactivate`
