import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import Util
import math

def generate_latex_table_from_csv(input_csv, output_tex):
    # Read the CSV file into a DataFrame
    parts = input_csv.split('_')
    use_noise = False
    if parts[-2] == 'noisy':
        use_noise = True
    label_max_val_Y = int(parts[-1].split('.')[0])
    label_min_val_Y = -label_max_val_Y
    # label_num_targets = int(parts[-1].split('.')[0])

    df = pd.read_csv(input_csv)
    # Convert avg_recovery_time to seconds
    df['avg_recovery_time'] = df['avg_recovery_time'] / 1e6  # Assuming time is in nanoseconds

    # Start the LaTeX table string
    latex_code = r"""
\begin{table}
\begin{tabular}{rrrrrr}
\toprule
Type & $n$ & $k$ & $q$ & $\ell_1$ Loss & Time (ms) \\
\midrule
"""
    # Iterate through the DataFrame rows to build the LaTeX table rows
    for _, row in df.iterrows():
        min_val_Y = row['min_val_Y']
        max_val_Y = row['max_val_Y']
        num_targets = row['num_targets']
        k = row['intersection_size']
        recovery_type = row['recovery_type']        
        num_invocations = row['num_invocations']
        avg_l1_loss = row['avg_l1_loss']
        avg_recovery_time = row['avg_recovery_time']  # Already converted to seconds

        if recovery_type == 'mle':
            recovery_type = 'MLE'
            k = num_targets
            num_invocations = int(num_invocations)
        elif recovery_type == 'compressed_sensing_l1':
            recovery_type = 'CS-l1'
            k = int(k)
            c = 2
            num_invocations = Util.compute_num_invocations(c, num_targets, k)
        elif recovery_type == 'compressed_sensing_l2':
            recovery_type = 'CS-l2'
            k = int(k)
            c = 2
            num_invocations = Util.compute_num_invocations(c, num_targets, k)
        elif recovery_type == 'discrete_fourier':
            k = int(k)
            recovery_type = 'DFT'
            num_invocations = 2 * k
            
        # Add a row to the LaTeX table
        num_targets_exponent = int(math.log10(num_targets))
        latex_code += f"{recovery_type} & $10^{num_targets_exponent}$ & {k} & {num_invocations} & {avg_l1_loss:.2f} & {avg_recovery_time:.2f} \\\\ \n"

    if use_noise:
        if label_max_val_Y == 100:
            noise_magnitude = 10
            noise_std = 2.5
        elif label_max_val_Y == 1000:
            noise_magnitude = 100
            noise_std = 5
        else:
            raise ValueError("Invalid max_val_Y value.")
        table_caption = f"Value range = $[{label_min_val_Y}, {label_max_val_Y}]$, with noise from $\calN(\mu=0, \sigma={noise_std})$, clipped at magnitude ${noise_magnitude}$."
    else:
        table_caption = f"Value range = $[{label_min_val_Y}, {label_max_val_Y}]$."
    latex_code += r"""
\bottomrule
\end{tabular}
"""
    latex_code += f"\caption{{{table_caption}}}\n"
    latex_code += r"""
\end{table}
"""

    # Write the LaTeX code to the output .tex file
    with open(output_tex, 'w') as f:
        f.write(latex_code)


if __name__ == "__main__":
    max_val_Y_list = [100, 1000]
    noise_list = ['noisy', 'noiseless']
    for max_val_Y in max_val_Y_list:
        for noise in noise_list:
            generate_latex_table_from_csv(f'./results_table_{noise}_{max_val_Y}.csv', f'./stats_table_{noise}_{max_val_Y}.tex')