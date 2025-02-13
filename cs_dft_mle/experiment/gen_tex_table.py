import sys
import os
import pandas as pd
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import Util

def process_dataframe(df):
    processed_rows = []
    for _, row in df.iterrows():
        num_targets = row['num_targets']
        k = row['intersection_size']
        recovery_type = row['recovery_type']
        num_invocations = row['num_invocations']
        avg_l1_loss = row['avg_l1_loss']
        avg_recovery_time = row['avg_recovery_time'] / 1e6  # Convert to milliseconds

        if recovery_type == 'mle':
            recovery_type = 'MLE'
            k = num_targets
            num_invocations = int(num_invocations)
        elif recovery_type == 'compressed_sensing_l1':
            recovery_type = 'CS-$\ell_1$'
            k = int(k)
            c = 2
            num_invocations = Util.compute_num_invocations(c, num_targets, k)
        elif recovery_type == 'compressed_sensing_l2':
            recovery_type = 'CS-$\ell_2$'
            k = int(k)
            c = 2
            num_invocations = Util.compute_num_invocations(c, num_targets, k)
        elif recovery_type == 'discrete_fourier':
            k = int(k)
            recovery_type = 'DFT'
            num_invocations = 2 * k

        processed_rows.append({
            'num_targets': num_targets,
            'k': k,
            'recovery_type': recovery_type,
            'num_invocations': num_invocations,
            'avg_l1_loss': avg_l1_loss,
            'avg_recovery_time': avg_recovery_time
        })
    
    return pd.DataFrame(processed_rows)

def generate_latex_table_from_dataframe(df, output_tex, label_min_val_Y, label_max_val_Y, use_noise):
    latex_code = r"""
\begin{table}[t]
    \begin{tabular}{c|r|r|r|r|r}
    \toprule
    $n$ & $k$ & Type & $q$ & $\ell_1$ loss & Time (ms) \\
    \midrule
"""

    grouped = df.groupby('num_targets')

    for num_targets_idx, (num_targets, group) in enumerate(grouped):
        num_targets_exponent = int(math.log10(num_targets))
        n_value = f"$10^{num_targets_exponent}$"
        
        first_row = True

        for k_idx, (k, sub_group) in enumerate(group.groupby('k')):
            sub_group_len = len(sub_group)
            min_l1_loss = sub_group['avg_l1_loss'].min()
            min_l1_loss_rows = sub_group[sub_group['avg_l1_loss'] == min_l1_loss]
            min_q = min_l1_loss_rows['num_invocations'].min()
            min_q_rows = min_l1_loss_rows[min_l1_loss_rows['num_invocations'] == min_q]
            min_recovery_time = min_q_rows['avg_recovery_time'].min()
            first_row_k = True

            for idx, (_, row) in enumerate(sub_group.iterrows()):
                recovery_type = row['recovery_type']
                num_invocations = row['num_invocations']
                avg_l1_loss = row['avg_l1_loss']
                avg_recovery_time = row['avg_recovery_time']

                # Highlight based on the smallest l1 loss, q, and recovery time
                if (avg_l1_loss == min_l1_loss and num_invocations == min_q and avg_recovery_time == min_recovery_time):
                    avg_l1_loss_str = f"\\cellcolor[HTML]{{FFDDC1}}\\textbf{{{avg_l1_loss:.2f}}}"
                else:
                    avg_l1_loss_str = f"{avg_l1_loss:.2f}"

                if first_row and first_row_k:
                    latex_code += f"\\multirow{{{sub_group_len}}}{{*}}{{{n_value}}} & \\multirow{{{sub_group_len}}}{{*}}{{{k}}} & {recovery_type} & {num_invocations} & {avg_l1_loss_str} & {avg_recovery_time:.2f} \\\\ \n"
                    first_row = False
                    first_row_k = False
                elif first_row_k:
                    latex_code += f" & \\multirow{{{sub_group_len}}}{{*}}{{{k}}} & {recovery_type} & {num_invocations} & {avg_l1_loss_str} & {avg_recovery_time:.2f} \\\\ \n"
                    first_row_k = False
                else:
                    latex_code += f" & & {recovery_type} & {num_invocations} & {avg_l1_loss_str} & {avg_recovery_time:.2f} \\\\ \n"

            if k_idx <= sub_group_len - 1:
                latex_code += r"    \hhline{~-----}" + "\n"


        # Add a \midrule after each n group except the last one
        if num_targets_idx < len(grouped) - 1:
            latex_code += r"    \midrule" + "\n"



    # Remove the last \hhline line
    latex_code = latex_code.rstrip(r"\hhline{~-----}\n") + "\n"
    
    latex_code += r"""
    \bottomrule
    \end{tabular}
    """
    
    if use_noise:
        text_label_noise = 'noisy'
        if label_max_val_Y == 100:
            noise_magnitude = 10
            noise_std = 2.5
        elif label_max_val_Y == 1000:
            noise_magnitude = 100
            noise_std = 5
        else:
            raise ValueError("Invalid max_val_Y value.")
        table_caption = f"Value range $[{label_min_val_Y}, {label_max_val_Y}]$, with noise from $\\mathcal{{N}}(\\mu=0, \\sigma={noise_std})$, clipped at magnitude ${noise_magnitude}$."
    else:
        text_label_noise = 'noiseless'
        table_caption = f"Value range $[{label_min_val_Y}, {label_max_val_Y}]$."

    latex_code += f"\n    \\caption{{{table_caption}}}\n"
    latex_code += f"\n    \\label{{tab:{text_label_noise}_{label_max_val_Y}}}\n"
    latex_code += r"""
\end{table}
"""

    with open(output_tex, 'w') as f:
        f.write(latex_code)

def merge_to_csv(noise, max_val_Y):
    # Set the directory containing the CSV files
    directory = "./results"

    # List all CSV files in the directory
    if noise == 'noisy':
        csv_files = [file for file in os.listdir(directory) if file.endswith(f'_noise.csv')]
    else:
        csv_files = [file for file in os.listdir(directory) if not file.endswith(f'_noise.csv')]

    # Merge all CSV files
    merged_df = pd.concat([pd.read_csv(os.path.join(directory, file)) for file in csv_files], ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_csv_str = f"results_table_{noise}_{max_val_Y}.csv"
    merged_df.to_csv(merged_csv_str, index=False)
    print(f"All CSV files have been merged into {merged_csv_str}.")
        
def generate_latex_table_from_csv(input_csv, output_tex):
    parts = input_csv.split('_')
    use_noise = False
    if parts[-2] == 'noisy':
        use_noise = True
    label_max_val_Y = int(parts[-1].split('.')[0])
    label_min_val_Y = -label_max_val_Y

    df = pd.read_csv(input_csv)
    processed_df = process_dataframe(df)

    generate_latex_table_from_dataframe(processed_df, output_tex, label_min_val_Y, label_max_val_Y, use_noise)
    
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['100', '1000']:
        print("Usage: gen_tex_table.py <100 or 1000>")
        sys.exit(1)

    max_val_Y = int(sys.argv[1])
    noise_list = ['noisy', 'noiseless']

    for noise in noise_list:
        merge_to_csv(noise, max_val_Y)
        generate_latex_table_from_csv(
            f'./results_table_{noise}_{max_val_Y}.csv', 
            f'./stats_table_{noise}_{max_val_Y}.tex'
        )
        print(f"Table is generated in ./stats_table_{noise}_{max_val_Y}.tex.")
