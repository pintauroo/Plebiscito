"""
Module to plot all the info
"""

import csv
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import cumfreq
from matplotlib.lines import Line2D
import itertools
import matplotlib.ticker as ticker

def calculate_ci_alpha(df):

    req = df['alpha'].unique()
    print(req)
    confidence_level = 0.95

    df_lower = pd.DataFrame()
    df_upper = pd.DataFrame()

    for i in req:
        # print(req)
        df_selected = df[df['alpha'] == i]
        # Calculate the sample mean and standard deviation
        x = df_selected.mean(axis=0).to_frame().T
        s = df_selected.std(axis=0).to_frame().T

        # Calculate the standard error of the mean
        n = len(df_selected)
        # print(n)
        SE = s / np.sqrt(n)
        # print(SE)

        # Choose the desired level of confidence and corresponding critical value
        z_star = norm.ppf(1 - (1-confidence_level)/2)

        # Calculate the confidence interval
        lower = x - z_star * SE
        upper = x + z_star * SE

        df_lower = pd.concat([df_lower, lower])
        df_upper = pd.concat([df_upper, upper])
    
    return df_lower, df_upper

def calculate_ci(df):

    req = df['n_req'].unique()
    confidence_level = 0.95

    df_lower = pd.DataFrame()
    df_upper = pd.DataFrame()

    for i in req:
        # print(req)
        df_selected = df[df['n_req'] == i]
        # Calculate the sample mean and standard deviation
        x = df_selected.mean(axis=0).to_frame().T
        s = df_selected.std(axis=0).to_frame().T

        # Calculate the standard error of the mean
        n = len(df_selected)
        # print(n)
        SE = s / np.sqrt(n)
        # print(SE)

        # Choose the desired level of confidence and corresponding critical value
        z_star = norm.ppf(1 - (1-confidence_level)/2)

        # Calculate the confidence interval
        lower = x - z_star * SE
        upper = x + z_star * SE

        df_lower = pd.concat([df_lower, lower])
        df_upper = pd.concat([df_upper, upper])
    
    return df_lower, df_upper

def calculate_median(df):
    req = df['n_req'].unique()
    print(req)
    
    df_res = pd.DataFrame()
    # for i in range(1, N_req+1):
    for i in req:
        df_selected = df[df['n_req'] == i]
        df_res = pd.concat([df_res, df_selected.median(axis=0).to_frame().T])
    return df_res

def calculate_averages(filename):
    averages = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            n_req = int(row['n_req'])
            counts[n_req]['count'] += 1
            for key, value in row.items():
                if key == 'n_req':
                    continue
                averages[n_req][key] += float(value)
    
    for n_req, values in averages.items():
        for key in values:
            if key == 'count':
                continue
            values[key] /= counts[n_req]['count']
    
    return dict(averages)

def plot_data(grouped):

    x_values = grouped['n_req'].unique()    

    # Plot n_msg
    plt.plot(x_values, grouped['n_msg'])
    plt.title('n_msg vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('n_msg')
    plt.savefig('n_req.png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # Plot exec_time
    plt.plot(x_values, grouped['exec_time'])
    plt.title('exec_time vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('exec_time')
    plt.savefig('exec_time.png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


    # Plot jaini
    plt.plot(x_values, grouped['jaini'])
    plt.title('jaini vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('jaini')
    plt.savefig('jaini.png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # Plot utility values over all nodes
    utility_cols = [col for col in grouped.columns if 'utility' in col]
    for col in utility_cols:
        plt.plot(x_values, grouped[col])
    plt.legend(utility_cols)
    plt.title('Utility vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('Utility')
    plt.savefig('Utility.png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # Plot jobs values over all nodes
    jobs_cols = [col for col in grouped.columns if 'jobs' in col]
    for col in jobs_cols:
        plt.plot(x_values, grouped[col])
    plt.legend(jobs_cols)
    plt.title('Jobs vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('Jobs')
    plt.savefig('Jobs.png')

def plot_data_ci(df_lower, df_upper):
    x_values = df_lower['n_req'].unique()

    # Plot n_msg
    plt.plot(x_values, df_lower['n_msg'], label='n_msg')
    plt.fill_between(x_values, df_lower['n_msg'], df_upper['n_msg'], alpha=0.2)
    plt.title('n_msg vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('n_msg')
    plt.legend()
    plt.savefig('n_req.png')
    plt.clf()

    # Plot exec_time
    plt.plot(x_values, df_lower['exec_time'], label='exec_time')
    plt.fill_between(x_values, df_lower['exec_time'], df_upper['exec_time'], alpha=0.2)
    plt.title('exec_time vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('exec_time')
    plt.legend()
    plt.savefig('exec_time.png')
    plt.clf()

    # Plot jaini
    plt.plot(x_values, df_lower['jaini'], label='jaini')
    plt.fill_between(x_values, df_lower['jaini'], df_upper['jaini'], alpha=0.2)
    plt.title('jaini vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('jaini')
    plt.legend()
    plt.savefig('jaini.png')
    plt.clf()

    # Plot utility values over all nodes
    utility_cols = [col for col in df_lower.columns if 'utility' in col]
    for col in utility_cols:
        plt.plot(x_values, df_lower[col], label=col)
        plt.fill_between(x_values, df_lower[col], df_upper[col], alpha=0.2)
    plt.legend()
    plt.title('Utility vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('Utility')
    plt.savefig('Utility.png')
    plt.clf()

    # Plot jobs values over all nodes
    jobs_cols = [col for col in df_lower.columns if 'jobs' in col]
    for col in jobs_cols:
        plt.plot(x_values, df_lower[col], label=col)
        plt.fill_between(x_values, df_lower[col], df_upper[col], alpha=0.2)
    plt.legend()
    plt.title('Jobs vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('Jobs')
    plt.savefig('Jobs.png')
    plt.clf()

def plot_data_ci_compact(df_lower, df_upper):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    x_values = df_lower['n_req'].unique()    
    
    axs[0, 0].plot(x_values, df_lower['n_msg'], label='n_msg')
    axs[0, 0].fill_between(x_values, df_lower['n_msg'], df_upper['n_msg'], alpha=0.2)
    axs[0, 0].set_xlabel('n_req')
    axs[0, 0].set_ylabel('n_msg')
    axs[0, 0].legend()

    axs[0, 1].plot(x_values, df_lower['exec_time'], label='exec_time')
    axs[0, 1].fill_between(x_values, df_lower['exec_time'], df_upper['exec_time'], alpha=0.2)
    axs[0, 1].set_xlabel('n_req')
    axs[0, 1].set_ylabel('exec_time')
    axs[0, 1].legend()

    axs[0, 2].plot(x_values, df_lower['jaini'], label='jaini')
    axs[0, 2].fill_between(x_values, df_lower['jaini'], df_upper['jaini'], alpha=0.2)
    axs[0, 2].set_xlabel('n_req')
    axs[0, 2].set_ylabel('jaini')
    axs[0, 2].legend()

    utility_cols = [col for col in df_lower.columns if 'utility' in col]
    for i, col in enumerate(utility_cols):
        axs[1, 0].plot(x_values, df_lower[col], label=col)
        axs[1, 0].fill_between(x_values, df_lower[col], df_upper[col], alpha=0.2)
    axs[1, 0].set_xlabel('n_req')
    axs[1, 0].set_ylabel('Utility')
    axs[1, 0].legend()

    jobs_cols = [col for col in df_lower.columns if 'jobs' in col]
    for i, col in enumerate(jobs_cols):
        axs[1, 1].plot(x_values, df_lower[col], label=col)
        axs[1, 1].fill_between(x_values, df_lower[col], df_upper[col], alpha=0.2)
    axs[1, 1].set_xlabel('n_req')
    axs[1, 1].set_ylabel('Jobs')
    axs[1, 1].legend()

    axs[1, 2].axis('off')
    fig.tight_layout()
    fig.savefig('metrics.png')

def plot_data_ci_compact_full(df_lower, df_upper):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    x_values = df_lower['n_req'].unique()    
    alpha_values = df_lower['alpha'].unique()   
    
    axs[0, 0].plot(x_values, df_lower['n_msg'], label='n_msg')
    axs[0, 0].fill_between(x_values, df_lower['n_msg'], df_upper['n_msg'], alpha=0.2)
    axs[0, 0].set_xlabel('n_req')
    axs[0, 0].set_ylabel('n_msg')
    axs[0, 0].set_title('Number of Messages over requests number')
    axs[0, 0].legend()

    axs[0, 1].plot(x_values, df_lower['exec_time'], label='exec_time')
    axs[0, 1].fill_between(x_values, df_lower['exec_time'], df_upper['exec_time'], alpha=0.2)
    axs[0, 1].set_xlabel('n_req')
    axs[0, 1].set_ylabel('exec_time')
    axs[0, 1].set_title('Execution time over requests number')
    axs[0, 1].legend()

    axs[0, 2].plot(x_values, df_lower['jaini'], label='jaini')
    axs[0, 2].fill_between(x_values, df_lower['jaini'], df_upper['jaini'], alpha=0.2)
    axs[0, 2].set_xlabel('n_req')
    axs[0, 2].set_ylabel('jaini')
    axs[0, 2].set_title('Fairness over requests number')
    axs[0, 2].legend()

    utility_cols = [col for col in df_lower.columns if 'utility' in col]
    for i, col in enumerate(utility_cols):
        axs[1, 0].plot(x_values, df_lower[col], label=col)
        axs[1, 0].fill_between(x_values, df_lower[col], df_upper[col], alpha=0.2)
    axs[1, 0].set_xlabel('n_req')
    axs[1, 0].set_ylabel('Utility')
    axs[1, 0].set_title('Utility function over requests number')
    axs[1, 0].legend()

    jobs_cols = [col for col in df_lower.columns if 'jobs' in col]
    for i, col in enumerate(jobs_cols):
        axs[1, 1].plot(x_values, df_lower[col], label=col)
        axs[1, 1].fill_between(x_values, df_lower[col], df_upper[col], alpha=0.2)
    axs[1, 1].set_xlabel('n_req')
    axs[1, 1].set_ylabel('Jobs')
    axs[1, 1].set_title('Node assigned jobs over requests number')
    axs[1, 1].legend()

    axs[1, 2].plot(x_values, df_lower['count_assigned']/x_values, label='count_assigned/n_req')
    axs[1, 2].fill_between(x_values, df_lower['count_assigned']/x_values, df_upper['count_assigned']/x_values, alpha=0.2)
    axs[1, 2].plot(x_values, df_lower['count_unassigned']/x_values, label='count_unassigned/n_req')
    axs[1, 2].fill_between(x_values, df_lower['count_unassigned']/x_values, df_upper['count_unassigned']/x_values, alpha=0.2)
    axs[1, 2].set_xlabel('n_req')
    axs[1, 2].set_ylabel('Count')
    axs[1, 2].set_title('Allocation ratio over requests number')
    axs[1, 2].legend()

    # axs[2, 0].plot(alpha_values, df_lower['tot_utility'], label='n_msg')
    # axs[2, 0].fill_between(x_values, df_lower['tot_utility'], df_upper['tot_utility'], alpha=0.2)
    # axs[2, 0].set_xlabel('alpha')
    # axs[2, 0].set_ylabel('tot_utility')
    # axs[2, 0].set_title('Number of Messages over requests number')
    # axs[2, 0].legend()

    fig.tight_layout()
    fig.savefig('metrics.png')

def plot_ci(grouped):

    x_values = grouped['alpha'].unique()    

    # Plot n_msg
    plt.plot(x_values, grouped['tot_utility'])
    plt.title('alpha vs tot_utility')
    plt.xlabel('alpha')
    plt.ylabel('tot_utility')
    plt.savefig('alpha.png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # Plot exec_time
    plt.plot(x_values, grouped['count_assigned'])
    plt.plot(x_values, grouped['count_unassigned'])
    plt.title('exec_time vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('exec_time')
    plt.savefig('assigned.png')

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # Plot exec_time
    plt.plot(x_values, grouped['count_assigned'])
    plt.plot(x_values, grouped['count_unassigned'])
    plt.title('exec_time vs n_req')
    plt.xlabel('n_req')
    plt.ylabel('exec_time')
    plt.savefig('assigned.png')


def plot_alpha(df_lower_alpha_BW__GPU, df_upper_alpha_BW__GPU, df_lower_alpha_BW__CPU, df_upper_alpha_BW__CPU, df_lower_alpha_GPU__CPU, df_upper_alpha_GPU__CPU):
    x_values = df_lower_alpha_BW__GPU['alpha'].unique()
    n_measurements = 3  # Number of measurements
    bar_width = 0.1 / n_measurements  # Width of each bar, adjusted for spacing
    alpha_values = ['BW vs GPU', 'alpha_BW vs CPU', 'alpha_GPU vs CPU']

    fig, ax = plt.subplots()

    # Define colors for the bars
    bar_colors = ['red', 'green', 'blue']

    # Plot confidence intervals as bars
    ax.errorbar(x_values - bar_width, df_lower_alpha_BW__GPU['tot_utility'], yerr=[df_lower_alpha_BW__GPU['tot_utility'] - df_lower_alpha_BW__GPU['tot_utility'], df_upper_alpha_BW__GPU['tot_utility'] - df_lower_alpha_BW__GPU['tot_utility']], fmt='none', label='BW vs GPU', capsize=4, alpha=0.7, color=bar_colors[0])
    ax.errorbar(x_values, df_lower_alpha_BW__CPU['tot_utility'], yerr=[df_lower_alpha_BW__CPU['tot_utility'] - df_lower_alpha_BW__CPU['tot_utility'], df_upper_alpha_BW__CPU['tot_utility'] - df_lower_alpha_BW__CPU['tot_utility']], fmt='none', label='BW vs CPU', capsize=4, alpha=0.7, color=bar_colors[1])
    ax.errorbar(x_values + bar_width, df_lower_alpha_GPU__CPU['tot_utility'], yerr=[df_lower_alpha_GPU__CPU['tot_utility'] - df_lower_alpha_GPU__CPU['tot_utility'], df_upper_alpha_GPU__CPU['tot_utility'] - df_lower_alpha_GPU__CPU['tot_utility']], fmt='none', label='GPU vs CPU', capsize=4, alpha=0.7, color=bar_colors[2])

    ax.set_xlabel('alpha parameter')
    ax.set_ylabel('global utility')


    # Set x-axis ticks and labels
    x_tick_labels = [f'{round(x, 1):.1f}' for x in df_lower_alpha_BW__GPU['alpha'].unique()]  # Round values to 1 decimal place
    ax.set_xticks(x_values - bar_width)
    ax.set_xticklabels(x_tick_labels)

    # Add legend
    ax.legend()

    plt.savefig('alpha.pdf')


#Plot CDF for total utility over alpha
def plot_cdf(df, filename):

    req = df['alpha'].unique()
    print(req)

    cdf_df = pd.DataFrame()

    # Define the global fontsize variable
    global_fontsize = 20

    # Set the fontsize for x-axis and y-axis labels
    plt.rc('xtick', labelsize=global_fontsize)
    plt.rc('ytick', labelsize=global_fontsize)

    # Create a figure with the desired size
    fig = plt.figure(figsize=(10, 4))
    plt.subplots_adjust(left=0.15, bottom=0.2)  # Adjust the left margin of the figure

    # Define the line styles and colors for each curve
    line_styles = ['dotted', 'dashed', 'dashdot', 'dotted', 'dashdot']
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i in range(len(req)):
        df_selected = df[df['alpha'] == req[i]]
        column_values = df_selected['tot_utility']
        sorted_values = np.sort(column_values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

        # Create a new DataFrame with the CDF values and corresponding column values
        cdf_df_ = pd.DataFrame({'CDF': cdf, 'Column Values': sorted_values})
        print(cdf_df_)

        lbl = None
        if req[i] == 0:
            lbl = '\u03B1=' + str(req[i]) + ' (CPU)'
        elif req[i] == 1:
            lbl = '\u03B1=' + str(req[i]) + ' (BW)'
        else:
            lbl = '\u03B1=' + str(req[i])

        # Plotting code with different line styles and colors
        plt.plot(cdf_df_['Column Values'], cdf_df_['CDF'], linestyle=line_styles[i % len(line_styles)],
                 color=colors[i % len(colors)], label=lbl)

        plt.xlabel('Plebiscito Utility', fontsize=global_fontsize)
        plt.ylabel('CDF', fontsize=global_fontsize)
        plt.grid(True)

        # Move the legend outside the plot box
        plt.legend(fontsize=global_fontsize-2, bbox_to_anchor=(0.45, 1.22), columnspacing=0.12,
                   loc='upper center', ncol=len(req))

        plt.xticks(fontsize=global_fontsize)
        plt.yticks(fontsize=global_fontsize)

        cdf_df = pd.concat([cdf_df, cdf_df_])

    # Save the figure with the desired size
    fig.savefig(str(filename) + '_' + str(req[i]) + '.pdf', dpi=900)
    print(cdf_df)


# Plot assigned vs unassigned jobs
def plot_assigned_jobs(df):
    df_lower, df_upper = calculate_ci_alpha(df)

    # Assuming you have the following data
    x_values = df['alpha'].unique()
    n_req = df_lower['n_req'].unique()  

    print(n_req)
    # print(x_values)
  
   # Calculate the heights and errors for the bars
    height_assigned = np.array(df_lower['count_assigned']) / n_req
    height_unassigned = np.array(df_lower['count_unassigned']) / n_req
    errors_assigned = (np.array(df_upper['count_assigned']) - np.array(df_lower['count_assigned'])) / n_req
    errors_unassigned = (np.array(df_upper['count_unassigned']) - np.array(df_lower['count_unassigned'])) / n_req

    # Set the width and spacing between the bars
    bar_width = 0.35
    bar_spacing = 0.1

    # Calculate the x-axis positions for the bars
    x_positions_assigned = np.arange(len(x_values))
    x_positions_unassigned = x_positions_assigned + bar_width + bar_spacing

    # Create the bar plot
    plt.bar(x_positions_assigned, height_assigned, yerr=errors_assigned, width=bar_width, label='count_assigned/n_req', alpha=0.5, capsize=5)
    plt.bar(x_positions_unassigned, height_unassigned, yerr=errors_unassigned, width=bar_width, label='count_unassigned/n_req', alpha=0.5, capsize=5)

    # Set the x-axis tick positions and labels
    plt.xticks(x_positions_assigned + bar_width/2, x_values)

    # Set the x-axis label, y-axis label, and title
    plt.xlabel('n_req')
    plt.ylabel('Count')
    plt.title('Allocation ratio over requests number')

    # Display the legend
    plt.legend()
    plt.savefig('tst.pdf', dpi=900)



def calc_tot_used_res(df, one, two):

    output = []
    for index, row in df.iterrows():
        # Initialize sum variable for each entry
        count_res = 0
        print(index)
        # Sum over columns with 'node_x_leftover_gpu' label
        
        for x in range(10):
            column_label = str(one)+f'{x}'+str(two)
            if column_label in df.columns:
                print(str(column_label)+str(row[column_label])+str(f'node_{x}_initial_gpu')+str(row[f'node_{x}_initial_gpu']))
                count_res += row[column_label]
        
        # Append sum to output array
        output.append(count_res)
        # output_used_gpu_lower.append(entry_sum)
    return output



def plot_gpu_cpu_res(df):

    df_lower, df_upper = calculate_ci_alpha(df)
    # print(df_lower)

    # Assuming you have the following data
    x_values = df['alpha'].unique()
    n_req = df_lower['n_req'].unique()  



    # Assuming your DataFrame is named 'df'
    output_used_gpu_upper = calc_tot_used_res(df_upper, 'node_', '_leftover_gpu')
    output_used_gpu_lower = calc_tot_used_res(df_lower, 'node_', '_leftover_gpu')
    output_used_cpu_lower = calc_tot_used_res(df_lower, 'node_', '_leftover_cpu')
    output_used_cpu_upper = calc_tot_used_res(df_upper, 'node_', '_leftover_cpu')
    tot_cpu_lower = calc_tot_used_res(df_lower, 'node_', '_initial_cpu')
    tot_cpu_upper = calc_tot_used_res(df_upper, 'node_', '_initial_cpu')
    print(tot_cpu_lower)
    print(tot_cpu_upper)
    print(output_used_cpu_upper)
    print(output_used_cpu_lower)


    # Calculate the heights and errors for the bar
    height_assigned_gpu = np.array(df_lower['tot_gpu']) 
    height_unassigned_gpu = np.array(output_used_gpu_lower) 
    errors_assigned_gpu = (np.array(df_upper['tot_gpu']) - np.array(df_lower['tot_gpu'])) 
    errors_unassigned_gpu = (np.array(output_used_gpu_upper) - np.array(output_used_gpu_lower))

    height_assigned_cpu = np.array(tot_cpu_lower) 
    height_unassigned_cpu = np.array(output_used_cpu_lower) 
    errors_assigned_cpu = (np.array(tot_cpu_upper) - np.array(tot_cpu_lower)) 
    errors_unassigned_cpu = (np.array(output_used_cpu_upper) - np.array(output_used_cpu_lower))


    # Set the width and spacing between the bars
    bar_width = 0.35
    bar_spacing = 0.1

    # Calculate the x-axis positions for the bars
    x_positions_assigned = np.arange(len(x_values))
    x_positions_unassigned = x_positions_assigned + bar_width + bar_spacing

    # Create the bar plot
    bar_width = 0.5
    plt.bar(x_positions_assigned,   height_assigned_gpu, yerr=errors_assigned_gpu, width=bar_width, label='count_assigned/n_req', alpha=0.5, capsize=5)
    plt.bar(x_positions_assigned, height_unassigned_gpu, yerr=errors_unassigned_gpu, width=bar_width, label='count_unassigned/n_req', alpha=0.5, capsize=5, linewidth=1.5)    
    
    
    plt.bar(x_positions_unassigned,   height_assigned_cpu, yerr=errors_assigned_cpu, width=bar_width, label='count_assigned/n_req', alpha=0.5, capsize=5)
    plt.bar(x_positions_unassigned, height_unassigned_cpu, yerr=errors_unassigned_cpu, width=bar_width, label='count_unassigned/n_req', alpha=0.5, capsize=5, linewidth=1.5)    
    # plt.bar(x_positions_unassigned, height_unassigned, yerr=errors_unassigned, width=bar_width, label='count_unassigned/n_req', alpha=0.5, capsize=5)

    # Set the x-axis tick positions and labels
    plt.xticks(x_positions_assigned + bar_width/2, x_values)

    # Set the x-axis label, y-axis label, and title
    plt.xlabel('n_req')
    plt.ylabel('Count')

    # Display the legend
    plt.legend()
    plt.savefig('tst.pdf', dpi=900)


# Main function to run the script
def main():
    
    # Define line styles and colors
    line_styles = ['dotted', 'dashed', 'dashdot', 'dotted', 'dashdot']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    color_cycle = itertools.cycle(colors)

    # Read the data from the CSV file
    filenames = ['alpha_BW_CPU', 'alpha_GPU_BW', 'alpha_GPU_CPU']
    fig, axes = plt.subplots(len(filenames), len(['count_unassigned', 'count_assigned', 'tot_used_gpu', 'tot_used_cpu', 'tot_bw']), figsize=(20, 10))

    for file_index, filename_ in enumerate(filenames):
        print('ktm')
        filename = '/home/andrea/Documents/Projects/decomposition_framework/'+str(filename_)+'.csv'
        resources = filename_.split("_")
        df = pd.read_csv(filename)
        req = df['alpha'].unique()
        cdf_df = pd.DataFrame()

        for label_index, label in enumerate(['count_unassigned', 'count_assigned', 'tot_used_gpu', 'tot_used_cpu', 'tot_used_bw']):
            ax = axes[file_index, label_index]

            for i in range(len(req)):
                df_selected = df[df['alpha'] == req[i]]
                if 'tot' in label:
                    column_values = df_selected[label] / df_selected[label.replace("_used", "")]
                else:
                    column_values = df_selected[label]

                sorted_values = np.sort(column_values)
                cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

                # Create a new DataFrame with the CDF values and corresponding column values
                cdf_df_ = pd.DataFrame({'CDF': cdf, 'Column Values': sorted_values})

                lbl = None
                if req[i] == 0:
                    lbl = '\u03B1=' + str(req[i]) + ' ' + str(resources[2])
                elif req[i] == 1:
                    lbl = '\u03B1=' + str(req[i]) + ' ' + str(resources[1])
                else:
                    lbl = '\u03B1=' + str(req[i])

                # Get the next color from the color cycle
                color = next(color_cycle)

                # Plotting code with different line styles and colors
                ax.plot(cdf_df_['Column Values'], cdf_df_['CDF'], linestyle=line_styles[i % len(line_styles)],
                        color=color, label=lbl)

                ax.set_xlabel(label)
                ax.set_ylabel('CDF')
                ax.grid(True)
                ax.legend(fontsize=10, loc='upper right')

        plt.tight_layout()

    # Save the figure with the desired size
    fig.savefig('plots_combined.pdf', dpi=900)
    # print(df['alpha'])
    # plot_cdf(pd.read_csv(filename))

   

    # alpha_BW__GPU = '/home/andrea/Documents/Projects/decomposition_framework/alpa_BW_GPU.csv'
    # alpha_BW__CPU = '/home/andrea/Documents/Projects/decomposition_framework/alpha_BW__CPU.csv'
    # alpha_GPU__CPU = '/home/andrea/Documents/Projects/decomposition_framework/alpa_GPU_CPU.csv'
    # # df = calculate_median(pd.read_csv(filename))
    # df_lower_alpha_BW__GPU, df_upper_alpha_BW__GPU = calculate_ci_alpha(pd.read_csv(alpha_BW__GPU))
    # df_lower_alpha_BW__CPU, df_upper_alpha_BW__CPU = calculate_ci_alpha(pd.read_csv(alpha_BW__CPU))
    # df_lower_alpha_GPU__CPU, df_upper_alpha_GPU__CPU = calculate_ci_alpha(pd.read_csv(alpha_GPU__CPU))

    # plot_alpha(df_lower_alpha_BW__GPU, df_upper_alpha_BW__GPU, df_lower_alpha_BW__CPU, df_upper_alpha_BW__CPU, df_lower_alpha_GPU__CPU, df_upper_alpha_GPU__CPU)    # plot_data_ci_compact_full(df_lower, df_upper)
  
    # return df
    # Calculate averages
    # averages = calculate_averages(filename)


    # df = pd.DataFrame.from_dict(averages, orient='index')
    # df.to_csv('averages.csv', index=False)

    # plot_data(df)


if __name__ == '__main__':
    main()
