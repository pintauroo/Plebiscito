"""
Module to plot all the info
"""

import csv
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


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


# Main function to run the script
def main():
    filename = '/home/andrea/Documents/Projects/bidding/bidding_results.csv'

    # df = calculate_median(pd.read_csv(filename))
    df_lower, df_upper = calculate_ci_alpha(pd.read_csv(filename))

    print(df_upper)

    plot_ci(df_lower)

    # plot_data_ci_compact_full(df_lower, df_upper)
  
    # return df
    # Calculate averages
    # averages = calculate_averages(filename)


    # df = pd.DataFrame.from_dict(averages, orient='index')
    # df.to_csv('averages.csv', index=False)

    # plot_data(df)


if __name__ == '__main__':
    main()
