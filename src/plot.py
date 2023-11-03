import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plot_folder(dirname):
    # check if the plot directory exists, if not create it
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def plot_node_resource_usage(filename, res_type, n_nodes, dir_name):    
    # plot node resource usage using data from filename
    df = pd.read_csv(filename + ".csv")
    
    # select only the columns matching the pattern node_*_updated_gpu
    df = df.filter(regex=("node.*"+res_type))
    
    d = {}
    for i in range(n_nodes):
        d["node_" + str(i)] = df["node_" + str(i) + "_used_" + res_type] / df["node_" + str(i) + "_initial_" + res_type]
    
    df_2 = pd.DataFrame(d)
    
    # use matplotlib to plot the data and save the plot to a file
    df_2.plot()
    
    plt.ylabel(f"{res_type} usage")
    plt.xlabel("time")
    plt.savefig(os.path.join(dir_name, 'node_' + res_type + '_resource_usage.png'))
    
    # clear plot
    plt.clf()
    
def plot_job_execution_delay(filename, dir_name):
    df = pd.read_csv(filename + ".csv")
        
    res = df["deadline"] - df["exec_time"] + df["duration"]
        
    # plot histogram using the res variable
    res.astype(int).hist()
    
    # save the plot to a file
    plt.ylabel(f"Occurrences")
    plt.xlabel("Job delay time (s)")
    plt.savefig(os.path.join(dir_name, 'job_deadline.png'))
    
    # clear plot
    plt.clf()
    
def plot_job_deadline(filename, dir_name):
    df = pd.read_csv(filename + ".csv")
        
    res = df["exec_time"] - df["arrival_time"]
        
    # plot histogram using the res variable
    res.astype(int).hist()
    
    plt.ylabel(f"Occurrences")
    plt.xlabel("Job deadline exceeded (s)")
    
    # save the plot to a file
    plt.savefig(os.path.join(dir_name, 'job_execution_delay.png'))
    
    # clear plot
    plt.clf()
    
def plot_all(n_edges, filename):
    plot_node_resource_usage(filename, "gpu", n_edges)
    plot_node_resource_usage(filename, "cpu", n_edges)
    
    plot_job_execution_delay("jobs_report")
    plot_job_deadline("jobs_report")
    
if __name__ == "__main__":
    
    dir_name = "plot"
    generate_plot_folder(dir_name)
        
    plot_node_resource_usage("GPU", "gpu", 5, dir_name)
    plot_node_resource_usage("GPU", "cpu", 5, dir_name)
    
    plot_job_execution_delay("jobs_report", dir_name)
    plot_job_deadline("jobs_report", dir_name)