import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.size': 13})

def clean_data_as_dataframe(filename):
    df = pd.read_csv(filename)
    print(f'Row count of {filename} before clean is: {len(df.index)}')
    
    #print(df)
    
    for column in df:
        df.loc[df[column] == -0.0, column] = 0.0
        df = df[df[column]>=0]

    print(f'Row count of {filename} after clean is: {len(df.index)}')
    
    return df

# filenames = ['stefano', 'alpha_GPU_CPU']

basepath = ''
folder = 'Plebiscito_results/Risorse_abbondanti'
filenames = ['alpha_GPU_CPU', 'stefano']

res = []

# for filename_ in filenames:
#     filename = os.path.join('stefano-data', '', str(filename_)+'.csv') 
#     df = clean_data_as_dataframe(filename)

#     norm = {}
    
#     for index, row in df.iterrows():
#         for i in range(int(row["n_nodes"])):
#             if "node" + str(i) not in norm:
#                 norm["node" + str(i)] = []
            
#             max_cpu = 0
#             min_cpu = float('inf')
#             max_gpu = 0
#             min_gpu = float('inf')
            
#             for index2, row2 in df.iterrows():    
#                 if row2["node_" + str(i) + "_updated_cpu"] < min_cpu:
#                     min_cpu = row2["node_" + str(i) + "_updated_cpu"]
#                 if row2["node_" + str(i) + "_updated_cpu"] > max_cpu:
#                     max_cpu = row2["node_" + str(i) + "_updated_cpu"]
#                 if row2["node_" + str(i) + "_updated_gpu"] < min_gpu:
#                     min_gpu = row2["node_" + str(i) + "_updated_gpu"]
#                 if row2["node_" + str(i) + "_updated_gpu"] > max_gpu:
#                     max_gpu = row2["node_" + str(i) + "_updated_gpu"]
            
#             # cpu_scaled = (row["node_" + str(i) + "_updated_cpu"] - min_cpu)/(max_cpu - min_cpu)
#             # gpu_scaled = (row["node_" + str(i) + "_updated_gpu"] - min_gpu)/(max_gpu - min_gpu)
#             cpu_scaled = (row["node_" + str(i) + "_updated_cpu"])/max_cpu
#             gpu_scaled = (row["node_" + str(i) + "_updated_gpu"])/max_gpu
#             tmp = round(math.sqrt((cpu_scaled)**2 + (gpu_scaled)**2), 3)
#             norm["node" + str(i)].append(tmp)
    
#     pd.DataFrame(norm).to_csv(filename_+"_norm_processed.csv")

fig, ax = plt.subplots()#figsize=(15, 10))
colors = ["blue", "red", "green", "orange", "black"]

count = 0
    
for id, filename_ in enumerate(filenames):
    filename = os.path.join(basepath, folder, str(filename_)+'.csv') 
    df = clean_data_as_dataframe(filename)
    # df = df[df['alpha'] == 1]

    for a in [0, 0.5, 1]:
        # if filename_ == 'stefano' and a == 0 :
        #     print('ktm')
        #     df = df[df['alpha'] == 0.01]
        # else:
        if filename_ == "stefano" and a != 1:
            continue
        
        df_ = df[df['alpha'] == a]
        # print(df_)

        jini = {}
        jini["jini_cpu"] = []
        jini["jini_gpu"] = []
        
        for index, row in df_.iterrows():
            sum_cpu = 0
            sum_cpu_square = 0
            sum_gpu = 0
            sum_gpu_square = 0
                            
            
            for i in range(int(row["n_nodes"])):
                sum_cpu += float(row["node_" + str(i) + "_updated_cpu"])
                sum_cpu_square += float(row["node_" + str(i) + "_updated_cpu"])**2
                sum_gpu += float(row["node_" + str(i) + "_updated_gpu"])
                sum_gpu_square += float(row["node_" + str(i) + "_updated_gpu"])**2
                
            jini["jini_cpu"].append(sum_cpu**2 / (int(row["n_nodes"])* sum_cpu_square))
            jini["jini_gpu"].append(sum_gpu**2 / (int(row["n_nodes"])* sum_gpu_square))
        
        df_f = pd.DataFrame(jini)
        # print(df_f)
        lower_cpu = df_f["jini_cpu"].quantile(0.05)
        higher_cpu = df_f["jini_cpu"].quantile(0.95)
        lower_gpu = df_f["jini_gpu"].quantile(0.05)
        higher_gpu = df_f["jini_gpu"].quantile(0.95)

        label = ""
        if filename_ == "stefano":
            label = "CPU/GPU ratio"
        elif filename_ == "alpha_GPU_CPU":
            label = "α=" + str(a) + "<CPU GPU>"
        
        print(f"({lower_cpu},{lower_gpu}), {higher_cpu - lower_cpu}, {higher_gpu- lower_gpu}")
        ax.add_patch(Rectangle((lower_cpu, lower_gpu), higher_cpu - lower_cpu, higher_gpu - lower_gpu, alpha=0.2, label=label, color=colors[count], fill=True))
        ax.set_ylabel('GPU')
        ax.set_xlabel('CPU')
        ax.set_xbound(lower=0.6, upper=1)
        ax.set_ybound(lower=0.6, upper=1)

        count += 1

ax.legend(loc='upper left', ncol=1)
fig.tight_layout()
fig.savefig("jini.pdf")