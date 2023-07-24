import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(font_scale=2.1)
# sns.set(rc={'ytick.major.size': 5})
sns.color_palette("light:b", as_cmap=True)

basedir = "Plebiscito_results/Topology_aware"
filenames = ["alpha_BW_CPU_bw_usage", "alpha_GPU_BW_bw_usage"]

for filename in filenames:
    
    df = pd.read_csv(os.path.join(basedir, filename) + ".csv")
    
    for alpha in df["alpha"].unique():
        df_f = df.loc[df['alpha'] == alpha]
        df_f2 = df_f.drop('alpha', axis=1)
        df_f2 = df_f2[df_f2.columns.drop(list(df_f2.filter(regex='.*_usage')))]

        df_n = df_f2.sum()

        # print(df_f.sum())
        # print(df_n.T)

        # Compute the ratio for each edge
        ratios = {}
        for i in range(len(df_n) // 2):
            initial_key = f'Edge_{i}_initial'
            final_key = f'Edge_{i}_final'
            ratio_key = f'{i}'
            ratios[ratio_key] = (df_f.sum()[initial_key] - df_f.sum()[final_key])/ df_f.sum()[initial_key] * 100

        # Convert the ratios to a new Pandas Series
        ratios_series = pd.Series(ratios) 
        # print(ratios_series)

        Index= ['Usage (%)']
        Cols = ratios_series.keys()
        data = pd.DataFrame(ratios, index=Index, columns=Cols)

        fig, ax = plt.subplots (figsize=(16, 5))
        sns.heatmap(data, vmin=0, vmax=100, cbar_kws = dict(use_gridspec=False,location="top"), cmap=sns.color_palette("Blues", as_cmap=True), square=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
        ax.set(xlabel="Infrastructure edges")
        
        fig.savefig(str(alpha) + "_" + filename + ".png")
        plt.close()
