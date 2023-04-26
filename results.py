import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


path = "/home/andrea/Documents/decomposition/consensus/bidding/res/"
messages = np.array([])
times = np.array([])
df_time = pd.DataFrame()
df_messages = pd.DataFrame()

# writing to file
# for fold in range (1, 50):
for fold in {20, 27, 47, 50}:
    message = np.array([])
    time = np.array([])

    print(fold)
    for i in range (0, 20):
        file = open(str(path)+"/"+str(fold)+"/"+str(i), 'r')
        Lines = file.readlines()
        time = np.append(time, float(Lines[0])) 
        for line in Lines:
            pass
        message = np.append(message, int(line)) 
    # print(message)
    
    # messages = np.append(messages, message)
    # times = np.append(times, time)

    df_time[fold] = time
    df_messages[fold] = message

# print(df_messages.info())
print(df_time.describe())
# # print(df_messages.info())
# print(df.describe())




# # Calculate the mean of each column
# mean = df_messages.median()

# # Plot the mean of each column
# mean.plot(kind='bar')
# plt.xlabel('Sample')
# plt.ylabel('Median')
# plt.title('Median of #messages')
# plt.show()




# #Calculate the confidence intervals with bootstrap method
# n_samples = 1000

# conf_int = pd.DataFrame(
#     {col: df_messages[col].sample(n_samples, replace=True).quantile([0.025, 0.975]) for col in df_messages.columns}
# ).transpose()

# # Plot the median of each column
# fig, axs = plt.subplots()
# axs.set_xlabel("Sample")
# axs.set_ylabel("Median")
# axs.set_title("Median of Samples with Confidence Interval")
# for col in df_messages.columns:
#     axs.errorbar(col,df_messages[col].median(),yerr=[[df_messages[col].median()-conf_int.loc[col].values[0]],[conf_int.loc[col].values[1]-df_messages[col].median()]], fmt='o')

# plt.show()



# Remove outliers using Interquartile Range (IQR) method
Q1 = df_time.quantile(0.25)
Q3 = df_time.quantile(0.75)
IQR = Q3 - Q1
df = df_time[~((df_time < (Q1 - 1.5 * IQR)) |(df_time > (Q3 + 1.5 * IQR))).any(axis=1)]

#Calculate the confidence intervals with bootstrap method
n_samples = 1000

conf_int = pd.DataFrame(
    {col: df_time[col].sample(n_samples, replace=True).quantile([0.025, 0.975]) for col in df_time.columns}
).transpose()

# Plot the median of each column
fig, axs = plt.subplots()
axs.set_xlabel("Sample")
axs.set_ylabel("Median")
axs.set_title("Median of Samples with Confidence Interval")
for col in df_time.columns:
    axs.errorbar(col,df_time[col].median(),yerr=[[df_time[col].median()-conf_int.loc[col].values[0]],[conf_int.loc[col].values[1]-df_time[col].median()]], fmt='o')

plt.show()