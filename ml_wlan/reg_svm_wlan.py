import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import sys
import io

print('******************************************************')
print('ML WLAN problem through SUPPORT VECTOR MACHINE')
print('******************************************************')

# Whole raw data set
dataset_path = 'problem1_dataset.csv';
print('Reading dataset', dataset_path, '...')
# df = pd.read_csv('problem1_dataset_demo.csv', sep=';')
df = pd.read_csv(dataset_path, sep=';')
print(' - Dataset read!')

# print(df.inf.info())  # Overview of data set
num_samples = len(df.index)
print(' - Number of samples in the data set: ', str(num_samples))

# Create new feature: mean throughput
df_throughput = df[['s_A', 's_B', 's_C', 's_D', 's_E', 's_F']]
df_mean_throughput = df_throughput.mean(axis=1)
df_min_throughput = df['s_min']
df_mean_delay = df['d_av']
df_max_delay = df['d_max']

# ---------- Tune input data ----------
# Raw input features of interest
df_x = df[['p_A', 'p_B', 'p_C', 'p_D', 'p_E', 'p_F',
           'Ptx_A', 'Ptx_B', 'Ptx_C', 'Ptx_D', 'Ptx_E', 'Ptx_F']]

# Add every possible value that each feature can take (primary = [1,4])
# - The idea is to avoid missing some value in the input when getting the corresponding dummy variables.
df_x_redundant = pd.DataFrame({'p_A': [0, 1, 2, 3],
                               'p_B': [0, 1, 2, 3],
                               'p_C': [0, 1, 2, 3],
                               'p_D': [0, 1, 2, 3],
                               'p_E': [0, 1, 2, 3],
                               'p_F': [0, 1, 2, 3],
                               'Ptx_A': [0, 0, 1, 1],
                               'Ptx_B': [0, 0, 1, 1],
                               'Ptx_C': [0, 0, 1, 1],
                               'Ptx_D': [0, 0, 1, 1],
                               'Ptx_E': [0, 0, 1, 1],
                               'Ptx_F': [0, 0, 1, 1]})
# print('df_x_original:\n', df_x_original)
df_x = pd.concat([df_x, df_x_redundant], sort=False, ignore_index=True)
# print('df_x:\n', df_x)

# Get dummies (binary features) of the primary channel feature: e.g., p_A --> p_A1, p_A2, p_A3 and p_A4
df_p_A = pd.get_dummies(df_x['p_A'])
df_p_B = pd.get_dummies(df_x['p_B'])
df_p_C = pd.get_dummies(df_x['p_C'])
df_p_D = pd.get_dummies(df_x['p_D'])
df_p_E = pd.get_dummies(df_x['p_E'])
df_p_F = pd.get_dummies(df_x['p_F'])
df_p_A.rename(columns={0: 'p_A_0', 1: 'p_A_1', 2: 'p_A_2', 3: 'p_A_3'}, inplace=True)
df_p_B.rename(columns={0: 'p_B_0', 1: 'p_B_1', 2: 'p_B_2', 3: 'p_B_3'}, inplace=True)
df_p_C.rename(columns={0: 'p_C_0', 1: 'p_C_1', 2: 'p_C_2', 3: 'p_C_3'}, inplace=True)
df_p_D.rename(columns={0: 'p_D_0', 1: 'p_D_1', 2: 'p_D_2', 3: 'p_D_3'}, inplace=True)
df_p_E.rename(columns={0: 'p_E_0', 1: 'p_E_1', 2: 'p_E_2', 3: 'p_E_3'}, inplace=True)
df_p_F.rename(columns={0: 'p_F_0', 1: 'p_F_1', 2: 'p_F_2', 3: 'p_F_3'}, inplace=True)

frames = [df_p_A, df_p_B, df_p_C, df_p_D, df_p_E, df_p_F,
          df_x['Ptx_A'], df_x['Ptx_B'], df_x['Ptx_C'], df_x['Ptx_D'], df_x['Ptx_E'], df_x['Ptx_F']]
df_x = pd.concat(frames, axis=1, join='inner')

# remove redundant entries
df_x.drop(df_x.tail(len(df_x_redundant.index)).index, inplace=True)  # drop last n rows

# ---------- EoF Tune input data ----------

train_ratio = 0.001
test_ratio = 1 - train_ratio
print('Train/Test ratio: ', train_ratio, '/', test_ratio)

# ---------- Optimize throuhgput ----------
print('-------------------------')
print('OPTIMIZATION 1: min throughput [Mbps]')

df_y = df_min_throughput
# plt.figure()
# ax = df_y.plot.kde()
# plt.xlabel("y: Min throughput [Mbps]")
# plt.ylabel("pdf(y)")
# plt.show()
# plt.close()

NUM_HEAD_LINES = 3
print('Input (X):')
print(df_x.head(NUM_HEAD_LINES))
print('\nOutput (y):')
print(df_y.head(NUM_HEAD_LINES))

X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(df_x, df_y, test_size=test_ratio,
                                     random_state=1)
print('Train/Test no. samples: ', len(X_train), '/', len(X_test))

# Create our two SVR models
h1 = svm.SVR()
h2 = svm.LinearSVR()

# Train the models
h1.fit(X_train, y_train)
h2.fit(X_train, y_train)

# Compute error
y_train_pred_h1 = h1.predict(X_train)
y_test_pred_h1 = h1.predict(X_test)
mse_train_h1 = metrics.mean_squared_error(y_train, y_train_pred_h1)
mae_train_h1 = metrics.mean_absolute_error(y_train, y_train_pred_h1)
mse_test_h1 = metrics.mean_squared_error(y_test, y_test_pred_h1)
mae_test_h1 = metrics.mean_absolute_error(y_test, y_test_pred_h1)
print('h1 - RMSE train: ', np.sqrt(mse_train_h1), ' Mbps')
print('h1 - MAE train: ', mae_train_h1, ' Mbps')
print('h1 - RMSE test: ', np.sqrt(mse_test_h1), ' Mbps')
print('h1 - MAE test: ', mae_test_h1, ' Mbps')

y_train_pred_h2 = h2.predict(X_train)
y_test_pred_h2 = h2.predict(X_test)
mse_train_h2 = metrics.mean_squared_error(y_train, y_train_pred_h2)
mae_train_h2 = metrics.mean_absolute_error(y_train, y_train_pred_h2)
mse_test_h2 = metrics.mean_squared_error(y_test, y_test_pred_h2)
mae_test_h2 = metrics.mean_absolute_error(y_test, y_test_pred_h2)
print('h2 - RMSE train: ', np.sqrt(mse_train_h2), ' Mbps')
print('h2 - MAE train: ', mae_train_h2, ' Mbps')
print('h2 - RMSE test: ', np.sqrt(mse_test_h2), ' Mbps')
print('h2 - MAE test: ', mae_test_h2, ' Mbps')

# Pick optimal conf
actual_best_y = df_y.max()
print('Actual best min throughput: ', actual_best_y, 'Mbps')
print('Picking the optimal through h1...')
exhaustive_prediction = h1.predict(df_x)
pred_best_conf_ix = np.argmax(exhaustive_prediction)
pred_best_y = exhaustive_prediction[pred_best_conf_ix]
actual_pred_best_y = df_y.loc[pred_best_conf_ix]
print('- best_conf ix:', pred_best_conf_ix)
print('- Predicted/actual min throughput: ', pred_best_y, '/', actual_pred_best_y, 'Mbps')

print('Picking the optimal through h2...')
exhaustive_prediction = h2.predict(df_x)
pred_best_conf_ix = np.argmax(exhaustive_prediction)
pred_best_y = exhaustive_prediction[pred_best_conf_ix]
actual_pred_best_y = df_y.loc[pred_best_conf_ix]
print('- best_conf ix:', pred_best_conf_ix)
print('- Predicted/actual min throughput: ', pred_best_y, '/', actual_pred_best_y, 'Mbps')

# ---------- Optimize max delay ----------
print('-------------------------')
print('OPTIMIZATION 2: max delay [ms]')

df_y = df_max_delay
# plt.figure()
# ax = df_y.plot.kde()
# plt.xlabel("y: Min throughput [Mbps]")
# plt.ylabel("pdf(y)")
# plt.show()
# plt.close()

print('Input (X):')
print(df_x.head(NUM_HEAD_LINES))
print('\nOutput (y):')
print(df_y.head(NUM_HEAD_LINES))

X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(df_x, df_y, test_size=test_ratio,
                                     random_state=1)
print('Train/Test no. samples: ', len(X_train), '/', len(X_test))

# Create our two SVR models
h1 = svm.SVR()
h2 = svm.LinearSVR()

# Train the models
h1.fit(X_train, y_train)
h2.fit(X_train, y_train)

# Compute error
y_train_pred_h1 = h1.predict(X_train)
y_test_pred_h1 = h1.predict(X_test)
mse_train_h1 = metrics.mean_squared_error(y_train, y_train_pred_h1)
mae_train_h1 = metrics.mean_absolute_error(y_train, y_train_pred_h1)
mse_test_h1 = metrics.mean_squared_error(y_test, y_test_pred_h1)
mae_test_h1 = metrics.mean_absolute_error(y_test, y_test_pred_h1)
print('h1 - RMSE train: ', np.sqrt(mse_train_h1), ' ms')
print('h1 - MAE train: ', mae_train_h1, ' ms')
print('h1 - RMSE test: ', np.sqrt(mse_test_h1), ' ms')
print('h1 - MAE test: ', mae_test_h1, ' ms')

y_train_pred_h2 = h2.predict(X_train)
y_test_pred_h2 = h2.predict(X_test)
mse_train_h2 = metrics.mean_squared_error(y_train, y_train_pred_h2)
mae_train_h2 = metrics.mean_absolute_error(y_train, y_train_pred_h2)
mse_test_h2 = metrics.mean_squared_error(y_test, y_test_pred_h2)
mae_test_h2 = metrics.mean_absolute_error(y_test, y_test_pred_h2)
print('h2 - RMSE train: ', np.sqrt(mse_train_h2), ' ms')
print('h2 - MAE train: ', mae_train_h2, ' ms')
print('h2 - RMSE test: ', np.sqrt(mse_test_h2), ' ms')
print('h2 - MAE test: ', mae_test_h2, ' ms')

# Pick optimal conf
actual_best_y = df_y.min()
print('Actual best max delay: ', actual_best_y, 'ms')
print('Picking the optimal through h1...')
exhaustive_prediction = h1.predict(df_x)
pred_best_conf_ix = np.argmin(exhaustive_prediction)
pred_best_y = exhaustive_prediction[pred_best_conf_ix]
actual_pred_best_y = df_y.loc[pred_best_conf_ix]
print('- best_conf ix:', pred_best_conf_ix)
print('- Predicted/actual max delay: ', pred_best_y, '/', actual_pred_best_y, 'ms')

print('Picking the optimal through h2...')
exhaustive_prediction = h2.predict(df_x)
pred_best_conf_ix = np.argmin(exhaustive_prediction)
pred_best_y = exhaustive_prediction[pred_best_conf_ix]
actual_pred_best_y = df_y.loc[pred_best_conf_ix]
print('- best_conf ix:', pred_best_conf_ix)
print('- Predicted/actual max delay: ', pred_best_y, '/', actual_pred_best_y, 'ms')