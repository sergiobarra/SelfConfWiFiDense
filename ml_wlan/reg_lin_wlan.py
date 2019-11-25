import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import sys
import io


# ---- Functions -----
# Compute model accuracy
def compute_model_accuracy(model, x, y):
    y_pred = model.predict(x)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    mae = metrics.mean_absolute_error(y, y_pred)
    print('- RMSE: %.4f' % rmse)
    print('- MAE: %.4f' % mae)
    return rmse, mae


# Get predicted optimal through exhaustive search
def get_predicted_optimal(model, df_x, df_y, maximize=True, optimal_margin=0):
    exhaust_pred = model.predict(df_x)

    if maximize:
        pred_best_conf_ix = np.argmax(exhaust_pred)
        objective_y = df_y.max()
    else:
        pred_best_conf_ix = np.argmin(exhaust_pred)
        objective_y = df_y.min()

    # pred_best_y = exhaust_pred[pred_best_conf_ix]
    chosen_y = df_y.loc[pred_best_conf_ix]
    abs_error = abs(objective_y - chosen_y)
    print('- best_conf ix:', pred_best_conf_ix)

    if maximize:
        if chosen_y >= (objective_y - optimal_margin):
            optimal_prediction = True
        else:
            optimal_prediction = False
    else:
        if chosen_y <= (objective_y + optimal_margin):
            optimal_prediction = True
        else:
            optimal_prediction = False

    print("- Perf chosen / optimal: %.2f / %.2f --> Abs error: %.2f --> Optimal (0,1)? %d"
          % (chosen_y, objective_y, abs_error, optimal_prediction))

    return chosen_y, objective_y, abs_error, optimal_prediction


# Plot pdf of a output data
def plot_data_pdf(data, xlabel, ylabel):
    print('Plotting output pdf...')
    plt.figure()
    ax = data.plot.kde()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()


print('******************************************************')
print('ML WLAN problem through MULTIVARIATE LINEAR REGRESSION')
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

NUM_HEAD_LINES = 3
print('Input (X):')
print(df_x.head(NUM_HEAD_LINES))

# ---------- EoF Tune input data ----------

train_ratio = 0.001
test_ratio = 1 - train_ratio
print('Train/Test ratio: ', train_ratio, '/', test_ratio)

# ---------- Optimize throuhgput ----------
print('-------------------------')
print('OPTIMIZATION 1: min throughput [Mbps]')

# Define output
OPTIMAL_THROUGHPUT_MARGIN = 1  # if achieved throughput [Mbps] >= optimal throughput - OPTIMAL_THROUGHPUT_MARGIN, OKAY
df_y = df_min_throughput
print('\nOutput (y):')
print(df_y.head(NUM_HEAD_LINES))
# plot_data_pdf(df_y, "y: Min throughput [Mbps]", "pdf(y)")

X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(df_x, df_y, test_size=test_ratio,
                                     random_state=1)
print('Train/Test no. samples: ', len(X_train), '/', len(X_test))

print('--------------------')
print('Applying Normal Equation...')
linreg_ne = linear_model.LinearRegression()
linreg_ne.fit(X_train, y_train)
print('Test:')
rmse_train, mae_train = compute_model_accuracy(linreg_ne, X_train, y_train)
print('Train:')
rmse_test, mae_test = compute_model_accuracy(linreg_ne, X_test, y_test)
print('Optimal:')
# Get predicted optimal through exhaustive search
chosen_y, objective_y, abs_error, optimal_prediction = get_predicted_optimal(linreg_ne, df_x, df_y, maximize=True,
                                                                             optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)

print('--------------------')
print('Running SGD regression...')
old_stdout = sys.stdout
sys.stdout = mystdout = io.StringIO()
linreg_gd = linear_model.SGDRegressor(alpha=0.0001, average=False, early_stopping=False,
                                      epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                                      learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                                      n_iter_no_change=10, penalty='l2', power_t=0.25, random_state=None,
                                      shuffle=True, tol=0.001, validation_fraction=0.1, verbose=1,
                                      warm_start=False)
linreg_gd.fit(X_train, y_train)
print('SGD finished')
sys.stdout = old_stdout
loss_history = mystdout.getvalue()
loss_list = []
for line in loss_history.split('\n'):
    if len(line.split("loss: ")) == 1:
        continue
    loss_list.append(float(line.split("loss: ")[-1]))
# plt.figure()
# plt.plot(np.arange(len(loss_list)), loss_list)
# plt.xlabel("Time in epochs")
# plt.ylabel("Loss")
# plt.show()
# plt.close()

print('Test:')
rmse_train, mae_train = compute_model_accuracy(linreg_gd, X_train, y_train)
print('Train:')
rmse_test, mae_test = compute_model_accuracy(linreg_gd, X_test, y_test)
print('Optimal:')
# Get predicted optimal through exhaustive search
chosen_y, objective_y, abs_error, optimal_prediction = get_predicted_optimal(linreg_gd, df_x, df_y, maximize=True,
                                                                             optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)

# ---------- Optimize max delay ----------
print('-------------------------')
print('OPTIMIZATION 2: Max. delay [ms]')
OPTIMAL_DELAY_MARGIN = 1  # if achieved delay [ms] <= optimal delay + OPTIMAL_DELAY_MARGIN, OKAY
df_y = df_max_delay
print('\nOutput (y):')
print(df_y.head(NUM_HEAD_LINES))
# plot_data_pdf(df_y, "y: Max delay [ms]", "pdf(y)")

X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(df_x, df_y, test_size=test_ratio,
                                     random_state=1)
print('Train/Test no. samples: ', len(X_train), '/', len(X_test))

print('--------------------')
print('Applying Normal Equation...')
linreg_ne = linear_model.LinearRegression()
linreg_ne.fit(X_train, y_train)
print('Test:')
rmse_train, mae_train = compute_model_accuracy(linreg_ne, X_train, y_train)
print('Train:')
rmse_test, mae_test = compute_model_accuracy(linreg_ne, X_test, y_test)
print('Optimal:')
# Get predicted optimal through exhaustive search
chosen_y, objective_y, abs_error, optimal_prediction = get_predicted_optimal(linreg_ne, df_x, df_y, maximize=False,
                                                                             optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)

print('--------------------')
print('Running SGD regression...')
old_stdout = sys.stdout
sys.stdout = mystdout = io.StringIO()
linreg_gd = linear_model.SGDRegressor(alpha=0.0001, average=False, early_stopping=False,
                                      epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                                      learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                                      n_iter_no_change=10, penalty='l2', power_t=0.25, random_state=None,
                                      shuffle=True, tol=0.001, validation_fraction=0.1, verbose=1,
                                      warm_start=False)
linreg_gd.fit(X_train, y_train)
print('SGD finished')
sys.stdout = old_stdout
loss_history = mystdout.getvalue()
loss_list = []
for line in loss_history.split('\n'):
    if len(line.split("loss: ")) == 1:
        continue
    loss_list.append(float(line.split("loss: ")[-1]))
# plt.figure()
# plt.plot(np.arange(len(loss_list)), loss_list)
# plt.xlabel("Time in epochs")
# plt.ylabel("Loss")
# plt.show()
# plt.close()

print('Test:')
rmse_train, mae_train = compute_model_accuracy(linreg_gd, X_train, y_train)
print('Train:')
rmse_test, mae_test = compute_model_accuracy(linreg_gd, X_test, y_test)
print('Optimal:')
# Get predicted optimal through exhaustive search
chosen_y, objective_y, abs_error, optimal_prediction = get_predicted_optimal(linreg_gd, df_x, df_y, maximize=False,
                                                                             optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
