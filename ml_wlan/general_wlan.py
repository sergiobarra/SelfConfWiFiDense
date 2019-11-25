import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import sys
import io
from beautifultable import BeautifulTable
import time
from tqdm import tqdm
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import csv


# ---- Functions -----
# Compute model accuracy
def compute_model_accuracy(model, x, y):
    y_pred = model.predict(x)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    mae = metrics.mean_absolute_error(y, y_pred)
    # print('- RMSE: %.4f' % rmse)
    # print('- MAE: %.4f' % mae)
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
    # print('- best_conf ix:', pred_best_conf_ix)

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

    # print("- Perf chosen / optimal: %.2f / %.2f --> Abs error: %.2f --> Optimal (0,1)? %d"
    #       % (chosen_y, objective_y, abs_error, optimal_prediction))

    return chosen_y, objective_y, abs_error, optimal_prediction


# Get average accuracy of a given model
@ignore_warnings(category=ConvergenceWarning)
def get_average_model_accuracy(model, df_x, df_y, num_rand_trials, train_ratio, maximize=True,
                               optimal_margin=0, neural_net=False):
    print(type(model))

    rmse_train_tot = 0
    mae_train_tot = 0
    rmse_test_tot = 0
    mae_test_tot = 0
    optimal_tot = 0

    test_ratio = 1 - train_ratio

    for trial in tqdm(range(1, num_rand_trials + 1)):

        # print('---- Trial %d' % trial)
        # Pick random train / test data samples
        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(df_x, df_y, test_size=test_ratio,
                                             random_state=trial)
        # Train model
        if neural_net:
            model.fit(x_train, y_train, epochs=150, batch_size=50, verbose=0, validation_split=0.2)
        else:
            model.fit(x_train, y_train)

        # Error metrics
        rmse_train, mae_train = compute_model_accuracy(model, x_train, y_train)
        rmse_train_tot += rmse_train
        mae_train_tot += mae_train
        rmse_test, mae_test = compute_model_accuracy(model, x_test, y_test)
        rmse_test_tot += rmse_test
        mae_test_tot += mae_test

        # Infer optimal through exhaustive search
        chosen_y, objective_y, abs_error, optimal_prediction = get_predicted_optimal(
            model, df_x, df_y, maximize=maximize, optimal_margin=optimal_margin)
        if optimal_prediction:
            optimal_tot += 1

    # Average accuracy metrics
    rmse_train_av = rmse_train_tot / num_rand_trials
    mae_train_av = mae_train_tot / num_rand_trials
    rmse_test_av = rmse_test_tot / num_rand_trials
    mae_test_av = mae_test_tot / num_rand_trials
    optimal_av = optimal_tot / num_rand_trials

    # Print accuracy metrics
    table = BeautifulTable()
    table.column_headers = ["RMSE-train", "MAE-train", "RMSE-test", "MAE-test", "Optimal"]
    table.append_row(["%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                      "%.4f" % mae_test_av, "%.4f" % optimal_av])
    print("\n")
    print(table)

    return rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av


# Plot pdf of a output data
def plot_data_pdf(data, xlabel, ylabel):
    print('Plotting output pdf...')
    plt.figure()
    ax = data.plot.kde()
    plt.xlim(left=0)  # adjust the left leaving right unchanged
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()


print('******************************************************')
print('ML WLAN problem through different models')
print('******************************************************')

# --------- GET TRAINING SET ---------
# Whole raw data set
dataset_path = 'problem3_dataset.csv'
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

# --------- DEFINE MODELS ---------
# Linear regression with normal equation
linreg_ne = linear_model.LinearRegression()
# Linear regression with gradient descent
linreg_gd = linear_model.SGDRegressor(alpha=0.0001, average=False, early_stopping=False,
                                      epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                                      learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                                      n_iter_no_change=10, penalty='l2', power_t=0.25, random_state=None,
                                      shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
                                      warm_start=False)
# SVM SVR
svm_svr = svm.SVR(gamma='auto')
# SVM linear SVR
svm_linsvr = svm.LinearSVR()
# Feedforward Neural Net: we are using the 30 input variables (p_A_0,...,Ptx_F), along with two hidden layers of 12
# and 8 neurons respectively, and finally using the linear activation function to process the output.
nn = Sequential()
nn.add(Dense(12, input_dim=30, kernel_initializer='normal', activation='relu'))
nn.add(Dense(8, activation='relu'))
nn.add(Dense(1, activation='linear'))
# model_throughput.summary()
# mean_squared_error (mse) and mean_absolute_error (mae) are our loss functions
nn.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# --------- LEARN PARAMETERS (ML) & INFER OPTIMAL CONFIGURATION ---------
NUM_RAND_TRIALS = 100
NUM_RAND_TRIALS_SVM_SVR = 10    # To assign less trials since it takes much more time
train_ratio = 0.01
test_ratio = 1 - train_ratio
print('Train/Test ratio: ', train_ratio, '/', test_ratio)
print('Train/Test no. samples: ', round(train_ratio * num_samples), '/', round(test_ratio * num_samples))
save_results = True

# --- Optimize throuhgput ---
print('-------------------------')
print('OPTIMIZATION 1: min throughput [Mbps]')
df_y = df_min_throughput
OPTIMAL_THROUGHPUT_MARGIN = 1  # if achieved throughput [Mbps] >= optimal throughput - OPTIMAL_THROUGHPUT_MARGIN, OKAY
# print('\nOutput (y):')
# print(df_y.head(NUM_HEAD_LINES))
# plot_data_pdf(df_y, "y: Min throughput [Mbps]", "pdf(y)")

objective_y = df_y.max()
objective_y_ix = df_y.idxmax()
optimal_confs_ix = df_y.loc[df_y == objective_y]
num_optimal_confs = len(optimal_confs_ix)
pd.options.display.max_colwidth = 100

print('objective_y = %.2f Mbps' % objective_y)
print('objective_y_ix = %d' % objective_y_ix)
print('num_optimal_confs = %d' % num_optimal_confs)

print(df.loc[objective_y_ix, 'scenario'])

if save_results:
    filename_csv = 'results_minthroughput_ratio_%.4f.csv' % train_ratio

    with open(filename_csv, 'w', newline='') as results_csv:
        results_csv_writer = csv.writer(results_csv, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_csv_writer.writerow(['Model', 'RMSE-train', 'MAE-train', 'RMSE-test', 'MAE-test', 'Optimal'])

        print('- Linear regression with NE:')
        rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
            get_average_model_accuracy(linreg_ne, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
                                       optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
        results_csv_writer.writerow(["LR-NE", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])

        # print('- Linear regression with GD:')
        # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
        #     get_average_model_accuracy(linreg_gd, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
        #                                optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
        # results_csv_writer.writerow(["LR-GD", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
        #                              "%.4f" % mae_test_av, "%.4f" % optimal_av])

        print('- SVM SVR:')
        rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
            get_average_model_accuracy(svm_svr, df_x, df_y, NUM_RAND_TRIALS_SVM_SVR, train_ratio, maximize=True,
                                       optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
        results_csv_writer.writerow(["SVM-SVR", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])

        print('- SVM Linear:')
        rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
            get_average_model_accuracy(svm_linsvr, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
                                       optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
        results_csv_writer.writerow(["SVM-LinSVR", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])

        print('- Neural Net:')
        rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
            get_average_model_accuracy(nn, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
                                       optimal_margin=OPTIMAL_THROUGHPUT_MARGIN, neural_net=True)
        results_csv_writer.writerow(["NN", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])

print('-------------------------')
print('OPTIMIZATION 2: max delay [ms]')
df_y = df_max_delay
OPTIMAL_DELAY_MARGIN = 1  # if achieved delay [ms] <= optimal delay + OPTIMAL_DELAY_MARGIN, OKAY
# print('\nOutput (y):')
# print(df_y.head(NUM_HEAD_LINES))
# plot_data_pdf(df_y, "y: Max delay [ms]", "pdf(y)")
objective_y = df_y.min()
objective_y_ix = df_y.idxmin()
optimal_confs_ix = df_y.loc[df_y == objective_y]
num_optimal_confs = len(optimal_confs_ix)
pd.options.display.max_colwidth = 100

print('objective_y = %.2f ms' % objective_y)
print('objective_y_ix = %d' % objective_y_ix)
print('num_optimal_confs = %d' % num_optimal_confs)

print(df.loc[objective_y_ix, 'scenario'])

if save_results:
    filename_csv = 'results_maxdelay_ratio_%.4f.csv' % train_ratio
    with open(filename_csv, 'w', newline='') as results_csv:
        results_csv_writer = csv.writer(results_csv, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_csv_writer.writerow(['Model', 'RMSE-train', 'MAE-train', 'RMSE-test', 'MAE-test', 'Optimal'])

        print('- Linear regression with NE:')
        rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
            get_average_model_accuracy(linreg_ne, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=False,
                                       optimal_margin=OPTIMAL_DELAY_MARGIN)
        results_csv_writer.writerow(["LR-NE", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])

        # print('- Linear regression with GD:')
        # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
        #     get_average_model_accuracy(linreg_gd, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=False,
        #                                optimal_margin=OPTIMAL_DELAY_MARGIN)
        # results_csv_writer.writerow(["LR-GD", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
        #                              "%.4f" % mae_test_av, "%.4f" % optimal_av])

        print('- SVM SVR:')
        rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
            get_average_model_accuracy(svm_svr, df_x, df_y, NUM_RAND_TRIALS_SVM_SVR, train_ratio, maximize=False,
                                       optimal_margin=OPTIMAL_DELAY_MARGIN)
        results_csv_writer.writerow(["SVM-SVR", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])

        print('- SVM Linear:')
        rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
            get_average_model_accuracy(svm_linsvr, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=False,
                                       optimal_margin=OPTIMAL_DELAY_MARGIN)
        results_csv_writer.writerow(["SVM-LinSVR", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])

        print('- Neural Net:')
        rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
            get_average_model_accuracy(nn, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=False,
                                       optimal_margin=OPTIMAL_DELAY_MARGIN, neural_net=True)
        results_csv_writer.writerow(["NN", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])


print('\n\n******* LIFE IS NICE. DO NOT FORGET :D **************')
print(".    --.--.---..---.  --.-- .-.    .   .--.-- .--..---.    .--.")
print("|      |  |    |        |  (   )   |\  |  |  :    |        |   :")
print("|      |  |--- |---     |   `-.    | \ |  |  |    |---    o|   |")
print("|      |  |    |        |  (   )   |  \|  |  :    |        |   ;")
print("'---'--'--'    '---'  --'-- `-'    '   '--'-- `--''---'   o'--'")
