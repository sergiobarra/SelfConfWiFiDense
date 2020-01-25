import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.optimizers import SGD
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
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
import random

NUM_RAND_TRIALS = 10    # No. of training datasets to randomly pick
TRAIN_RATIO = 0.20      # No. of train samples / No. of samples
NUM_EPOCHS_FFNN = 50    # No. of epochs for training the FF-NN
NUM_EPOCHS_CNN = 50     # No. of epochs for training the CNN
BATCH_SIZE = 1000       # Batch size for training the NNs
PLOT_NN_ERROR = False   # Plot the evolution of NN error?


# ---- Custom Functions -----

# Define custom loss
def custom_loss(y_true, y_pred):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    print(y_true)
    return K.mean(K.sqrt(((y_true - y_pred) ** 2)))


# Compute prediction accuracy of the "exhaustive search in train" model
def compute_random_accuracy(x, y):
    zero_array = np.zeros((x.shape[0]))
    rmse = np.sqrt(metrics.mean_squared_error(y, zero_array))
    mae = metrics.mean_absolute_error(y, zero_array)
    # print('- RMSE: %.4f' % rmse)
    # print('- MAE: %.4f' % mae)
    return rmse, mae


# Compute model accuracy
def compute_model_accuracy(model, x, y):
    y_pred = model.predict(x)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    mae = metrics.mean_absolute_error(y, y_pred)
    # print('- RMSE: %.4f' % rmse)
    # print('- MAE: %.4f' % mae)
    return rmse, mae


# Get predicted optimal
def get_predicted_optimal(model, df_x, df_y, maximize=True, optimal_margin=0, is_cnn=False):

    if is_cnn:
        x = np.asarray(df_x)
        x = x.reshape(len(x), 30, 1)
        exhaust_pred = model.predict(x)
        #mae = metrics.mean_absolute_error(df_y, exhaust_pred)
        #print('- MAE CNN: %.4f' % mae)
    else:
        exhaust_pred = model.predict(df_x)
        #mae = metrics.mean_absolute_error(df_y, exhaust_pred)
        #print('- MAE: %.4f' % mae)

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


# Get predicted best (actually known) in training set
def get_predicted_optimal_record_best_in_train(df_x, df_y, y_train):
    y_record_pred = y_train.max()  # known maximum in training set
    print('y_record_pred:', y_record_pred)
    objective_y = df_y.max()  # Known maximum in whole data set
    abs_error_record = abs(objective_y - y_record_pred)
    print('abs_error_record:', abs_error_record)
    return abs_error_record


# Get predicted top-10 best configs through "exhaustive search in train" model
def get_predicted_optimal_record_random(df_x, df_y):
    record_pred_ind = random.sample(range(1, df_x.shape[0]), 10)
    print('record_pred_ind:', record_pred_ind)
    y_record_pred = df_y.iloc[record_pred_ind].values
    print('y_record_pred:', y_record_pred)
    objective_y = df_y.max()  # Known maximum
    abs_error_record = abs(objective_y - y_record_pred)
    print('abs_error_record:', abs_error_record)
    return abs_error_record


# Get predicted top-10 best configs
def get_predicted_optimal_record(model, df_x, df_y,is_cnn=False):
    objective_y = df_y.max()  # Known maximum
    # print('objective_y:', objective_y)

    if is_cnn:
        x = np.asarray(df_x)
        x = x.reshape(len(x), 30, 1)
        exhaust_pred = model.predict(x).ravel()  # Predict output of every possible input
    else:
        exhaust_pred = model.predict(df_x).ravel()  # Predict output of every possible input

    # print('exhaust_pred:', exhaust_pred)
    # print(type(exhaust_pred))
    best_pred_ix = exhaust_pred.argmax()
    # print('best_pred_ix:', best_pred_ix)
    best_pred = max(exhaust_pred)
    # print('best_pred:', best_pred)
    actual_best_pred = df_y.iloc[best_pred_ix]
    # print('actual_best_pred:', actual_best_pred)
    record_pred_ind = np.argpartition(exhaust_pred, -10)[-10:]  # Pick the top-10 performing predicted inputs
    # print('record_pred_ind:', record_pred_ind)
    record_pred = exhaust_pred[record_pred_ind]
    # print('record_pred:', record_pred)
    y_record_pred = df_y.iloc[record_pred_ind].values
    # print('y_record_pred:', y_record_pred)
    abs_error_record = abs(objective_y - y_record_pred)
    print('abs_error_record:', abs_error_record)
    return abs_error_record


# Plot NN error evolution
def plot_nn_error(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Get training and test loss histories
    mean_abs_error = history.history['mean_absolute_error']
    mean_abs_test_loss = history.history['val_mean_absolute_error']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, mean_abs_error, 'r--')
    plt.plot(epoch_count, mean_abs_test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('MAE [Mbps]')
    plt.show()


# Google's activation function
def swish(x, beta=1):
    return x * K.sigmoid(beta * x)


# Get average accuracy of "exhaustive search in train" model
def get_average_best_in_train_accuracy(df_x, df_y, num_rand_trials, train_ratio, maximize=True,
                                       optimal_margin=0, neural_net=False):
    rmse_train_tot = 0
    mae_train_tot = 0
    rmse_test_tot = 0
    mae_test_tot = 0
    optimal_tot = 0
    dif_optimal_tot = 0

    test_ratio = 1 - train_ratio

    for trial in tqdm(range(1, num_rand_trials + 1)):
        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(df_x, df_y, test_size=test_ratio,
                                             random_state=trial)

        # Random model that predicts 0 reward to every input
        # Error metrics
        rmse_train, mae_train = compute_random_accuracy(x_train, y_train)
        rmse_train_tot += rmse_train
        mae_train_tot += mae_train
        rmse_test, mae_test = compute_random_accuracy(x_test, y_test)
        rmse_test_tot += rmse_test
        mae_test_tot += mae_test

        # Infer optimal ---> pick random inputs
        abs_error_record = get_predicted_optimal_record_best_in_train(df_x, df_y, y_train)
        print('Minimum error: %.2f Mbps' % abs_error_record)
        dif_optimal_tot += abs_error_record

    # Average accuracy metrics
    rmse_train_av = rmse_train_tot / num_rand_trials
    mae_train_av = mae_train_tot / num_rand_trials
    rmse_test_av = rmse_test_tot / num_rand_trials
    mae_test_av = mae_test_tot / num_rand_trials
    prob_optimal_av = optimal_tot / num_rand_trials
    dif_optimal_av = dif_optimal_tot / num_rand_trials

    # Print accuracy metrics
    table = BeautifulTable()
    table.column_headers = ["RMSE-train", "MAE-train", "RMSE-test", "MAE-test", "Prob. Opt.", "Dif. Opt."]
    table.append_row(["%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                      "%.4f" % mae_test_av, "%.4f" % prob_optimal_av, "%.4f" % dif_optimal_av])
    print("\n")
    print(table)

    return rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, prob_optimal_av, dif_optimal_av


# Get average accuracy of random selection
def get_average_random_accuracy(df_x, df_y, num_rand_trials, train_ratio, maximize=True,
                                optimal_margin=0, neural_net=False):
    rmse_train_tot = 0
    mae_train_tot = 0
    rmse_test_tot = 0
    mae_test_tot = 0
    optimal_tot = 0
    dif_optimal_tot = 0

    test_ratio = 1 - train_ratio

    for trial in tqdm(range(1, num_rand_trials + 1)):
        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(df_x, df_y, test_size=test_ratio,
                                             random_state=trial)

        # Random model that predicts 0 reward to every input
        # Error metrics
        rmse_train, mae_train = compute_random_accuracy(x_train, y_train)
        rmse_train_tot += rmse_train
        mae_train_tot += mae_train
        rmse_test, mae_test = compute_random_accuracy(x_test, y_test)
        rmse_test_tot += rmse_test
        mae_test_tot += mae_test

        # Infer optimal ---> pick random inputs
        abs_error_record = get_predicted_optimal_record_random(df_x, df_y)
        print('Minimum error: %.2f Mbps' % min(abs_error_record))
        dif_optimal_tot += min(abs_error_record)

    # Average accuracy metrics
    rmse_train_av = rmse_train_tot / num_rand_trials
    mae_train_av = mae_train_tot / num_rand_trials
    rmse_test_av = rmse_test_tot / num_rand_trials
    mae_test_av = mae_test_tot / num_rand_trials
    prob_optimal_av = optimal_tot / num_rand_trials
    dif_optimal_av = dif_optimal_tot / num_rand_trials

    # Print accuracy metrics
    table = BeautifulTable()
    table.column_headers = ["RMSE-train", "MAE-train", "RMSE-test", "MAE-test", "Prob. Opt.", "Dif. Opt."]
    table.append_row(["%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                      "%.4f" % mae_test_av, "%.4f" % prob_optimal_av, "%.4f" % dif_optimal_av])
    print("\n")
    print(table)

    return rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, prob_optimal_av, dif_optimal_av


# Get average accuracy of a given model
@ignore_warnings(category=ConvergenceWarning)
def get_average_model_accuracy(model, df_x, df_y, num_rand_trials, train_ratio, maximize=True,
                               optimal_margin=0, neural_net=0, is_cnn = False):
    print(type(model))

    rmse_train_tot = 0
    mae_train_tot = 0
    rmse_test_tot = 0
    mae_test_tot = 0
    optimal_tot = 0
    dif_optimal_tot = 0

    test_ratio = 1 - train_ratio

    for trial in tqdm(range(1, num_rand_trials + 1)):

        # print('---- Trial %d' % trial)
        # Pick random train / test data samples
        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(df_x, df_y, test_size=test_ratio,
                                             random_state=trial)
        # Train model
        if neural_net == 1:

            model.load_weights('ffnn_default_weights.h5')

            history = model.fit(x_train, y_train, epochs=NUM_EPOCHS_FFNN, batch_size=BATCH_SIZE, verbose=1,
                                validation_split=0.2)
            if PLOT_NN_ERROR:
                plot_nn_error(history)

        elif neural_net == 2:

            model.load_weights('cnn_default_weights.h5')

            # x_train, x_test, y_train, y_test = \
            #     model_selection.train_test_split(np.asarray(df_x), np.asarray(df_y), test_size=test_ratio,
            #                                      shuffle=True)
            x_train = np.asarray(x_train)
            x_train = x_train.reshape(len(x_train), 30, 1)

            x_test = np.asarray(x_test)
            x_test = x_test.reshape(len(x_test), 30, 1)

            history = model.fit(x_train, y_train,
                                batch_size=BATCH_SIZE,
                                epochs=NUM_EPOCHS_CNN,
                                verbose=1,
                                validation_split=0.2)

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
            model, df_x, df_y, maximize=maximize, optimal_margin=optimal_margin, is_cnn=is_cnn)
        if optimal_prediction:
            optimal_tot += 1
        # dif_optimal_tot += abs_error
        abs_error_record = get_predicted_optimal_record(model, df_x, df_y,is_cnn=is_cnn)
        print('Minimum error: %.2f Mbps' % min(abs_error_record))
        dif_optimal_tot += min(abs_error_record)

    # Average accuracy metrics
    rmse_train_av = rmse_train_tot / num_rand_trials
    mae_train_av = mae_train_tot / num_rand_trials
    rmse_test_av = rmse_test_tot / num_rand_trials
    mae_test_av = mae_test_tot / num_rand_trials
    prob_optimal_av = optimal_tot / num_rand_trials
    dif_optimal_av = dif_optimal_tot / num_rand_trials

    # Print accuracy metrics
    table = BeautifulTable()
    table.column_headers = ["RMSE-train", "MAE-train", "RMSE-test", "MAE-test", "Prob. Opt.", "Dif. Opt."]
    table.append_row(["%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                      "%.4f" % mae_test_av, "%.4f" % prob_optimal_av, "%.4f" % dif_optimal_av])
    print("\n")
    print(table)

    return rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, prob_optimal_av, dif_optimal_av


# Plot pdf of a output data
def plot_data_pdf(data, xlabel, ylabel):
    print('Plotting output pdf...')
    plt.figure()
    ax = data.plot.kde()
    print('Setting x-axis limits from 0 to max (%.2f)...' % data.max())
    plt.xlim(left=0)  # adjust the left leaving right unchanged
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(right=data.max())
    plt.xticks(np.arange(0, data.max() + 1, 5))
    plt.show()
    plt.close()


print('******************************************************')
print('ML WLAN problem through different models')
print('******************************************************')

get_custom_objects().update({'swish': Activation(swish)})

# --------- GET TRAINING SET ---------
# Whole raw data set
dataset_path = 'problem5_dataset.csv'
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
df_y = df_min_throughput

# Plot correlation heatmap
# df_dataset = pd.concat([df_x, df_y], axis=1, join='inner')
# plt.figure()
# corr = df_dataset.corr()
# ax = sns.heatmap(
#     corr,
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True
# )
# ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right'
# )
# plt.show()

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

df_x.loc[(df_x.Ptx_A == 14), 'Ptx_A'] = 0
df_x.loc[(df_x.Ptx_B == 14), 'Ptx_B'] = 0
df_x.loc[(df_x.Ptx_C == 14), 'Ptx_C'] = 0
df_x.loc[(df_x.Ptx_D == 14), 'Ptx_D'] = 0
df_x.loc[(df_x.Ptx_E == 14), 'Ptx_E'] = 0
df_x.loc[(df_x.Ptx_F == 14), 'Ptx_F'] = 0
df_x.loc[(df_x.Ptx_A == 20), 'Ptx_A'] = 1
df_x.loc[(df_x.Ptx_B == 20), 'Ptx_B'] = 1
df_x.loc[(df_x.Ptx_C == 20), 'Ptx_C'] = 1
df_x.loc[(df_x.Ptx_D == 20), 'Ptx_D'] = 1
df_x.loc[(df_x.Ptx_E == 20), 'Ptx_E'] = 1
df_x.loc[(df_x.Ptx_F == 20), 'Ptx_F'] = 1

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

# Shallow NN
nn_shallow = Sequential()
nn_shallow_num_nodes_hidden_layer_1 = 15
nn_shallow.add(Dense(nn_shallow_num_nodes_hidden_layer_1, input_dim=30, kernel_initializer='normal', activation='relu'))
# nn_shallow.add(Dropout(0.5))
nn_shallow.add(Dense(1, activation='linear'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
nn_shallow.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# Medium deep NN
nn_quasideep = Sequential()
nn_quasideep_num_nodes_hidden_layer_1 = 25
nn_quasideep_num_nodes_hidden_layer_2 = 15
nn_quasideep.add(
    Dense(nn_quasideep_num_nodes_hidden_layer_1, input_dim=30, kernel_initializer='normal', activation='relu'))
nn_quasideep.add(Dense(nn_quasideep_num_nodes_hidden_layer_2, activation='relu'))
nn_quasideep.add(Dense(1, activation='linear'))
nn_quasideep.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# Super deep NN
nn_deep = Sequential()
nn_deep_num_nodes_hidden_layer_1 = 512
nn_deep_num_nodes_hidden_layer_2 = 256
nn_deep_num_nodes_hidden_layer_3 = 196
nn_deep_num_nodes_hidden_layer_4 = 128
nn_deep_num_nodes_hidden_layer_5 = 96
nn_deep.add(Dense(nn_deep_num_nodes_hidden_layer_1, input_dim=30, kernel_initializer='normal', activation='relu'))
# nn.add(Dense(num_nodes_hidden_layer_1, input_dim=30, kernel_initializer='normal', activation='swish'))
nn_deep.add(Dense(nn_deep_num_nodes_hidden_layer_2, activation='relu'))
nn_deep.add(Dense(nn_deep_num_nodes_hidden_layer_3, activation='relu'))
nn_deep.add(Dense(nn_deep_num_nodes_hidden_layer_4, activation='relu'))
nn_deep.add(Dense(nn_deep_num_nodes_hidden_layer_5, activation='relu'))
nn_deep.add(Dense(1, activation='linear'))
# model_throughput.summary()
# mean_squared_error (mse) and mean_absolute_error (mae) are our loss functions
nn_deep.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
# nn.compile(loss=custom_loss, optimizer='adam', metrics=['mse', 'mae'])
nn_deep.save_weights('ffnn_default_weights.h5')

# Deep CNN
n_features = 30
n_outputs = 1
cnn_deep = Sequential()
# cnn_deep.add(Dropout(0.2, input_shape=(n_features, n_outputs)))
kernel_size = 4    # defines the size of the sliding window.
filters = 64    # how many different windows you will have. (All of them with the same length, which is kernel_size)
cnn_deep.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=(n_features, n_outputs)))
cnn_deep.add(MaxPooling1D(pool_size=4))
cnn_deep.add(Flatten())
cnn_deep.add(Dense(1000, activation='relu'))
cnn_deep.add(Dense(n_outputs, activation='linear'))
cnn_deep.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
cnn_deep.save_weights('cnn_default_weights.h5')

# --------- LEARN PARAMETERS (ML) & INFER OPTIMAL CONFIGURATION ---------
train_ratio = TRAIN_RATIO
test_ratio = 1 - train_ratio
print('Train/Test ratio: ', train_ratio, '/', test_ratio)
print('Train/Test no. samples: ', round(train_ratio * num_samples), '/', round(test_ratio * num_samples))
save_results = False

# --- Optimize throuhgput ---
print('-------------------------')
print('OPTIMIZATION 1: min throughput [Mbps]')
df_y = df_min_throughput
OPTIMAL_THROUGHPUT_MARGIN = 1  # if achieved throughput [Mbps] >= optimal throughput - OPTIMAL_THROUGHPUT_MARGIN, OKAY
# print('\nOutput (y):')
# print(df_y.head(NUM_HEAD_LINES))
# plot_data_pdf(df_y, "y: Min throughput [Mbps]", "pdf(y)")

mean_y = df_y.mean()
objective_y = df_y.max()
objective_y_ix = df_y.idxmax()
optimal_confs_ix = df_y.loc[df_y == objective_y]
num_optimal_confs = len(optimal_confs_ix)
pd.options.display.max_colwidth = 100

print('mean_y = %.2f Mbps' % mean_y)
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
                                       optimal_margin=OPTIMAL_THROUGHPUT_MARGIN, is_cnn=False)
        results_csv_writer.writerow(["LR-NE", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])

        # print('- Linear regression with GD:')
        # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
        #     get_average_model_accuracy(linreg_gd, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
        #                                optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
        # results_csv_writer.writerow(["LR-GD", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
        #                              "%.4f" % mae_test_av, "%.4f" % optimal_av])

        # print('- SVM SVR:')
        # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
        #     get_average_model_accuracy(svm_svr, df_x, df_y, NUM_RAND_TRIALS_SVM_SVR, train_ratio, maximize=True,
        #                                optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
        # results_csv_writer.writerow(["SVM-SVR", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
        #                              "%.4f" % mae_test_av, "%.4f" % optimal_av])

        # print('- SVM Linear:')
        # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av, optimal_av = \
        #     get_average_model_accuracy(svm_linsvr, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
        #                                optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
        # results_csv_writer.writerow(["SVM-LinSVR", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
        #                              "%.4f" % mae_test_av, "%.4f" % optimal_av])
        results_csv_writer.writerow(["NN", "%.4f" % rmse_train_av, "%.4f" % mae_train_av, "%.4f" % rmse_test_av,
                                     "%.4f" % mae_test_av, "%.4f" % optimal_av])
else:

    print('**********************************')
    print('Running models...')
    print('- Best in train:')
    rmse_train_av, mae_train_av, rmse_test_av, mae_test_av_rand, optimal_av, dif_optimal_av_rand = \
        get_average_best_in_train_accuracy(df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
                                           optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)

    # print('- Random selection:')
    # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av_rand, optimal_av, dif_optimal_av_rand = \
    #     get_average_random_accuracy(df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
    #                                 optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
    # #
    # print('- Linear regression with NE:')
    # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av_lin_ne, optimal_av, dif_optimal_av_lin_ne = \
    #     get_average_model_accuracy(linreg_ne, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
    #                                optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
    #
    # print('- SVM Linear:')
    # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av_svm_lin, optimal_av, dif_optimal_av_svm_lin = \
    #     get_average_model_accuracy(svm_linsvr, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
    #                                optimal_margin=OPTIMAL_THROUGHPUT_MARGIN)
    #
    # print('- Shallow Neural Net:')
    # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av_nn_sh, optimal_av, dif_optimal_av_nn_sh = \
    #     get_average_model_accuracy(nn_shallow, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
    #                                optimal_margin=OPTIMAL_THROUGHPUT_MARGIN, neural_net=1)
    #
    # print('- Quasi-deep Neural Net:')
    # rmse_train_av, mae_train_av, rmse_test_av, mae_test_av_nn_med, optimal_av_nn_med, dif_optimal_av_nn_med = \
    #     get_average_model_accuracy(nn_quasideep, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
    #                                optimal_margin=OPTIMAL_THROUGHPUT_MARGIN, neural_net=1)
    #

    print('- CNN Deep Neural Net:')
    rmse_train_av, mae_train_av, rmse_test_av, mae_test_av_dnn, optimal_av, dif_optimal_av_dnn = \
        get_average_model_accuracy(cnn_deep, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
                                   optimal_margin=OPTIMAL_THROUGHPUT_MARGIN, neural_net=2, is_cnn=True)

    print('- Deep Neural Net:')
    rmse_train_av, mae_train_av, rmse_test_av, mae_test_av_dnn, optimal_av, dif_optimal_av_dnn = \
        get_average_model_accuracy(nn_deep, df_x, df_y, NUM_RAND_TRIALS, train_ratio, maximize=True,
                                   optimal_margin=OPTIMAL_THROUGHPUT_MARGIN, neural_net=1, is_cnn=False)

    #
    # print("\n")
    # print('Difference to the optimal [Mbps]:')
    # table = BeautifulTable()
    # table.column_headers = ["Metric", "Rand", "LR", "SVM", "NN-sh", "NN-med", "DNN"]
    # table.append_row(["Diff. opt.","%.2f" % dif_optimal_av_rand, "%.2f" % dif_optimal_av_lin_ne, "%.2f" % dif_optimal_av_svm_lin,
    #                   "%.2f" % dif_optimal_av_nn_sh, "%.2f" % dif_optimal_av_nn_med, "%.2f" % dif_optimal_av_dnn])
    # table.append_row(
    #     ["MAE-test", "%.2f" % mae_test_av_rand, "%.2f" % mae_test_av_lin_ne, "%.2f" % mae_test_av_svm_lin,
    #      "%.2f" % mae_test_av_nn_sh, "%.2f" % mae_test_av_nn_med, "%.2f" % mae_test_av_dnn])
    # print(table)

print('\n\n')
print(".    --.--.---..---.  --.-- .-.    .   .--.-- .--..---.    .--.")
print("|      |  |    |        |  (   )   |\  |  |  :    |        |   :")
print("|      |  |--- |---     |   `-.    | \ |  |  |    |---    o|   |")
print("|      |  |    |        |  (   )   |  \|  |  :    |        |   ;")
print("'---'--'--'    '---'  --'-- `-'    '   '--'-- `--''---'   o'--'")
