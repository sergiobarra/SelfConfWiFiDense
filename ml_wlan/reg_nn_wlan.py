import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print('******************************************************')
print('ML WLAN problem through ARTIFICIAL NEURAL NETWORK')
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

# Now, we train the neural network. We are using the 30 input variables (p_A_0,...,Ptx_F),
# along with two hidden layers of 12 and 8 neurons respectively, and finally using the linear activation function to
# process the output.
model_throughput = Sequential()
model_throughput.add(Dense(12, input_dim=30, kernel_initializer='normal', activation='relu'))
model_throughput.add(Dense(64, activation='relu'))
model_throughput.add(Dense(1, activation='linear'))
model_throughput.summary()

# mean_squared_error (mse) and mean_absolute_error (mae) are our loss functions
model_throughput.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

history_throughput = model_throughput.fit(X_train, y_train, epochs=150, batch_size=50, verbose=0, validation_split=0.2)

# print(history.history.keys())
# "Loss"
plt.plot(history_throughput.history['loss'])
plt.plot(history_throughput.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Pick optimal conf
actual_best_y = df_y.max()
print('Actual best min throughput: ', actual_best_y, 'Mbps')
print('Picking the optimal through NN...')
exhaustive_prediction = model_throughput.predict(df_x)
pred_best_conf_ix = np.argmax(exhaustive_prediction)
pred_best_y = exhaustive_prediction[pred_best_conf_ix]
actual_pred_best_y = df_y.loc[pred_best_conf_ix]
print('- best_conf ix:', pred_best_conf_ix)
print('- Predicted/actual min throughput: ', pred_best_y, '/', actual_pred_best_y, 'Mbps')

# ---------- Optimize delay ----------
print('-------------------------')
print('OPTIMIZATION 2: max delay [Mbps]')

df_y = df_max_delay
# plt.figure()
# ax = df_y.plot.kde()
# plt.xlabel("y: Min throughput [Mbps]")
# plt.ylabel("pdf(y)")
# plt.show()
# plt.close()

print('\nOutput (y):')
print(df_y.head(NUM_HEAD_LINES))

X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(df_x, df_y, test_size=test_ratio,
                                     random_state=1)
print('Train/Test no. samples: ', len(X_train), '/', len(X_test))

# Now, we train the neural network. We are using the 30 input variables (p_A_0,...,Ptx_F),
# along with two hidden layers of 12 and 8 neurons respectively, and finally using the linear activation function to
# process the output.
model_delay = Sequential()
model_delay.add(Dense(12, input_dim=30, kernel_initializer='normal', activation='relu'))
# model.add(Dense(8, activation='relu'))
model_delay.add(Dense(1, activation='linear'))
model_delay.summary()

# mean_squared_error (mse) and mean_absolute_error (mae) are our loss functions
model_delay.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

history_delay = model_delay.fit(X_train, y_train, epochs=150, batch_size=50, verbose=0, validation_split=0.2)

# print(history.history.keys())
# "Loss"
plt.plot(history_delay.history['loss'])
plt.plot(history_delay.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Pick optimal conf
actual_best_y = df_y.min()
print('Actual best max delay: ', actual_best_y, 'ms')
print('Picking the optimal through NN...')
exhaustive_prediction = model_delay.predict(df_x)
pred_best_conf_ix = np.argmin(exhaustive_prediction)
pred_best_y = exhaustive_prediction[pred_best_conf_ix]
actual_pred_best_y = df_y.loc[pred_best_conf_ix]
print('- best_conf ix:', pred_best_conf_ix)
print('- Predicted/actual max delay: ', pred_best_y, '/', actual_pred_best_y, 'ms')


# ---------------------------------
print('\n\n******* LIFE IS NICE. DO NOT FORGET :D **************')
print(".    --.--.---..---.  --.-- .-.    .   .--.-- .--..---.    .--.")
print("|      |  |    |        |  (   )   |\  |  |  :    |        |   :")
print("|      |  |--- |---     |   `-.    | \ |  |  |    |---    o|   |")
print("|      |  |    |        |  (   )   |  \|  |  :    |        |   ;")
print("'---'--'--'    '---'  --'-- `-'    '   '--'-- `--''---'   o'--'")