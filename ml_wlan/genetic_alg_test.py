import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns


def initialize():
    # Whole raw data set
    dataset_path = 'problem1_dataset_demo.csv'
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

    df_y = df_min_throughput

    df_dataset = pd.concat([df_x, df_y], sort=False, ignore_index=True)

    return df_dataset


def fitness(member):
    return True


# Bread solutions: pick a piece of each solution and mix them
def crossover(a, b):
    return a


# To introduce random noise
def mutate(member):
    return member


# create random member of population
def create_new_member(df_dataset):

    df_member = df_dataset.sample()
    return df_member


# create new population
def create_next_generation(population):
    return population


def main(number_of_iterations):

    return True
