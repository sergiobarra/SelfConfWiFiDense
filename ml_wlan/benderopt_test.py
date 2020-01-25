from benderopt import minimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns

LOWEST_CH = 0
HIGHEST_CH = 3
df_dataset = pd.DataFrame()
df_x = pd.DataFrame()
df_y = pd.DataFrame()
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 30

# https://github.com/Dreem-Organization/benderopt

def initialize():
    # Whole raw data set
    dataset_path = 'problem3_dataset.csv'
    print('Reading dataset', dataset_path, '...')
    # df = pd.read_csv('problem1_dataset_demo.csv', sep=';')
    df = pd.read_csv(dataset_path, sep=';')
    print(' - Dataset read!')

    # print(df.inf.info())  # Overview of data set
    num_samples = len(df.index)
    print(' - Number of samples in the data set: ', str(num_samples))

    # ---------- Tune input data ----------
    # Raw input features of interest
    df = df[['p_A', 'p_B', 'p_C', 'p_D', 'p_E', 'p_F',
             'Ptx_A', 'Ptx_B', 'Ptx_C', 'Ptx_D', 'Ptx_E', 'Ptx_F', 's_min']]

    # Add every possible value that each feature can take (primary = [1,4])
    df.loc[(df.Ptx_A == 14), 'Ptx_A'] = 0
    df.loc[(df.Ptx_B == 14), 'Ptx_B'] = 0
    df.loc[(df.Ptx_C == 14), 'Ptx_C'] = 0
    df.loc[(df.Ptx_D == 14), 'Ptx_D'] = 0
    df.loc[(df.Ptx_E == 14), 'Ptx_E'] = 0
    df.loc[(df.Ptx_F == 14), 'Ptx_F'] = 0
    df.loc[(df.Ptx_A == 20), 'Ptx_A'] = 1
    df.loc[(df.Ptx_B == 20), 'Ptx_B'] = 1
    df.loc[(df.Ptx_C == 20), 'Ptx_C'] = 1
    df.loc[(df.Ptx_D == 20), 'Ptx_D'] = 1
    df.loc[(df.Ptx_E == 20), 'Ptx_E'] = 1
    df.loc[(df.Ptx_F == 20), 'Ptx_F'] = 1

    action_ix_col = range(num_samples)

    df["action_ix"] = action_ix_col

    return df


optimization_problem = [
    {
        "name": "p_A",
        "category": "uniform",
        "search_space": {
            "low": LOWEST_CH,
            "high": HIGHEST_CH + 0.1,
            "step": 1
        }
    },
    {
        "name": "p_B",
        "category": "uniform",
        "search_space": {
            "low": LOWEST_CH,
            "high": HIGHEST_CH + 0.1,
            "step": 1
        }
    },
    {
        "name": "p_C",
        "category": "uniform",
        "search_space": {
            "low": LOWEST_CH,
            "high": HIGHEST_CH + 0.1,
            "step": 1
        }
    },
    {
        "name": "p_D",
        "category": "uniform",
        "search_space": {
            "low": LOWEST_CH,
            "high": HIGHEST_CH + 0.1,
            "step": 1
        }
    },
    {
        "name": "p_E",
        "category": "uniform",
        "search_space": {
            "low": LOWEST_CH,
            "high": HIGHEST_CH + 0.1,
            "step": 1
        }
    },
    {
        "name": "p_F",
        "category": "uniform",
        "search_space": {
            "low": LOWEST_CH,
            "high": HIGHEST_CH + 0.1,
            "step": 1
        }
    },
    {
        "name": "Ptx_A",
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 1 + 0.1,
            "step": 1
        }
    },
    {
        "name": "Ptx_B",
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 1 + 0.1,
            "step": 1
        }
    },
    {
        "name": "Ptx_C",
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 1 + 0.1,
            "step": 1
        }
    },
    {
        "name": "Ptx_D",
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 1 + 0.1,
            "step": 1
        }
    },
    {
        "name": "Ptx_E",
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 1 + 0.1,
            "step": 1
        }
    },
    {
        "name": "Ptx_F",
        "category": "uniform",
        "search_space": {
            "low": 0,
            "high": 1 + 0.1,
            "step": 1
        }
    },
]


def function_to_optimize(p_A, p_B, p_C, p_D, p_E, p_F, Ptx_A, Ptx_B, Ptx_C, Ptx_D, Ptx_E, Ptx_F):
    action_ix = df_dataset[(df_dataset['p_A'] == p_A) &
                           (df_dataset['p_B'] == p_B) &
                           (df_dataset['p_C'] == p_C) &
                           (df_dataset['p_D'] == p_D) &
                           (df_dataset['p_E'] == p_E) &
                           (df_dataset['p_F'] == p_F) &
                           (df_dataset['Ptx_A'] == Ptx_A) &
                           (df_dataset['Ptx_B'] == Ptx_B) &
                           (df_dataset['Ptx_C'] == Ptx_C) &
                           (df_dataset['Ptx_D'] == Ptx_D) &
                           (df_dataset['Ptx_E'] == Ptx_E) &
                           (df_dataset['Ptx_F'] == Ptx_F)]

    return 0


df_dataset = initialize()

print(df_dataset.head(3))

df_target = df_dataset[(df_dataset['p_A'] == 0) &
                       (df_dataset['p_B'] == 0) &
                       (df_dataset['p_C'] == 0) &
                       (df_dataset['p_D'] == 1) &
                       (df_dataset['p_E'] == 0) &
                       (df_dataset['p_F'] == 0) &
                       (df_dataset['Ptx_A'] == 0) &
                       (df_dataset['Ptx_B'] == 0) &
                       (df_dataset['Ptx_C'] == 0) &
                       (df_dataset['Ptx_D'] == 0) &
                       (df_dataset['Ptx_E'] == 0) &
                       (df_dataset['Ptx_F'] == 0)]

print('df_target:\n', df_target)

#
# # We launch the optimization
# best_sample = minimize(function_to_optimize,
#                        optimization_problem,
#                        optimizer_type="parzen_estimator",
#                        number_of_evaluation=40,
#                        seed=None,
#                        debug=None)

# print(best_sample)
