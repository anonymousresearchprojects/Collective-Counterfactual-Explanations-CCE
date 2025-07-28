from carla.data.catalog import OnlineCatalog
from sklearn.datasets import make_moons
import csv
import os
import pandas as pd
from carla.data.catalog import CsvCatalog
import numpy as np

def geneerate_moons_data():
    x, y = make_moons(2000, noise=0.15)
    output_dir = './data/moons'
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    with open('./data/moons/moons.csv', 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature1', 'feature2', 'target'])
        
        for i in range(len(x)):
            writer.writerow([*x[i], y[i]])
            
        f.close()

def get_data(data_name):
    """
    Retrieve a dataset by its name.
    Parameters:
    data_name (str): The name of the dataset to retrieve. Supported data names are:
                     'adult', 'compas', 'give_me_some_credit', 'heloc'.
    Returns:
    dataset: The dataset corresponding to the provided data_name.
    Raises:
    ValueError: If the provided data_name is not supported.
    """

    if data_name not in ["adult", "compas", "give_me_some_credit", "heloc", "moons"]:
        raise ValueError(f"Unsupported data_name: {data_name}. Supported data names are: 'adult', 'compas', 'give_me_some_credit', 'heloc'.")
    
    if data_name == 'moons':
            geneerate_moons_data()
    if not os.path.exists(f'data/{data_name}/{data_name}.csv'):
        dataset = OnlineCatalog(data_name)
        actionable_dict = {
            'adult': ['education-num', 'hours-per-week'],
            'compas': ['priors_count', 'length_of_stay'],
            'give_me_some_credit': ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio'],
            'heloc': ['PercentTradesNeverDelq', 'NumTradesOpeninLast12M']
        }        
        target = dataset.target
        df = dataset.df
        all_names = actionable_dict.get(data_name, []) + [target]
        dframe = df[all_names]
        dframe.columns = ['feature1', 'feature2', 'target']

        dim0 = dframe.loc[dframe["target"] == 0].shape[0]
        dim1 = dframe.loc[dframe["target"] == 1].shape[0]
        df0 = dframe.loc[dframe["target"] == 0].sample(n = min(dim0, 1000))
        df1 = dframe.loc[dframe["target"] == 1].sample(n = min(dim1, 1000))
        df_n = pd.concat([df0, df1], ignore_index=True)
        df_n[target] = np.where(df_n[target] > 0.5, 0, 1)
        df = df_n.sort_values(target, ignore_index=True)


        output_dir = f'./data/{data_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f'data/{data_name}/{data_name}.csv', 'w+', newline='') as f:
            f.write(dframe.to_csv(index=False))    
            f.close()


    dataset_csv = CsvCatalog(f'data/{data_name}/{data_name}.csv',
        categorical=[],
        continuous=['feature1', 'feature2'],
        immutables=[],
        target='target'
    )
    
    return dataset_csv