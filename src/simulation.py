import pandas as pd
import numpy as np
from itertools import product
from src.utils import chi_squared_divergence
from src.utils import euclidean

# --------------------------
# Setup and parameter grids
# --------------------------

def collectiong_data(T):

    methods = ['wacher', 'growing_spheres',  'focus', 'clue', 'collective','roar', 'cchvae','face', 'dice', 'guided_prototypes', 'moc']
    data_names = ["adult", "compas", 'moons', "heloc"]
    models = ["ann"]
    seeds = range(T)
    # Create parameter grid similar to expand.grid
    params = pd.DataFrame(list(product(data_names, models, methods, seeds)),
                    columns=['data_name', 'model', 'method', 'seed'])
    
    # Add columns like mutate(...)
    params['stable'] = -1
    params['cost'] = -1
    params['factual'] = -1
    params['counterfactual'] = -1
 


    for i in range(params.shape[0]):
        import os
        data_name = params.loc[i, 'data_name']
        model = params.loc[i, 'model']
        method = params.loc[i, 'method']
        seed = params.loc[i, 'seed']
        # if not exist the below path continue for
        if not os.path.exists(f"simulations/{data_name}/{data_name}_{model}_{method}_{seed}_counterfactuals.csv"):
            continue
        pos = pd.read_csv(f"simulations/{data_name}/{data_name}_{model}_{method}_{seed}_positive.csv")
        neg = pd.read_csv(f"simulations/{data_name}/{data_name}_{model}_{method}_{seed}_negative.csv")
        fac = pd.read_csv(f"simulations/{data_name}/{data_name}_{model}_{method}_{seed}_factuals.csv")
        cft = pd.read_csv(f"simulations/{data_name}/{data_name}_{model}_{method}_{seed}_counterfactuals.csv")

        pos = pos.iloc[:, 0:2]; neg = neg.iloc[:, 0:2] ; fac = fac.iloc[:, 0:2]; cft = cft.iloc[:, 0:2]
        # Remove rows with NaN in cft
        mask = cft.notnull().all(axis=1)
        fac = fac[mask]
        cft = cft[mask]
        if cft.shape[0] == 0:
            continue
        
        params.loc[i, 'cost'] = np.mean(euclidean(fac, cft))
        params.loc[i, 'factual'] = fac.shape[0]
        params.loc[i, 'counterfactual'] = cft.shape[0]

        # if pos.shape[0] < cft.shape[0]:
        #     pos = pos.sample(n = cft.shape[0], replace=True)
        # elif pos.shape[0] > cft.shape[0]:
        #     cft = cft.sample(n = pos.shape[0], replace=True)


        if pos.shape[0] < cft.shape[0]:
            cft = cft.sample(n = pos.shape[0])
        elif pos.shape[0] > cft.shape[0]:
            pos = pos.sample(n = cft.shape[0])

    
        # row bind two dataframes
        moving = pd.concat([pos,  cft], axis=0)
        # substitue in in params dataframe
        params.loc[i, 'stable'] = chi_squared_divergence(pos, moving)

    params.to_csv("simulations/simulation.csv", index=False)