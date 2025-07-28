import os
import logging
import time
import tensorflow as tf
import warnings
import numpy as np
from src.load_data import get_data
from src.recourse_methods import collective_counterfactual, get_ar_model, get_ml_model
from src.simulation import collectiong_data
from src.utils import print_in_box, seed_everything
import pandas as pd
logging.getLogger("torch").setLevel(logging.ERROR)
import argparse


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("torch").setLevel(logging.CRITICAL)
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
carla_logger = logging.getLogger('carla')
carla_logger.setLevel(logging.CRITICAL)
logging.getLogger("thefuzz").setLevel(logging.CRITICAL)


def run_single_benchmark(data_name, model_name, ce_method, seed):
    seed_everything(seed)
    
    start = time.time()
    
    # Load data
    data = get_data(data_name)
    
    # Load model
    model = get_ml_model(model_name, ce_method, data, data_name)
    
    # Get psotive and negative instances
    positive_data = data.df[model.predict(data.df) > 0.5]
    positive_data = positive_data[["feature1", "feature2"]]
    negative_data = data.df[model.predict(data.df) <= 0.5]
    negative_data = negative_data[["feature1", "feature2"]]
    factuals =  negative_data
    
    print_in_box(f'data: {data_name} model: {model_name} method: {ce_method} seed: {seed}')
    if ce_method != 'collective':
        ce_model = get_ar_model(ce_method, data, model, model_name)
        counterfactuals = ce_model.get_counterfactuals(factuals)
    else:
        counterfactuals, pi = collective_counterfactual(factuals, positive_data) 
        df = pd.DataFrame(pi)
        df.to_csv(f'./simulations/{data_name}/{data_name}_{model_name}_{ce_method}_{seed}_pi.csv', index=False)
    output_dir = f'./simulations/{data_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    positive_data.to_csv(f'./simulations/{data_name}/{data_name}_{model_name}_{ce_method}_{seed}_positive.csv', index=False)
    negative_data.to_csv(f'./simulations/{data_name}/{data_name}_{model_name}_{ce_method}_{seed}_negative.csv', index=False)
    factuals.to_csv(f'./simulations/{data_name}/{data_name}_{model_name}_{ce_method}_{seed}_factuals.csv', index=False)
    counterfactuals.to_csv(f'./simulations/{data_name}/{data_name}_{model_name}_{ce_method}_{seed}_counterfactuals.csv', index=False)
    
    end = time.time()
    elapsed_time = end - start
    print_in_box(f'Time taken for {data_name} {model_name} {ce_method}: {elapsed_time} seconds')


def run_benchmark(s):
    # Set Parameters:
    data_names =  ["adult"] # ["adult","compas", 'moons', "heloc"]# 
    model_names = ["ann"]
    # All methods: 'wacher', 'growing_spheres', 'roar','clue', 'actionable_recourse',
    #              'focus','cchvae', 'crud', 'revise', 'collective', 'guided_prototypes'
    # ["dice", "face","moc", "guided_prototypes"]
    ce_methods = ["moc"]# ["dice", "face","moc", "guided_prototypes"] #['wacher', 'growing_spheres', 'focus','clue', 'collective','roar', 'cchvae', 'dice', 'guided_prototypes', 'moc'] # 

    for i in range(len(data_names)):
        for j in range(len(model_names)):
            for k in range(len(ce_methods)):
                    data_name = data_names[i]
                    model_name = model_names[j]
                    ce_method = ce_methods[k]
                    seed_everything(s)
                    try:
                        run_single_benchmark(data_names[i], model_names[j], ce_methods[k], s)
                    except Exception as e:
                        print(f"Error running benchmark for {data_names[i]}, {model_names[j]}, {ce_methods[k]}, seed {s}: {e}")
                    
# df = pd.read_csv('data/again.csv')
# for s in range(T):
#     for i in range(len(df)):
#         data_name = df.loc[i, 'data_name']
#         model_name = "ann"
#         ce_method = df.loc[i, 'method']
#         seed = s
#         try:
#             run_single_benchmark(data_name, model_name, ce_method, seed)
#         except Exception as e:
#             print(f"Error running benchmark for {data_name}, {model_name}, {ce_method}, seed {seed}: {e}")
# collectiong_data(T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks for a given seed.")
    parser.add_argument("--seed", type=int, default=101, help="Random seed (default: 42).")
    args = parser.parse_args()
    run_benchmark(args.seed)