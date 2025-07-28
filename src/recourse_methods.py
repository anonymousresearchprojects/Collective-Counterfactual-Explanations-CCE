from carla import  MLModelCatalog
from carla.recourse_methods import *
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from src.uot import unbalanced_ot_kl_chi2, unbalanced_ot_solver
from src.utils import barycentric_projection, get_backend, pairwise_distances, sample_from_transport_plan, sample_from_transport_plan2, top_k_normalize
# from deap import base, creator, tools, algorithms  # Commented out to avoid conflicts


class GuidedPrototypes:
    """
    Guided Prototypes implementation for counterfactual explanations.
    Based on the paper: "Guided Prototypes for Counterfactual Explanations" by Van Looveren and Klaise.
    """
    
    def __init__(self, ml_model, hyperparams=None):
        self.ml_model = ml_model
        self.hyperparams = hyperparams or {}
        
        # Default hyperparameters
        self.k = self.hyperparams.get('k', 5)  # Number of prototypes
        self.beta = self.hyperparams.get('beta', 0.1)  # Weight for prototype loss
        self.gamma = self.hyperparams.get('gamma', 0.1)  # Weight for diversity loss
        self.max_iter = self.hyperparams.get('max_iter', 1000)
        self.lr = self.hyperparams.get('lr', 0.01)
        self.tol = self.hyperparams.get('tol', 1e-6)
        
        # Initialize prototypes
        self.prototypes = None
        self.scaler = StandardScaler()
        
    def _find_prototypes(self, positive_data):
        """Find k prototypes from positive class data."""
        # Ensure we're working with the correct features
        positive_features = positive_data[self.ml_model.feature_input_order]
        
        if len(positive_features) <= self.k:
            return positive_features.values
        
        # Use k-means or select diverse points
        # For simplicity, we'll use the first k points or sample k diverse points
        if len(positive_features) > self.k:
            # Use k-means to find diverse prototypes
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
            kmeans.fit(positive_features)
            prototypes = kmeans.cluster_centers_
        else:
            prototypes = positive_features.values
            
        return prototypes
    
    def _guided_prototype_loss(self, factual, prototype):
        """Compute guided prototype loss."""
        # Distance to prototype
        dist_to_prototype = np.linalg.norm(factual - prototype, axis=1)
        
        # Model prediction for factual
        factual_pred = self.ml_model.predict(pd.DataFrame(factual, columns=self.ml_model.feature_input_order))
        
        # Model prediction for prototype
        prototype_pred = self.ml_model.predict(pd.DataFrame(prototype, columns=self.ml_model.feature_input_order))
        
        # Guided loss: encourage factual to move towards prototype
        guided_loss = dist_to_prototype + self.beta * (1 - prototype_pred)
        
        return guided_loss
    
    def _compute_gradient(self, cf, prototype):
        """Compute gradient for optimization."""
        # Gradient: move towards prototype (negative because we want to minimize distance)
        gradient = 2 * (cf - prototype)
        return gradient
    
    def _diversity_loss(self, prototypes):
        """Compute diversity loss to ensure prototypes are different."""
        if len(prototypes) < 2:
            return 0
        
        diversity_loss = 0
        for i in range(len(prototypes)):
            for j in range(i+1, len(prototypes)):
                dist = np.linalg.norm(prototypes[i] - prototypes[j])
                diversity_loss += 1.0 / (dist + 1e-8)  # Penalize close prototypes
                
        return diversity_loss
    
    def _find_nearest_prototype(self, factual, positive_data):
        """Find the nearest positive prototype to the factual instance."""
        if len(positive_data) == 0:
            return factual
        
        # Ensure we're working with the same features as the factual
        factual_features = factual[self.ml_model.feature_input_order].values if isinstance(factual, pd.Series) else factual
        positive_features = positive_data[self.ml_model.feature_input_order]
        
        # Use k-means to find prototypes if we have enough data
        if len(positive_features) > self.k:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
            kmeans.fit(positive_features)
            prototypes = kmeans.cluster_centers_
        else:
            prototypes = positive_features.values
        
        # Find nearest prototype
        distances = [np.linalg.norm(factual_features - prototype) for prototype in prototypes]
        nearest_idx = np.argmin(distances)
        return prototypes[nearest_idx]
    
    def _move_towards_prototype(self, factual, prototype, step_size=0.5):
        """Move factual instance towards prototype by a fixed step size."""
        direction = prototype - factual
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # If already at prototype, add small random perturbation
            return factual + np.random.normal(0, 0.1, factual.shape)
        
        # Normalize direction and move by step_size
        normalized_direction = direction / distance
        counterfactual = factual + step_size * normalized_direction
        
        return counterfactual
    
    def _optimize_counterfactual(self, factual, positive_data):
        """Optimize counterfactual using guided prototypes."""
        # Find prototypes
        self.prototypes = self._find_prototypes(positive_data)
        
        # Ensure we're working with the same features as the factual
        factual_features = factual[self.ml_model.feature_input_order].values if isinstance(factual, pd.Series) else factual
        prototypes_features = self.prototypes[self.ml_model.feature_input_order].values if isinstance(self.prototypes, pd.DataFrame) else self.prototypes
        
        # Scale data - fit scaler on all data to ensure consistent scaling
        all_data_for_scaling = np.vstack([factual_features.reshape(1, -1), prototypes_features])
        self.scaler.fit(all_data_for_scaling)
        
        factual_scaled = self.scaler.transform(factual_features.reshape(1, -1))
        prototypes_scaled = self.scaler.transform(prototypes_features)
        
        # Initialize counterfactual as factual
        cf = factual_scaled.copy()
        
        # Get original prediction
        factual_df = pd.DataFrame(factual_features.reshape(1, -1), columns=self.ml_model.feature_input_order)
        original_pred = self.ml_model.predict(factual_df)[0]
        
        # Gradient descent optimization
        for iteration in range(self.max_iter):
            cf_old = cf.copy()
            
            # Transform back to original scale for prediction
            cf_original = self.scaler.inverse_transform(cf)
            cf_df = pd.DataFrame(cf_original, columns=self.ml_model.feature_input_order)
            current_pred = self.ml_model.predict(cf_df)[0]
            
            # Check if we've crossed the decision boundary
            if current_pred > 0.5 and original_pred <= 0.5:
                break
            
            # Compute gradients for each prototype
            total_grad = np.zeros_like(cf)
            
            for prototype in prototypes_scaled:
                # Compute gradient towards prototype
                grad = self._compute_gradient(cf, prototype.reshape(1, -1))
                total_grad += grad
            
            # Average the gradients
            if len(prototypes_scaled) > 0:
                total_grad /= len(prototypes_scaled)
            
            # Add prediction-based gradient to push towards positive class
            if current_pred <= 0.5:
                # Add small random perturbation to explore
                exploration_grad = np.random.normal(0, 0.01, cf.shape)
                total_grad += exploration_grad
            
            # Update counterfactual
            cf = cf - self.lr * total_grad
            
            # Check convergence
            if np.linalg.norm(cf - cf_old) < self.tol:
                break
        
        # Transform back to original scale
        cf_original = self.scaler.inverse_transform(cf)
        
        return cf_original.flatten()
    
    def get_counterfactuals(self, factuals):
        """Generate counterfactuals for given factual instances."""
        # Get positive class data for prototypes
        all_data = self.ml_model.data.df
        positive_mask = self.ml_model.predict(all_data) > 0.5
        positive_data = all_data[positive_mask]
        
        if len(positive_data) == 0:
            # If no positive data, return factuals
            return factuals
        
        counterfactuals = []
        
        for _, factual in factuals.iterrows():
            try:
                # Ensure we're working with the correct features
                factual_features = factual[self.ml_model.feature_input_order]
                cf = self._optimize_counterfactual(factual_features, positive_data)
                
                # Verify that the counterfactual is actually different from factual
                if np.linalg.norm(cf - factual_features.values) < 1e-6:
                    # If counterfactual is too similar to factual, try a different approach
                    # Move towards the nearest positive prototype
                    nearest_prototype = self._find_nearest_prototype(factual_features, positive_data)
                    cf = self._move_towards_prototype(factual_features.values, nearest_prototype)
                
                counterfactuals.append(cf)
            except Exception as e:
                # If optimization fails, try simple prototype-based approach
                print(f"Warning: Guided Prototypes optimization failed for factual {factual.name}: {e}")
                try:
                    factual_features = factual[self.ml_model.feature_input_order]
                    nearest_prototype = self._find_nearest_prototype(factual_features, positive_data)
                    cf = self._move_towards_prototype(factual_features.values, nearest_prototype)
                    counterfactuals.append(cf)
                except:
                    # Last resort: return factual
                    factual_features = factual[self.ml_model.feature_input_order]
                    counterfactuals.append(factual_features.values)
        
        return pd.DataFrame(counterfactuals, columns=self.ml_model.feature_input_order, index=factuals.index)


# class MOC:
#     """
#     Model-agnostic Counterfactuals (MOC) implementation for tabular data.
#     Reference: Artelt & Hammer (2019)
#     Simplified version without DEAP dependency.
#     Ensures that the generated counterfactual flips the classifier's label.
#     """
#     def __init__(self, ml_model, hyperparams=None):
#         self.ml_model = ml_model
#         self.hyperparams = hyperparams or {}
#         self.max_iter = self.hyperparams.get('max_iter', 100)
#         self.lr = self.hyperparams.get('lr', 0.01)
#         self.tol = self.hyperparams.get('tol', 0.5)  # how close to target
#         self.feature_names = ml_model.feature_input_order

#     def get_counterfactuals(self, factuals):
#         # For each factual, generate a counterfactual
#         counterfactuals = []
#         for _, factual in factuals.iterrows():
#             cf = self._find_counterfactual(factual)
#             counterfactuals.append(cf)
#         return pd.DataFrame(counterfactuals, columns=factuals.columns, index=factuals.index)

#     def _find_counterfactual(self, factual):
#         """Find counterfactual using gradient-based optimization, ensuring label flip."""
#         factual_np = factual.values.astype(float)
#         model = self.ml_model
#         # Get original prediction and class
#         factual_df = pd.DataFrame(factual_np.reshape(1, -1), columns=self.feature_names)
#         orig_pred = model.predict(factual_df)[0]
#         orig_class = int(orig_pred > 0.5)
#         target_class = 1 - orig_class  # flip class

#         # Initialize counterfactual as factual
#         cf = factual_np.copy()
#         found_flip = False
#         best_cf = cf.copy()
#         best_pred_loss = float('inf')

#         # Gradient descent optimization
#         for iteration in range(self.max_iter):
#             cf_old = cf.copy()
#             cf_df = pd.DataFrame(cf.reshape(1, -1), columns=self.feature_names)
#             pred = model.predict(cf_df)[0]
#             pred_class = int(pred > 0.5)
#             pred_loss = abs(pred - target_class)

#             # If label flips, return immediately
#             if pred_class == target_class:
#                 found_flip = True
#                 best_cf = cf.copy()
#                 break

#             # Track closest to target class
#             if pred_loss < best_pred_loss:
#                 best_pred_loss = pred_loss
#                 best_cf = cf.copy()

#             # Simple gradient: move towards target class
#             if pred < target_class:
#                 cf = cf + self.lr * np.random.normal(0, 0.1, cf.shape)
#             else:
#                 cf = cf - self.lr * np.random.normal(0, 0.1, cf.shape)

#             # Keep counterfactual close to factual
#             dist_to_factual = np.linalg.norm(cf - factual_np)
#             if dist_to_factual > 2.0:  # Limit distance
#                 cf = factual_np + 2.0 * (cf - factual_np) / dist_to_factual

#             # Check convergence
#             if np.linalg.norm(cf - cf_old) < 1e-6:
#                 break

#         # If no flip found, return the closest attempt (or factual as fallback)
#         # Check if best_cf flips the class
#         best_cf_df = pd.DataFrame(best_cf.reshape(1, -1), columns=self.feature_names)
#         best_pred_class = int(model.predict(best_cf_df)[0] > 0.5)
#         if best_pred_class == target_class:
#             return best_cf
#         else:
#             # Fallback: return factual
#             return factual_np

# --------------------------------------------
import numpy as np
import pandas as pd

class MOC:
    """
    Multi-Objective Counterfactual Explanations (MOC) implementation.
    Reference: Dandl, Susanne, et al. "Multi-objective counterfactual explanations." 
    International Conference on Parallel Problem Solving from Nature. Springer, Cham, 2020.
    """
    def __init__(self, ml_model, hyperparams=None):
        self.ml_model = ml_model
        self.hyperparams = hyperparams or {}
        self.num_generations = self.hyperparams.get("num_generations", 50)
        self.population_size = self.hyperparams.get("population_size", 50)
        self.num_counterfactuals = self.hyperparams.get("num_counterfactuals", 1)
        self.mutation_rate = self.hyperparams.get("mutation_rate", 0.1)
        self.crossover_rate = self.hyperparams.get("crossover_rate", 0.5)
        self.target_class = self.hyperparams.get("target_class", 1)
        self.feature_names = self.ml_model.feature_input_order
        self.feature_bounds = self._get_feature_bounds()
        self.immutables = getattr(self.ml_model.data, "immutables", [])
        self.immutable_idx = [self.feature_names.index(f) for f in self.immutables if f in self.feature_names]

    def _get_feature_bounds(self):
        # Get min/max for each feature from the training data
        df = self.ml_model.data.df
        bounds = []
        for f in self.feature_names:
            col = df[f]
            bounds.append((col.min(), col.max()))
        return bounds

    def get_counterfactuals(self, factuals):
        # factuals: pd.DataFrame
        cf_list = []
        for idx, factual in factuals.iterrows():
            cf = self._generate_counterfactual(factual)
            cf_list.append(cf)
        return pd.DataFrame(cf_list, columns=self.feature_names, index=factuals.index)

    def _generate_counterfactual(self, factual):
        factual_np = factual[self.feature_names].values.astype(float)
        # Initialize population
        population = self._initialize_population(factual_np)
        for gen in range(self.num_generations):
            # Evaluate objectives
            objectives = self._evaluate_population(population, factual_np)
            # Non-dominated sorting
            fronts = self._non_dominated_sort(objectives)
            # Selection (elitism)
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) > self.population_size:
                    # Fill up to population_size
                    needed = self.population_size - len(new_population)
                    new_population.extend([population[i] for i in front[:needed]])
                    break
                else:
                    new_population.extend([population[i] for i in front])
            # Variation (crossover + mutation)
            while len(new_population) < self.population_size:
                parent1, parent2 = self._tournament_selection(population, objectives), self._tournament_selection(population, objectives)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, factual_np)
                new_population.append(child)
            population = np.array(new_population)
        # Final evaluation and selection of best counterfactual
        final_objectives = self._evaluate_population(population, factual_np)
        # Filter those that flip the class
        valid_idx = [i for i, obj in enumerate(final_objectives) if obj[1] == 0]
        if valid_idx:
            # Choose the one with minimal distance
            best_idx = min(valid_idx, key=lambda i: final_objectives[i][0])
            return population[best_idx]
        else:
            # No valid counterfactual found, return factual
            return factual_np

    def _initialize_population(self, factual_np):
        pop = []
        for _ in range(self.population_size):
            individual = factual_np.copy()
            for i, (low, high) in enumerate(self.feature_bounds):
                if i in self.immutable_idx:
                    continue
                # Random perturbation within bounds
                individual[i] = np.random.uniform(low, high)
            pop.append(individual)
        return np.array(pop)

    def _evaluate_population(self, population, factual_np):
        # Returns list of tuples: (distance, class_loss)
        objectives = []
        for ind in population:
            # Distance objective (L1)
            dist = np.sum(np.abs(ind - factual_np))
            # Class loss: 0 if flips to target, else abs(pred - target_class)
            pred = self.ml_model.predict(pd.DataFrame([ind], columns=self.feature_names))[0]
            pred_class = int(pred > 0.5)
            class_loss = 0 if pred_class == self.target_class else abs(pred - self.target_class) + 1
            objectives.append((dist, class_loss))
        return objectives

    def _non_dominated_sort(self, objectives):
        # Fast non-dominated sort (returns list of fronts, each front is list of indices)
        pop_size = len(objectives)
        S = [[] for _ in range(pop_size)]
        n = [0 for _ in range(pop_size)]
        rank = [0 for _ in range(pop_size)]
        fronts = [[]]
        for p in range(pop_size):
            for q in range(pop_size):
                if self._dominates(objectives[p], objectives[q]):
                    S[p].append(q)
                elif self._dominates(objectives[q], objectives[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return [front for front in fronts if front]

    def _dominates(self, obj1, obj2):
        # Minimization: obj1 dominates obj2 if all objectives <= and at least one <
        return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))

    def _tournament_selection(self, population, objectives, k=2):
        idxs = np.random.choice(len(population), k, replace=False)
        best = idxs[0]
        for i in idxs[1:]:
            if self._dominates(objectives[i], objectives[best]):
                best = i
        return population[best].copy()

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand()
            child = alpha * parent1 + (1 - alpha) * parent2
            # Keep immutable features unchanged
            for i in self.immutable_idx:
                child[i] = parent1[i]
            return child
        else:
            return parent1.copy()

    def _mutate(self, individual, factual_np):
        for i, (low, high) in enumerate(self.feature_bounds):
            if i in self.immutable_idx:
                continue
            if np.random.rand() < self.mutation_rate:
                # Small Gaussian mutation
                individual[i] += np.random.normal(0, 0.1 * (high - low))
                individual[i] = np.clip(individual[i], low, high)
        # Optionally, keep immutable features exactly as factual
        for i in self.immutable_idx:
            individual[i] = factual_np[i]
        return individual


# --------------------------------------------

class FaceWrapper:
    """
    Custom FACE (Feasible and Actionable Counterfactual Explanations) implementation.
    This is a simplified version that doesn't rely on CARLA's Face method to avoid
    the immutable_constraint_matrix1 error.
    """
    def __init__(self, ml_model, hyperparams=None):
        self.ml_model = ml_model
        self.hyperparams = hyperparams or {}
        self.mode = self.hyperparams.get('mode', 'knn')
        self.fraction = self.hyperparams.get('fraction', 0.5)
        self.k = self.hyperparams.get('k', 5)
        
    def get_counterfactuals(self, factuals):
        """Generate counterfactuals using FACE approach."""
        # Get positive class data for finding feasible regions
        all_data = self.ml_model.data.df
        positive_mask = self.ml_model.predict(all_data) > 0.5
        positive_data = all_data[positive_mask]
        
        if len(positive_data) == 0:
            # If no positive data, return factuals
            return factuals
        
        counterfactuals = []
        
        for _, factual in factuals.iterrows():
            try:
                cf = self._find_feasible_counterfactual(factual, positive_data)
                counterfactuals.append(cf)
            except Exception as e:
                # If optimization fails, return factual
                print(f"Warning: FACE optimization failed for factual {factual.name}: {e}")
                counterfactuals.append(factual.values)
        
        return pd.DataFrame(counterfactuals, columns=factuals.columns, index=factuals.index)
    
    def _find_feasible_counterfactual(self, factual, positive_data):
        """Find feasible counterfactual using FACE approach."""
        factual_np = factual.values.astype(float)
        
        if self.mode == 'knn':
            # Use k-nearest neighbors approach
            cf = self._knn_approach(factual_np, positive_data)
        else:
            # Use epsilon-ball approach
            cf = self._epsilon_approach(factual_np, positive_data)
        
        return cf
    
    def _knn_approach(self, factual, positive_data):
        """Find counterfactual using k-nearest neighbors approach."""
        from sklearn.neighbors import NearestNeighbors
        
        # Get positive data features only
        pos_features = positive_data[["feature1", "feature2"]].values
        
        # Find k nearest neighbors
        n_neighbors = min(self.k, len(pos_features))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(pos_features)
        distances, indices = nbrs.kneighbors(factual.reshape(1, -1))
        
        # Get the nearest positive instance
        nearest_positive = pos_features[indices[0][0]]
        
        # Interpolate between factual and nearest positive
        # Use fraction parameter to control how close to move towards positive
        cf = factual + self.fraction * (nearest_positive - factual)
        
        return cf
    
    def _epsilon_approach(self, factual, positive_data):
        """Find counterfactual using epsilon-ball approach."""
        # Get positive data features only
        pos_features = positive_data[["feature1", "feature2"]].values
        
        # Calculate distances to all positive instances
        distances = np.linalg.norm(pos_features - factual, axis=1)
        
        # Find instances within epsilon distance
        epsilon = np.percentile(distances, int(self.fraction * 100))
        within_epsilon = distances <= epsilon
        
        if np.any(within_epsilon):
            # Use the closest instance within epsilon
            closest_idx = np.argmin(distances[within_epsilon])
            closest_positive = pos_features[within_epsilon][closest_idx]
            
            # Move towards the closest positive instance
            cf = factual + 0.5 * (closest_positive - factual)
        else:
            # If no instances within epsilon, use the closest one
            closest_idx = np.argmin(distances)
            closest_positive = pos_features[closest_idx]
            cf = factual + 0.3 * (closest_positive - factual)
        
        return cf


def get_ml_model(model_name, method, dataset, force_train = False):
    """
        Retrieves and trains a machine learning model based on the specified model name and dataset.
        Parameters:
        model_name (str): The name of the model to retrieve. Supported values are 'ann' for artificial neural network and 'lr' for linear regression.
        dataset: The dataset to be used for training the model.
        force_train (bool): If True, forces the model to be retrained even if a trained model already exists. Default is False.
        Returns:
        ml_model: An instance of the trained machine learning model.
    """

    backend = get_backend(method)
    if backend == 'pytorch':
            if model_name == 'ann':
                ml_model = MLModelCatalog(
                dataset,
                model_type="ann",
                load_online=False,
                backend="pytorch"
                )
                ml_model.train(
                learning_rate=0.002,
                epochs=200,
                batch_size=1024,
                hidden_size=[16, 64, 2],
                force_train=force_train,
                )


            elif model_name == 'lr':
                ml_model = MLModelCatalog(
                    dataset,
                    model_type = "linear",
                    load_online = False,
                    use_pipeline = True,  
                    backend="pytorch"
                )
                ml_model.train(
                    learning_rate = 0.01,
                    epochs = 100,
                    batch_size = 32,
                    hidden_size = "0", 
                    force_train = force_train
                )
    else:
        ml_model = MLModelCatalog(
                dataset,
                model_type="forest",
                load_online=False,
                backend="sklearn"
                )
        ml_model.train(max_depth=2, n_estimators=5, force_train=force_train)

    return ml_model


def get_ar_model(method, dataset, ml_model, model_name):
    """
    Returns an algorithmic recourse model based on the specified method.
    Parameters:
    method (str): The name of the recourse method to use. Options include:

        - 'wacher', 
        - 'roar'
        - 'cchvae'
        - 'growing_spheres'
        - 'face'
        - 'crud'
        - 'focus'
        - 'dice'
        - 'clue'
        - 'actionable_recourse'
        - 'cem'
        - 'feature_tweek'
        - 'revise'
        - 'guided_prototypes'
        - 'moc'
    
    dataset (object): The dataset object containing the data and metadata.
    ml_model (object): The machine learning model for which recourse is to be generated.
    Returns:
    object: An instance of the specified algorithmic recourse model initialized with the appropriate hyperparameters.
    """

    if method == 'wacher':
     hyperparams = {
            "loss_type": "BCE", 
            "t_max_min": 1 / 60,
            "binary_cat_features": False
       }
     ar_model = Wachter(ml_model, hyperparams)

    elif method == 'roar':
     hyperparams = {
            "lr": 0.01,
            "lambda_": 0.01,
            "delta_max": 0.001,
            "t_max_min": 0.5,
            "loss_type": "BCE",
            "y_target": [0, 1],
            "binary_cat_features": False,
            "loss_threshold": 1e-3,
            "discretize": False,
            "sample": True
        }
     ar_model = Roar(ml_model, hyperparams)
    elif method == 'cchvae':
      hyperparams = {
            "data_name": dataset.name,
            "n_search_samples": 100,
            "p_norm": 2,
            "step": 1e-2,
            "max_iter": 1000,
            "clamp": True,
            "binary_cat_features": True,
            "vae_params": {
                "layers": [len(ml_model.feature_input_order)-\
                    len(dataset.immutables), 256,2],
                "train": True,
                "lambda_reg": 1e-6,
                "epochs": 500,
                "lr": 1e-3,
                "batch_size": 32,
            },  
        }
      ar_model = CCHVAE(ml_model, hyperparams)
    elif method == 'growing_spheres':
        ar_model = GrowingSpheres(ml_model)
    elif method == "face":
        hyperparams = {"mode": 'knn', "fraction": 0.5}
        ar_model = FaceWrapper(ml_model, hyperparams)
    elif method == 'crud':
        hyperparams ={
        "data_name": "crud", 
        "target_class": [0, 1],
        "lambda_param": 0.001,
        "optimizer": "RMSprop",
        "lr": 1e-2,
        "max_iter": 2000,
        "binary_cat_features": True,
        "vae_params": {
            "layers":  [len(ml_model.feature_input_order)-\
                    len(dataset.immutables), 256,2],
            "train": True,
            "epochs": 100,
            "lr": 1e-2,
            "batch_size": 32,
        },
        }
        ar_model = CRUD(ml_model, hyperparams)
    elif method == 'focus':
        hyperparams = {
            "optimizer": "adam",
            "lr": 0.001,
            "n_class": 2,
            "n_iter": 1000,
            "sigma": 1.0,
            "temperature": 1.0,
            "distance_weight": 0.01,
            "distance_func": "l1",
            }
        ar_model = FOCUS(ml_model, hyperparams)
    elif method == 'dice':
       hyperparams = {
            "num": 1,                    # only one CF per instance
            "desired_class": 1,          # target class label
            "posthoc_sparsity_param": 0  # disable expensive post‑hoc sparsification
            }
       ar_model = Dice(ml_model, hyperparams)
    elif method == "clue":
       hyperparams =  {
                "data_name": "custom",
                "train_vae": True,
                "width": 10,
                "depth": 5,
                "latent_dim": 12,
                "batch_size": 20,
                "epochs": 5,
                "lr": 0.001,
                "early_stop": 20,
            }
       ar_model = Clue(dataset, ml_model, hyperparams)
    elif method == "actionable_recourse":
        hyperparams = {
            "fs_size": 150, #default is 100
            "discretize": False,
            "sample": True,
        }
        coeffs, intercepts = None, None
        ar_model = ActionableRecourse(ml_model, hyperparams,  coeffs = coeffs, intercepts = intercepts)
    elif method == "cem":
       hyperparams =  {
           "mode": "PN", 
           "kappa": 0.1, 
           "data_name": "mnist", 
           "ae_params": {
               "train_ae": False, 
               "hidden_later": [10, 10], 
               "epochs": 5}}
       ar_model = CEM(ml_model, hyperparams)
    elif method == "feature_tweek":
        ar_model = FeatureTweak(ml_model)
    elif method == "revise":
        hyperparams = {
            "data_name": "test_data",
            "lambda": 0.5,
            "optimizer": "adam",
            "lr": 0.1,
            "max_iter": 1000,
            "target_class": [0,1],
            "binary_cat_features": True,
            "vae_params": {
            "layers":  [len(ml_model.feature_input_order)-\
                    len(dataset.immutables), 128,2],
            "train": True,
            "epochs": 100,
            "lr": 1e-2,
            "batch_size": 64,
        }
                }
        ar_model = Revise(ml_model, dataset, hyperparams)
    elif method == "guided_prototypes":
        hyperparams = {
            "k": 3,  # Number of prototypes (reduced for better focus)
            "beta": 0.5,  # Weight for prototype loss (increased)
            "gamma": 0.1,  # Weight for diversity loss
            "max_iter": 500,  # Reduced for faster convergence
            "lr": 0.05,  # Increased learning rate
            "tol": 1e-4  # Relaxed tolerance
        }
        ar_model = GuidedPrototypes(ml_model, hyperparams)
    elif method == "moc":
        hyperparams = {
            "num_generations": 10,      # Reduce from 50 to 10
            "population_size": 10,      # Reduce from 50 to 10
            "mutation_rate": 0.2,       # Increase for faster exploration
            "crossover_rate": 0.5,
            "num_counterfactuals": 1,
            "target_class": 1
        }
        ar_model = MOC(ml_model, hyperparams)
     
    return ar_model

def collective_counterfactual(factuals, positive_data):
    f_mask = factuals.notnull().all(axis=1)
    p_mask = positive_data.notnull().all(axis=1)
    factuals = factuals[f_mask]
    positive_data = positive_data[p_mask]
    # Compute the Cost Matrix
    C = pairwise_distances(factuals, positive_data)
    m = factuals.shape[0]
    n = positive_data.shape[0]
    P_minus = np.random.rand(m)
    P_minus /= P_minus.sum()
    P_plus = np.random.rand(n)
    P_plus /= P_plus.sum()

    lambda_1, lambda_2 = 1.0,3.0
    pi = unbalanced_ot_kl_chi2(P_minus, P_plus, C, lambda_1, 
                               lambda_2, epsilon=1e-2, 
                               max_iter=1000, tol=1e-9)
    
    # P_minus = np.ones(m)/m
    # P_plus = np.ones(n)/n    
    # eta_alpha, eta_beta = 0.01, 0.01
    # epsilon = 0.01
    # T = 100
    # pi, _, _ = unbalanced_ot_solver(C, P_minus, P_plus, lambda_1, lambda_2, 
    #                                     eta_alpha, eta_beta, epsilon, T)
    
    # max_col = np.argmax(pi, axis=1)
    # counterfactual = positive_data.iloc[max_col, [0,1]]
    # pi_top = top_k_normalize(pi, k=2)
    # counterfactual = barycentric_projection(factuals, positive_data, pi_top)
    counterfactual = sample_from_transport_plan2(factuals, positive_data, pi)
    return counterfactual, pi 