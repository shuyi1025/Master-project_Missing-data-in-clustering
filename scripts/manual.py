import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score


K = 5
B = 500
RANDOM_STATE = 123
MISSINGNESS_RATES = [0.10, 0.20, 0.30, 0.40, 0.50]
MECHANISMS = ["MCAR", "MAR", "MNAR"]

MAR_DRIVER_COLS = ["totalmins", "pageviews", "posts"]
MISSINGNESS_LOGIT_SLOPE = -1.25

def standardize_score(score):
    score = np.asarray(score, dtype=float)
    return (score - score.mean()) / score.std()


def logistic_missingness_prob(driver, target_rate):
    score = standardize_score(driver)
    intercept = calibrate_intercept(score, target_rate)

    prob = 1 / (1 + np.exp(-(intercept + MISSINGNESS_LOGIT_SLOPE * score)))

    return prob


def calibrate_intercept(score, target_rate):
    low = -20
    high = 20
    # Set a wide search range for the intercept.
    # The intercept controls the overall average missingness probability.
    for i in range(80):
        mid = (low + high) / 2
    # Use binary search to find the intercept that gives the target missingness rate.
        prob = 1 / (1 + np.exp(-(mid + MISSINGNESS_LOGIT_SLOPE * score)))
        # If the average probability is too low, increase the intercept.
        if prob.mean() < target_rate:
            low = mid
        else:
            high = mid
    # Return the calibrated intercept.
    return (low + high) / 2


def inject_missingness(x_complete, mechanism, rate, random_state):
    # Create a random number generator for reproducible missingness patterns.
    rng = np.random.default_rng(random_state)
    # Convert the missingness mechanism name to uppercase for consistency.
    mechanism = mechanism.upper()
# mask is a Boolean matrix with the same shape as the dataset. 
#True means that the cell is set to missing; False means it remains observed.
    if mechanism == "MCAR":
        mask = rng.random(x_complete.shape) < rate

    elif mechanism == "MAR":
        mask = np.zeros(x_complete.shape, dtype=bool)

        for j, target_col in enumerate(x_complete.columns):
        # Exclude the target variable itself from the MAR drivers.
            driver_cols = [
                col for col in MAR_DRIVER_COLS
                if col != target_col
            ]
            # Use the average of the driver variables as the missingness score.
            driver_score = x_complete[driver_cols].mean(axis=1)
            # Convert the score into missingness probabilities.
            prob = logistic_missingness_prob(driver_score, rate)
            # Generate the missingness indicator for this column.
            mask[:, j] = rng.random(len(x_complete)) < prob

    elif mechanism == "MNAR":
        mask = np.zeros(x_complete.shape, dtype=bool) #boolean： T/F
        # Use the target variable itself to determine missingness.
        for j, target_col in enumerate(x_complete.columns):

            prob = logistic_missingness_prob(x_complete[target_col], rate)

            mask[:, j] = rng.random(len(x_complete)) < prob

    else:
        raise ValueError("Unknown missingness mechanism")
    # Store the mask as a DataFrame and apply it to create missing values.
    mask = pd.DataFrame(mask, columns=x_complete.columns, index=x_complete.index)

    x_missing = x_complete.mask(mask)

    return x_missing, mask
#compares a random number for each observation with its calculated missingness probability. 
#If the random number is smaller than the probability, the corresponding position in the mask is set to True



def impute_median(x_missing):
    # Median imputation replaces missing values by the median of each column.
    imputer = SimpleImputer(strategy="median")

    x_completed_array = imputer.fit_transform(x_missing)

    x_completed = pd.DataFrame(
        x_completed_array,
        columns=x_missing.columns,
        index=x_missing.index
    )

    return x_completed


def impute_knn(x_missing):
    # KNN imputation fills each missing value using similar observations.
    # n_neighbors=5 means that the five nearest observations are used.
    imputer = KNNImputer(n_neighbors=5, weights="distance")

    # Fit the imputer and transform the incomplete dataset.
    x_completed_array = imputer.fit_transform(x_missing)

    # Convert the output back to a DataFrame.
    x_completed = pd.DataFrame(
        x_completed_array,
        columns=x_missing.columns,
        index=x_missing.index
    )

    return x_completed


def impute_random_forest(x_missing):
    # Random forest is used as the prediction model inside iterative imputation.
    estimator = RandomForestRegressor(
        n_estimators=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=3
    )

    # IterativeImputer repeatedly predicts missing values column by column.
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=10,
        tol=0.3,
        random_state=RANDOM_STATE,
        initial_strategy="median",
        skip_complete=True
    )

    x_completed_array = imputer.fit_transform(x_missing)

    x_completed = pd.DataFrame(
        x_completed_array,
        columns=x_missing.columns,
        index=x_missing.index
    )

    return x_completed


def pmm_once(x_missing):
    rng = np.random.default_rng(RANDOM_STATE)

    # Start with median-filled data.
    x_filled = impute_median(x_missing)

    # Record which entries were originally missing.
    missing_mask = x_missing.isna()

    for iteration in range(10):

        for target_col in x_missing.columns:

            # Identify missing and observed rows for the current target variable.
            missing_rows = missing_mask[target_col].to_numpy()

            if not missing_rows.any():
                continue

            observed_rows = ~missing_rows

            # Use all other variables as predictors.
            predictor_cols = [
                col for col in x_missing.columns
                if col != target_col
            ]

            x_obs = x_filled.loc[observed_rows, predictor_cols]
            y_obs = x_filled.loc[observed_rows, target_col]
            x_mis = x_filled.loc[missing_rows, predictor_cols]

            # Fit a regression model using observed rows.
            model = BayesianRidge()
            model.fit(x_obs, y_obs)

            # Predict both observed and missing rows.
            pred_obs = model.predict(x_obs)
            pred_mis = model.predict(x_mis)

            donor_values = y_obs.to_numpy()

            imputed_values = []

            for pred in pred_mis:
                #a donor is an observed case whose predicted value is close to the predicted value of a missing case.
                # Find observed donors with the closest predicted values.
                donor_index = np.argsort(np.abs(pred_obs - pred))[:5]

                # Randomly choose one donor value.
                imputed_values.append(
                    rng.choice(donor_values[donor_index])
                )

            # Fill the missing values in the current target column.
            x_filled.loc[missing_rows, target_col] = imputed_values

    return x_filled


def impute_mice_pmm(x_missing, n_imputations=10):
    completed_datasets = []

    for m in range(n_imputations):
        x_completed = pmm_once(x_missing)
        completed_datasets.append(x_completed)

    return completed_datasets




def run_kpod(x_missing):
    # Step 1: initially fill missing values using column medians.
    x_filled = impute_median(x_missing)

    # Store the original missing positions.
    # Only these positions will be updated during K-POD.
    missing_mask = x_missing.isna().to_numpy()

    previous_centers = None

    for iteration in range(50):

        # Step 2: run K-means on the temporarily completed dataset.
        model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(x_filled)
        centers = model.cluster_centers_

        # Step 3: replace missing entries by their assigned cluster centroid values.
        values = x_filled.to_numpy(copy=True)

        rows, cols = np.where(missing_mask)

        values[rows, cols] = centers[labels[rows], cols]

        x_filled = pd.DataFrame(
            values,
            columns=x_missing.columns,
            index=x_missing.index
        )

        # Step 4: stop if the cluster centres become stable.
        if previous_centers is not None:
            change = np.linalg.norm(centers - previous_centers)

            if change < 1e-4:
                break

        previous_centers = centers.copy()

    return x_filled




def run_simulation(x_complete, benchmark_model, benchmark_labels, method_name, impute_function):
    results = []

    for b in range(1, B + 1):

        for mechanism in MISSINGNESS_MECHANISMS:

            for rate in MISSINGNESS_RATES:

                # Inject missing values under MCAR, MAR, or MNAR.
                x_missing, mask = inject_missingness(
                    x_complete,
                    mechanism,
                    rate,
                    random_state=seed("missingness", b, int(rate * 1000), mechanism)
                )

                # Apply the selected missing-data method.
                x_completed = impute_function(x_missing)

                # Run K-means on the completed dataset.
                model, labels = run_kmeans(
                    x_completed,
                    random_state=seed("kmeans", b, int(rate * 1000), mechanism, method_name)
                )

                # Compare estimated labels with the full-data K-means benchmark.
                metrics = compute_metrics(
                    x_completed,
                    model,
                    labels,
                    benchmark_model,
                    benchmark_labels
                )

                # Store one row of simulation results.
                results.append({
                    "replication": b,
                    "mechanism": mechanism,
                    "target_missing_rate": rate,
                    "observed_missing_rate": mask.to_numpy().mean(),
                    "method": method_name,
                    **metrics
                })

    return pd.DataFrame(results)



def cluster_proportions(labels):
    # Count how many observations are assigned to each cluster.
    counts = np.bincount(labels, minlength=K)

    # Convert cluster counts into cluster proportions.
    proportions = counts / counts.sum()

    return proportions


def best_label_permutation(reference_centers, estimated_centers):
    # K-means cluster labels are arbitrary.
    # For example, cluster 1 in one run may correspond to cluster 3 in another run.
    # Therefore, we need to match estimated clusters to benchmark clusters.

    best_perm = None
    best_cost = np.inf

    # Try all possible ways of matching the estimated clusters to the reference clusters.
    for perm in permutations(range(K)):

        # Reorder the estimated centers according to the current permutation.
        reordered_centers = estimated_centers[list(perm)]

        # Calculate the total distance between reference centers and reordered estimated centers.
        cost = np.linalg.norm(reference_centers - reordered_centers, axis=1).sum()

        # Keep the permutation with the smallest total centroid distance.
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    return list(best_perm)



 def compute_metrics(x_completed, estimated_model, estimated_labels,
                    benchmark_model, benchmark_labels):

    # Step 1: match estimated cluster labels to benchmark cluster labels.
    perm = best_label_permutation(
        benchmark_model.cluster_centers_,
        estimated_model.cluster_centers_
    )

    # Step 2: reorder estimated cluster centers based on the best matching.
    matched_centers = estimated_model.cluster_centers_[perm]

    # Step 3: calculate the distance between each benchmark centroid
    # and its matched estimated centroid.
    centroid_distances = np.linalg.norm(
        benchmark_model.cluster_centers_ - matched_centers,
        axis=1
    )

    # Step 4: calculate cluster size proportions.
    estimated_props = cluster_proportions(estimated_labels)[perm]
    benchmark_props = cluster_proportions(benchmark_labels)

    # Step 5: calculate ARI.
    # ARI compares the estimated K-means labels with the full-data benchmark labels.
    ari = adjusted_rand_score(benchmark_labels, estimated_labels)

    # Step 6: calculate centroid errors.
    centroid_error_mean = centroid_distances.mean()
    centroid_error_max = centroid_distances.max()

    # Step 7: calculate the average absolute difference in cluster proportions.
    cluster_size_error = np.abs(benchmark_props - estimated_props).mean()

    # Step 8: calculate silhouette score.
    # Silhouette measures how well separated the clusters are.
    if len(np.unique(estimated_labels)) > 1:
        silhouette = silhouette_score(x_completed, estimated_labels)
    else:
        silhouette = np.nan

    # Step 9: return all performance metrics.
    return {
        "ari_vs_full_data_kmeans": ari,
        "centroid_error_mean": centroid_error_mean,
        "centroid_error_max": centroid_error_max,
        "cluster_size_error_mean_abs": cluster_size_error,
        "silhouette": silhouette
    }   