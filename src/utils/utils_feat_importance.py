import pandas as pd
import os
import mlflow


def log_feature_importance(trial_number, model, X, fold_n, exp_purpose, exp_date_str):
    """
    Logs the feature importances for a given model and fold number.
    """

    feature_importances = model.feature_importances_
    new_importance_df = pd.DataFrame(
        {"feat": X.columns, f"t{trial_number}_imp_fold_{fold_n+1}": feature_importances}
    )

    csv_path = f"feat_impor_{exp_purpose}_{exp_date_str}.csv"

    # Check if the CSV already exists
    if os.path.exists(csv_path):
        # If so, read it and merge with the new importance values
        existing_df = pd.read_csv(csv_path)
        importance_df = pd.merge(existing_df, new_importance_df, on="feat", how="outer")
    else:
        # If not, create a new DataFrame
        importance_df = new_importance_df

    # Save the updated DataFrame to CSV
    importance_df.to_csv(csv_path, index=False)


#    mlflow.log_artifact(csv_path)


def aggregate_feature_importance(list_files_feat_importance):
    list_of_dfs = []
    for file_path in list_files_feat_importance:
        feature_importance_df = pd.read_csv(file_path)

        folds = [col for col in feature_importance_df.columns if "imp_fold" in col]

        # Normalize by dividing each score by the sum of scores within its respective fold
        for fold in folds:
            fold_sum = feature_importance_df[fold].sum()
            feature_importance_df[fold] = feature_importance_df[fold] / fold_sum

        list_of_dfs.append(feature_importance_df)

    aggregated_df = pd.concat(list_of_dfs, ignore_index=True)

    df_median_importance = aggregated_df.groupby("feat").median().reset_index()

    df_median_importance["feat_imp_overall_mean"] = df_median_importance.loc[
        :, df_median_importance.columns != "feat"
    ].median(axis=1, skipna=True)
    cols = ["feat", "feat_imp_overall_mean"] + [
        col
        for col in df_median_importance.columns
        if col not in ["feat_imp_overall_mean", "feat"]
    ]
    df_median_importance = df_median_importance[cols]

    df_median_importance.sort_values(
        "feat_imp_overall_mean", ascending=False, inplace=True
    )
    return df_median_importance
