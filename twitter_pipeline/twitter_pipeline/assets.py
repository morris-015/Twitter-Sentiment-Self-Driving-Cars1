
import base64
from io import BytesIO
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn
import xgboost as xgb
from dagster import (
    AssetOut,
    IOManager,
    MetadataValue,
    Output,
    asset,
    io_manager,
    multi_asset,
    AutomationCondition,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

asset_group_name = "mlops_dasci270"


class LocalCSVIOManager(IOManager):
    """
    A custom IOManager to handle saving and loading CSV files locally.
    """

    def handle_output(self, context, obj: pd.DataFrame) -> None:
        """
        Save a Pandas DataFrame to a CSV file.
        Args:
            context: The context object provided by Dagster.
            obj (pd.DataFrame): The DataFrame to save.
        """
        obj.to_csv(f"{context.asset_key.path[-1]}.csv")

    def load_input(self, context) -> pd.DataFrame:
        """
        Load a Pandas DataFrame from a CSV file.
        Args:
            context: The context object provided by Dagster.
        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        return pd.read_csv(f"{context.asset_key.path[-1]}.csv")


@io_manager
def local_csv_io_manager() -> LocalCSVIOManager:
    """
    Instantiate the custom CSV IOManager.
    Returns:
        LocalCSVIOManager: An instance of the custom IOManager.
    """
    return LocalCSVIOManager()


@asset(group_name=asset_group_name, compute_kind="pandas", io_manager_key="local_csv_io_manager")
def hackernews_stories() -> pd.DataFrame:
    """
    Fetch the latest stories from Hacker News.
    Returns:
        pd.DataFrame: A DataFrame containing story data.
    """
    latest_item = requests.get("https://hacker-news.firebaseio.com/v0/maxitem.json").json()
    results = []
    scope = range(latest_item - 1000, latest_item - 100)
    for item_id in scope:
        item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json").json()
        results.append(item)
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df[df.type == "story"]
        df = df[~df.title.isna()]
    return df


@multi_asset(
    group_name=asset_group_name,
    compute_kind="scikit-learn",
    outs={"training_data": AssetOut(), "test_data": AssetOut()},
)
def training_test_data(hackernews_stories: pd.DataFrame) -> Tuple[Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series]]:
    """
    Split the Hacker News stories dataset into training and testing sets.
    Args:
        hackernews_stories (pd.DataFrame): The input DataFrame containing story data.
    Returns:
        Tuple[Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series]]: Training and testing data splits.
    """
    X = hackernews_stories.title
    y = hackernews_stories.descendants
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return (X_train, y_train), (X_test, y_test)


@multi_asset(
    group_name=asset_group_name,
    compute_kind="scikit-learn",
    outs={"tfidf_vectorizer": AssetOut(), "transformed_training_data": AssetOut()},
)
def transformed_train_data(
    training_data: Tuple[pd.Series, pd.Series]
) -> Tuple[TfidfVectorizer, Tuple[np.ndarray, np.ndarray]]:
    """
    Transform the training data using TF-IDF vectorization.
    Args:
        training_data (Tuple[pd.Series, pd.Series]): The training data (features and labels).
    Returns:
        Tuple[TfidfVectorizer, Tuple[np.ndarray, np.ndarray]]: TF-IDF vectorizer and transformed training data.
    """
    X_train, y_train = training_data
    vectorizer = TfidfVectorizer()
    transformed_X_train = vectorizer.fit_transform(X_train).toarray()
    y_train = y_train.fillna(0)
    transformed_y_train = np.array(y_train)
    return vectorizer, (transformed_X_train, transformed_y_train)


@asset(group_name=asset_group_name)
def transformed_test_data(
    test_data: Tuple[pd.Series, pd.Series], tfidf_vectorizer: TfidfVectorizer
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform the test data using a pre-trained TF-IDF vectorizer.
    Args:
        test_data (Tuple[pd.Series, pd.Series]): The test data (features and labels).
        tfidf_vectorizer (TfidfVectorizer): The pre-trained TF-IDF vectorizer.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed test data.
    """
    X_test, y_test = test_data
    transformed_X_test = tfidf_vectorizer.transform(X_test)
    y_test = y_test.fillna(0)
    transformed_y_test = np.array(y_test)
    return transformed_X_test, transformed_y_test


def make_plot(eval_metric: Dict[str, Any]) -> MetadataValue:
    """
    Create a plot from evaluation metrics.
    Args:
        eval_metric (Dict[str, Any]): The evaluation metrics.
    Returns:
        MetadataValue: A MetadataValue containing the plot as a base64 image.
    """
    plt.clf()
    training_plot = seaborn.lineplot(eval_metric)
    fig = training_plot.get_figure()
    buffer = BytesIO()
    fig.savefig(buffer)
    image_data = base64.b64encode(buffer.getvalue())
    return MetadataValue.md(f"![img](data:image/png;base64,{image_data.decode()})")


@asset(
    group_name=asset_group_name,
    automation_condition=AutomationCondition.eager(),
    compute_kind="xgboost",
    metadata={"library": "xgboost"},
)
def xgboost_comments_model(
    transformed_training_data: Tuple[np.ndarray, np.ndarray], transformed_test_data: Tuple[np.ndarray, np.ndarray]
) -> Output:
    """
    Train an XGBoost model to predict comment counts.
    Args:
        transformed_training_data (Tuple[np.ndarray, np.ndarray]): The transformed training data.
        transformed_test_data (Tuple[np.ndarray, np.ndarray]): The transformed test data.
    Returns:
        Output: The trained model and metadata.
    """
    transformed_X_train, transformed_y_train = transformed_training_data
    transformed_X_test, transformed_y_test = transformed_test_data
    xgb_r = xgb.XGBRegressor(objective="reg:squarederror", eval_metric=mean_absolute_error, n_estimators=20)
    xgb_r.fit(
        transformed_X_train,
        transformed_y_train,
        eval_set=[(transformed_X_test, transformed_y_test)],
    )

    metadata = {}
    for eval_metric in xgb_r.evals_result()["validation_0"].keys():
        metadata[f"{eval_metric} plot"] = make_plot(xgb_r.evals_result()["validation_0"][eval_metric])
    metadata["score (mean_absolute_error)"] = xgb_r.evals_result()["validation_0"]["mean_absolute_error"][-1]

    return Output(xgb_r, metadata=metadata)


@asset(
    group_name=asset_group_name,
    compute_kind="xgboost",
)
def comments_model_test_set_r_squared(
    transformed_test_data: Tuple[np.ndarray, np.ndarray], xgboost_comments_model: xgb.XGBRegressor
) -> float:
    """
    Evaluate the R-squared score of the model on the test set.
    Args:
        transformed_test_data (Tuple[np.ndarray, np.ndarray]): The transformed test data.
        xgboost_comments_model (xgb.XGBRegressor): The trained XGBoost model.
    Returns:
        float: The R-squared score.
    """
    transformed_X_test, transformed_y_test = transformed_test_data
    score = xgboost_comments_model.score(transformed_X_test, transformed_y_test)
    return score


@asset(group_name=asset_group_name, compute_kind="xgboost")
def latest_story_comment_predictions(
    xgboost_comments_model: xgb.XGBRegressor, tfidf_vectorizer: TfidfVectorizer
) -> np.ndarray:
    """
    Predict comment counts for the latest stories.
    Args:
        xgboost_comments_model (xgb.XGBRegressor): The trained XGBoost model.
        tfidf_vectorizer (TfidfVectorizer): The pre-trained TF-IDF vectorizer.
    Returns:
        np.ndarray: The predicted comment counts.
    """
    latest_item = requests.get("https://hacker-news.firebaseio.com/v0/maxitem.json").json()
    results = []
    scope = range(latest_item - 100, latest_item)
    for item_id in scope:
        item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json").json()
        results.append(item)

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df[df.type == "story"]
        df = df[~df.title.isna()]
    inference_x = df.title
    inference_x = tfidf_vectorizer.transform(inference_x)
    return xgboost_comments_model.predict(inference_x)