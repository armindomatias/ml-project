import pandas as pd 
from typing import Dict, List, Any 


class FeatureEngineeringPipeline:
    """
    A pipeline for creating and engineering features from preprocessed data.

    This class handles feature engineering steps including:
    - Creating lag features for time series data

    Args:
        config (Dict[str, Any]): Configuration dictionary containing feature engineering parameters
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config.get("feature_engineering")

    @staticmethod
    def add_lag_feats(df: pd.DataFrame, params: Dict[str, List[int]]) -> pd.DataFrame:
        """
        Add lag features to the DataFrame based on specified parameters.

        Args:
            df (pd.DataFrame): Input DataFrame
            params (Dict[str, List[int]]): Dictionary containing feature names and their corresponding lag periods.
                Example: {
                    'col1': [1, 2, 3],
                    'col2': [1, 5, 10],
                    ...
                }

        Returns:
            pd.DataFrame: DataFrame with added lag features
        """

        for feat, lags in params.items():
            if feat not in df.columns:
                print(f"⚠️  Warning: Column '{feat}' not found in DataFrame. Available columns: {list(df.columns)}")
                continue
                
            for lag in lags: 
                print(f"Creating lag feature: {feat}_lag_{lag}")
                df[f"{feat}_lag_{lag}"] = df[feat].shift(lag).bfill()

        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete feature engineering pipeline on the input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to be processed

        Returns:
            pd.DataFrame: DataFrame with engineered features including lag features
        """
        df = self.add_lag_feats(df=df, params=self.config.get("lag_params"))
        return df