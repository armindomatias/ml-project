import os
import sys
from pathlib import Path
from typing import Dict, Any 
import pandas as pd

from common.data_manager import DataManager
from common.utils import setup_logger
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline
from pipelines.training import TrainingPipeline
from pipelines.inference import InferencePipeline
from pipelines.postprocessing import PostprocessingPipeline


class PipelineRunner:
    """
    A class that orchestrates the execution of all stages in the ML pipeline.

    This includes:
    - Preprocessing
    - Feature engineering
    - Training
    - Inference
    - Postprocessing

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        logger (logging.Logger): Logger instance for tracking pipeline execution.
        data_manager (DataManager): Manages loading/saving and transformation of data.
        real_time_data (pd.DataFrame): Cached real-time production data for inference.
        current_database_data (pd.DataFrame): Cached production database data for inference.
        prod_data_path (str): Path to the production database file.
        preprocessing_pipeline (PreprocessingPipeline): Handles data preprocessing steps.
        feature_eng_pipeline (FeatureEngineeringPipeline): Handles feature engineering steps.
        training_pipeline (TrainingPipeline): Handles model training steps.
        inference_pipeline (InferencePipeline): Handles inference steps.
        postprocessing_pipeline (PostprocessingPipeline): Handles postprocessing steps.
    """


    def __init__(self, config: Dict[str, Any], data_manager) -> None:
        """
        Initialize the pipeline runner and its pipeline components.

        Args:
            config (Dict[str, Any]): Dictionary containing all pipeline configurations.
            data_manager (DataManager): Instance for managing I/O operations on data.
        """
        self.config = config
        self.logger = setup_logger(name=__name__)
        self.data_manager = data_manager

        # Initialize individual pipeline components
        self.preprocessing_pipeline = PreprocessingPipeline(config=config)
        self.feature_eng_pipeline = FeatureEngineeringPipeline(config=config)
        self.training_pipeline = TrainingPipeline(config=config)
        self.inference_pipeline = InferencePipeline(config=config)
        self.postprocessing_pipeline = PostprocessingPipeline(config=config)

        # Load real-time data ("db" where the real time data is ingested from the source)
        self.real_time_data = self.data_manager.load_data(path=
            os.path.join(
                config.get("data_manager").get("prod_data_folder"),
                config.get("data_manager").get("real_time_data_prod_name")
            )
        )

        # Path to current production database ("db" from where my actual model is fed)
        self.prod_data_path = os.path.join(
            config.get("data_manager").get("prod_data_folder"),
            config.get("data_manager").get("prod_database_name")
        )

        # Load existing production database ("db" from where my actual model is fed)
        self.current_database_name = self.data_manager.load_data(path=self.prod_data_path)

    def run_training(self) -> None:
        """
        Run the full training pipeline:
        1. Load and preprocess data
        2. Perform feature engineering
        3. Train the model
        4. Save the trained model

        Returns:
            None
        """
        try:
            self.logger.info("Starting training pipeline...")
            
            self.logger.info("Loading data from production database...")
            df = self.data_manager.load_data(self.prod_data_path)
            self.logger.info(f"Loaded data shape: {df.shape}")
            
            self.logger.info("Running preprocessing pipeline...")
            df = self.preprocessing_pipeline.run(df=df)
            self.logger.info(f"After preprocessing shape: {df.shape}")
            
            self.logger.info("Running feature engineering pipeline...")
            df = self.feature_eng_pipeline.run(df=df)
            self.logger.info(f"After feature engineering shape: {df.shape}")
            
            self.logger.info("Starting model training...")
            model = self.training_pipeline.run(df=df)
            self.logger.info("Model training completed successfully")
            
            self.logger.info("Saving trained model...")
            self.postprocessing_pipeline.run_train(model=model)
            self.logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed with error: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise
        
        return

    def run_inference(self, current_timestamp: pd.Timestamp) -> None:
        """
        Run the full inference pipeline:
        1. Load real-time data for the current timestamp
        2. Append to the production database
        3. Prepare the latest batch
        4. Preprocess, transform, and predict
        5. Postprocess and store the prediction
        6. Update the production database

        Args:
            current_timestamp (pd.Timestamp): The timestamp for which to run inference.

        Returns:
            None
        """
        try:
            self.logger.info(f"Starting inference pipeline for timestamp: {current_timestamp}")
            
            # Step 1: Retrieve real-time data for the current timestamp
            self.logger.info("Retrieving real-time data for current timestamp...")
            current_real_time_data = self.data_manager.get_timestamp_data(
                data = self.real_time_data,
                timestamp = current_timestamp
            )
            self.logger.info(f"Retrieved real-time data shape: {current_real_time_data.shape}")

            # Step 2: Append new data to production database
            self.logger.info("Appending new data to production database...")
            self.current_database_data = self.data_manager.append_data(
                current_data=self.current_database_data,
                new_data=current_real_time_data
            )
            self.logger.info(f"Updated database shape: {self.current_database_data.shape}")

            # Step 3: Get the last N rows as the latest batch
            batch_size = self.config.get("pipeline_runner").get("batch_size")
            self.logger.info(f"Getting last {batch_size} rows for batch processing...")
            df = self.data_manager.get_n_last_points(
                data=self.current_database_data,
                n=batch_size
            )
            self.logger.info(f"Batch data shape: {df.shape}")

            # Step 4: Run preprocessing and feature engineering
            self.logger.info("Running preprocessing pipeline...")
            df = self.preprocessing_pipeline.run(df=df)
            self.logger.info(f"After preprocessing shape: {df.shape}")
            
            self.logger.info("Running feature engineering pipeline...")
            df = self.feature_eng_pipeline.run(df=df)
            self.logger.info(f"After feature engineering shape: {df.shape}")

            # Step 5: Run inference
            self.logger.info("Running inference...")
            y_pred = self.inference_pipeline.run(x=df)
            self.logger.info(f"Generated prediction: {y_pred}")

            # Step 6: Postprocessing and saving the prediction
            self.logger.info("Running postprocessing...")
            df_pred = self.postprocessing_pipeline.run_inference(y_pred=y_pred, current_timestamp=current_timestamp)

            # Step 7: Save the prediction and updated database
            self.logger.info("Saving predictions and updated database...")
            self.data_manager.save_predictions(df_pred, current_timestamp)
            self.logger.info("Inference pipeline completed successfully")
            self.data_manager.save_data(data=self.current_database_data, path=self.prod_data_path)
            
        except Exception as e:
            self.logger.error(f"Inference pipeline failed with error: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise
        
        return        
