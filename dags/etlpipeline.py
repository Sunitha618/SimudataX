from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from utils import data_extraction , data_transformation , data_loading
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting dag...")

default_args={
    'owner':'airflow',
    'start_date':days_ago(1)
}

with DAG(dag_id='etl_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dags:
    
    @task
    def extract():
        try:

            prompt = data_extraction.Prompt()
            generator = data_extraction.DataGenerator()

            logger.info("Started Data extraction...")
            df, json_path = generator.generate_and_save_examples(
                prompt=prompt.prompt,
                number_of_examples=5,
                filename="data.json"
            )
            logger.info("Data extraction completed successfully!")
            return df , json_path
    
        except Exception as e:
            logger.error("Data extraction failed!", exc_info=True)
            raise

    @task


    def transform():
        input_file = "data/extracted_data/data.json"
        train_dir = "data/train_data"
        test_dir = "data/test_data"
        test_size = 0.1

        try:
            logger.info("Starting Data processing...")
        
            with open(input_file, 'r') as f:
                json_data = json.load(f)
        
            data_df = data_transformation.extract_data(json_data)
            train_df, test_df = data_transformation.manual_train_test_split(data_df, test_size=test_size)
            data_transformation.save_data(train_df, test_df, train_dir, test_dir)
            logger.info("Data processing completed successfully!")
        
        except Exception as e:
            logger.error("Data processing failed!", exc_info=True)
            raise


    @task
    def load():
        try:
            logger.info("Starting Data Loading...")
            data_loading.main()
            logger.info("Data Loading completed successfully!")
        except Exception as e:
            logger.error("Data Loading failed!", exc_info=True)
            raise

    extract() >> transform() >> load()