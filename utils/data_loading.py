from pymongo import MongoClient
import json
from pathlib import Path
import logging
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

load_dotenv()

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def connect_mongodb(connection_string):
    """
    Connect to MongoDB using a connection string
    
    Args:
        connection_string (str): MongoDB connection string URI
    
    Returns:
        tuple: (client, database) MongoDB client and database instances
    """
    try:
        client = MongoClient(connection_string)
        db_name = 'd-Nexus'
        db = client[db_name]
        
        # Test connection
        client.server_info()
        logger.info("Successfully connected to MongoDB")
        return client, db
    
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

def import_jsonl_to_mongodb(file_path, collection, batch_size=1000):
    """
    Import documents from a JSONL file to MongoDB collection
    
    Args:
        file_path (str): Path to the JSONL file
        collection: MongoDB collection object
        batch_size (int): Number of documents to insert in each batch
    
    Returns:
        int: Number of documents imported
    """
    documents = []
    total_imported = 0
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Parse each line as a JSON document
                doc = json.loads(line.strip())
                documents.append(doc)
                
                # Insert in batches for better performance
                if len(documents) >= batch_size:
                    collection.insert_many(documents)
                    total_imported += len(documents)
                    logger.info(f"Imported {total_imported} documents...")
                    documents = []
            
            # Insert any remaining documents
            if documents:
                collection.insert_many(documents)
                total_imported += len(documents)
                
        return total_imported
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error importing data: {str(e)}")
        raise

def main():
    """Main function to handle the import process"""
   
    connection_string = os.getenv("MONGODB_CONNECTION_STRING")

    # File paths
    train_file = 'data/train_data/train.jsonl'
    test_file = 'data/test_data/test.jsonl'
    
    try:
        # Connect to MongoDB
        client, db = connect_mongodb(connection_string)
        
        # Create or get collections
        train_collection = db['training']
        test_collection = db['testing']
        
        # Clear existing data if any
        train_collection.drop()
        test_collection.drop()
        logger.info("Cleared existing collections")
        
        # Import training data
        train_count = import_jsonl_to_mongodb(train_file, train_collection)
        logger.info(f"Successfully imported {train_count} training documents")
        
        # Import testing data
        test_count = import_jsonl_to_mongodb(test_file, test_collection)
        logger.info(f"Successfully imported {test_count} testing documents")
        
        
        logger.info("Import completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to complete import: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
            logger.info("Database connection closed")

# if __name__ == "__main__":
#     # Setup logging
#     logger = setup_logging()
    
#     load_dotenv()
#     main()