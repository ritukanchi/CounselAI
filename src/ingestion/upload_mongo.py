import os 
import json 
from pymongo import MongoClient 
from dotenv import load_dotenv

load_dotenv()
mongo_url = os.getenv('MONGO_URL')

config_file = "config.json"
with open(config_file, "r") as file:
    config = json.load(file)

collection_name = config['COLLECTION_NAME']['ipc-800']
mongo_db_name = config['MONGODB']

def insert_to_mongo(url, db_name, collection_name, section_type): 
    
    client = MongoClient(url)
    mongo_db = client[mongo_db_name]
    collection_name = mongo_db[collection_name]
    
    collection = mongo_db.create_collection(collection_name)

    print("Loading data from JSON")
    
    with open(f'./data/{section_type}_data.json', 'r') as ipc_data_file:
        ipc_data = json.load(ipc_data_file)

    print("Inserting the data")
    
    # Insert data into MongoDB idts im using it tho
    if isinstance(ipc_data, list):
        result = collection.insert_many(ipc_data)
    else:
        result = collection.insert_one(ipcs_data)
        
    print(f"Data inserted")


if __name__ == '__main__':
    insert_to_mongo(mongo_url, mongo_db_name, "crpc-data-001", "crpc")