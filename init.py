import os
import csv
import json
from dotenv import load_dotenv

def convert_csv_to_json(CSV_FILE):
    

    data = []
    with open(CSV_FILE, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        column_names = csv_reader.fieldnames  # Get the column names from the CSV
        for row in csv_reader:
            data.append(row)
            

    return json.dumps(data, indent=4)


print("This is the init.py file.")
# You can add initialization code here if needed.

# The data is already loaded into mongodb
# The embeddings are already generated and stored in the database on Turbopuffer
# Loading the environment variables from the .env file

load_dotenv('.env')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
TURBOPUFFER_API_KEY = os.environ.get('TURBOPUFFER_API_KEY')
VOYAGEAI_API_KEY = os.environ.get('VOYAGEAI_API_KEY')


json_data = convert_csv_to_json("Off Platform Search Evaluation Criteria - Sheet1.csv")
with open("queries.json", "w") as json_file:
    json_file.write(json_data)
    
    