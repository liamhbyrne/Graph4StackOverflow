"""
A basic remote monitoring system which sends logs about dataset creation to slack
"""
import json

import requests
import yaml
import os
import re
import time

with open('secrets.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)


def send_slack_message(payload):
    return requests.post(CONFIG['slack_logging_webhook'], json.dumps(payload))


# Get number of questions created in the processed folder
processed_folder_path = r"..\..\data\processed"

# Each file name is formatted as data_{i}_question_id_{question_id}_answer_id_{answer_id}.pt
# We want to get the number of unique question ids


# Run the following code every 30 minutes
while True:
    file_names = os.listdir(processed_folder_path)

    question_ids = [re.search(r"question_id_(\d+)_", x).groups()[0] for x in file_names]
    question_ids = set(question_ids)

    # Get docker logs

    send_slack_message({"text": f"Number of questions processed: {len(question_ids)}"})
    send_slack_message({"text": f"Number of files in directory: {len(file_names)}"})

    print("Sent slack message, sleeping . . .")
    time.sleep(60 * 30)
