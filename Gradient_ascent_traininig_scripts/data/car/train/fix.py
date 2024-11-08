import json

# Define the range and the common text description
start, end = 1, 1000
description = "a car"

# Open the metadata.jsonl file to write the JSON data
with open("metadata.jsonl", "w") as file:
    # Iterate through the range and format each entry
    for i in range(start, end + 1):
        entry = {
            "file_name": f"train/{i}.jpg",
            "text": description
        }
        # Write each entry as a JSON line
        file.write(json.dumps(entry) + "\n")
