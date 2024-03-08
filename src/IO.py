import json
from datetime import datetime


class IO:

    def __init__(self):
        pass

    @staticmethod
    def write_to_file(f1, filepath, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data = {'timestamp': timestamp, 'f1_scores': f1}

        with open(filepath, 'a') as file:
            file.write(json.dumps(data) + "\n")

    @staticmethod
    def read_f1_from_file(filepath):
        results = []
        with open(filepath, 'r') as file:
            for line in file:
                data = json.loads(line)
                results.append(data)
        return results
