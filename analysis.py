import os, sys, json

def total_time(path: str):
    total_time = 0
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), "r") as file: data = json.load(file)
        for value in data["configs"].values(): total_time += value["summary"]["time"]
    return total_time

if __name__ == "__main__":
    dir_path = os.path.join(os.path.abspath("output"), f"data-{sys.argv[1]}")
    print(total_time(dir_path))