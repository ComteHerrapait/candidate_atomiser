import pandas as pd
import json

def read(path="resources/data.json"):
    return pd.read_json(path)


def write(dataframe, path="resources/output.json"):
    dataframe.to_json(path_or_buf = path)


if __name__ == "__main__":
    print("running local parser")
else:
    print("imported ", __name__)
