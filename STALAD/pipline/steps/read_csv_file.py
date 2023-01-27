import pandas as pd


class ReadCSVFile:
    def process(self, file_path):
        try:
            raw_df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File '{file_path}' does not exist. Loading ESD failed.")

