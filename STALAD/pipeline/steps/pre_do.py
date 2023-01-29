import pandas as pd
from STALAD.pipeline.steps.step import Step


class PreDo(Step):
    def process(self, data_path):
        raw_df = self.read_csv_file(data_path)
        return raw_df

    @staticmethod
    def read_csv_file(data_path):
        try:
            raw_df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"File '{data_path}' does not exist. Loading ESD failed.")
        return raw_df


if __name__ == '__main__':
    DATA_PATH = './../../../OfficialData/GetDate_20180604_RF_alarm.csv'
    predo = PreDo()
    raw_df = predo.process(DATA_PATH)
    print(raw_df)
