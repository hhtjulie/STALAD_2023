from STALAD.pipeline.steps.step import Step
# from STALAD.pipeline.steps.pre_do import PreDo
# from STALAD.pipeline.steps.data_preprocess import DataPreprocess2CSST


class DataSplitter(Step):
    @staticmethod
    def process(data, train_ratio, val_ratio):
        cycle_num = data.shape[0]
        train_num = round(train_ratio * cycle_num)
        val_num = round(val_ratio * cycle_num)
        train_data = data[:train_num]
        val_data = data[train_num:train_num + val_num]
        test_data = data[(train_num + val_num):]
        return train_data, val_data, test_data


if __name__ == '__main__':
    DATA_PATH = './../../../OfficialData/GetDate_20180604_RF_alarm.csv'
    SVID = 8
    STEP_NUM_COL_ID = 4
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    predo, datapreprocess2csst = PreDo(), DataPreprocess2CSST()
    raw_df = predo.process(DATA_PATH)
    time_cycle_series, freq_transformed_series = datapreprocess2csst.process(raw_df, STEP_NUM_COL_ID, SVID)
    train_data, val_data, test_data = DataSplitter.process(time_cycle_series, TRAIN_RATIO, VAL_RATIO)
    print(f'train_data {len(train_data)}:\n', train_data)
    print(f'val_data {len(val_data)}:\n', val_data)
    print(f'test_data {len(test_data)}:\n', test_data)