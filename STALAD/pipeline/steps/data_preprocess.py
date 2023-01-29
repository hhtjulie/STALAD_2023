import numpy as np
# from STALAD.pipeline.steps.pre_do import PreDo


class DataPreprocess2CSST:
    def process(self, raw_df, step_num_col_idx, SVID_idx):
        neg1_row_index, pos1_row_index = self.get_cycle_identification_index(raw_df, step_num_col_idx)
        single_SVID_series = self.get_single_SVID_series(raw_df, SVID_idx)
        time_cycle_series = self.get_time_cycle_series(neg1_row_index, pos1_row_index, single_SVID_series)
        freq_transformed_series = self.spectral_transformation(time_cycle_series)
        return time_cycle_series, freq_transformed_series

    @staticmethod
    def get_cycle_identification_index(raw_df, step_num_col_idx: int):
        step_number_series = raw_df.iloc[:, step_num_col_idx]
        is_first_neg1 = True  # mark the just changes of step number to -1
        neg1_row_index = []
        pos1_row_index = []
        for i in range(len(step_number_series)):
            if step_number_series[i] == -1:  # note when the step number is -1
                if is_first_neg1:  # enter the next cycle if the step number just changes to -1
                    neg1_row_index.append(i)
                    is_first_neg1 = False
            else:
                if not is_first_neg1:
                    pos1_row_index.append(i)
                    is_first_neg1 = True
        return neg1_row_index, pos1_row_index

    @staticmethod
    def get_single_SVID_series(raw_df, SVID_idx: int):
        single_SVID_series = raw_df.iloc[:, SVID_idx]
        return single_SVID_series

    @staticmethod
    def get_time_cycle_series(neg1_row_index, pos1_row_index, single_SVID_series) -> list:
        cycles_single_SVID = []
        for i in range(len(pos1_row_index)):
            if i == len(pos1_row_index) - 1:
                single_cycle = list(single_SVID_series[pos1_row_index[i]:])
            else:
                single_cycle = list(single_SVID_series[pos1_row_index[i]:neg1_row_index[i + 1]])
            cycles_single_SVID.append(single_cycle)
        min_length = len(cycles_single_SVID[0])
        for cycle in cycles_single_SVID:
            if len(cycle) < min_length:
                min_length = len(cycle)
        time_cycle_series = []
        for cycle in cycles_single_SVID:
            new_single_cycle = cycle[0:min_length]
            time_cycle_series.append(new_single_cycle)
        time_cycle_series = np.array(time_cycle_series)
        return time_cycle_series  # an nparray with dim = (num of cycles, length of a cycle) ex.(97,218)

    @staticmethod
    def spectral_transformation(time_cycle_series) -> list:
        freq_transformed_series = []
        for cycle in time_cycle_series:
            cycle = np.array(cycle)
            freq_transformed_series.append(np.abs(np.fft.rfft(cycle)) / len(cycle))
        freq_transformed_series = np.array(freq_transformed_series)
        return freq_transformed_series


if __name__ == '__main__':
    DATA_PATH = './../../../OfficialData/GetDate_20180604_RF_alarm.csv'
    SVID = 8
    STEP_NUM_COL_ID = 4
    predo, datapreprocess2csst = PreDo(), DataPreprocess2CSST()
    raw_df = predo.process(DATA_PATH)
    time_cycle_series, freq_transformed_series = datapreprocess2csst.process(raw_df, STEP_NUM_COL_ID, SVID)
    print(time_cycle_series)
    print(freq_transformed_series)