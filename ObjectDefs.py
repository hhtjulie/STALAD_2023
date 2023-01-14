"""
Definition of the container for ESD.
"""

import time
from copy import deepcopy
from typing import Optional, Dict, Union, Tuple, List

import numpy as np
import pandas as pd


class ESD_container:
    raw_df: Optional[pd.DataFrame]

    def __init__(self, data_path: str = '../OfficialData/GetDate_20180604_RF_alarm.csv',
                 ElapsedTime_col_idx: int = 1, StepNumber_col_idx: int = 4) -> None:
        """
        constructor

        :param data_path:
        :param ElapsedTime_col_idx:
        :param StepNumber_col_idx:
        """
        self.data_path = data_path
        self.ElapsedTime_col_idx = ElapsedTime_col_idx
        self.StepNumber_col_idx = StepNumber_col_idx
        self.load()
        self.neg1_row_index,self.pos1_row_index = self.get_neg1_and_1_row_idx()

    def load(self):
        # load ESD according to given path
        try:
            self.raw_df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"File '{self.data_path}' does not exist. Loading ESD failed.")

    def get_neg1_and_1_row_idx(self):
        StepNumberSeries = self.raw_df.iloc[:,self.StepNumber_col_idx]
        is_first_neg1 = True  # mark the just changes of step number to -1
        neg1_row_index = []
        pos1_row_index = []
        for i in range(len(StepNumberSeries)):
            if StepNumberSeries[i] == -1:  # note when the step number is -1
                if is_first_neg1:  # enter the next cycle if the step number just changes to -1
                    neg1_row_index.append(i)
                    is_first_neg1 = False
            else:
                if not is_first_neg1:
                    pos1_row_index.append(i)
                    is_first_neg1 = True
        return neg1_row_index,pos1_row_index

    def get_trim_cycles_single_SVID(self, SVID_idx:int)->list:
        single_SVID_series = self.raw_df.iloc[:, SVID_idx]
        cycles_single_SVID = []
        for i in range(len(self.pos1_row_index)):
            if i == len(self.pos1_row_index)-1:
                single_cycle = list(single_SVID_series[self.pos1_row_index[i]:])
            else:
                single_cycle = list(single_SVID_series[self.pos1_row_index[i]:self.neg1_row_index[i + 1]])
            cycles_single_SVID.append(single_cycle)
        min_length = len(cycles_single_SVID[0])
        for cycle in cycles_single_SVID:
            if len(cycle) < min_length:
                min_length = len(cycle)

        trim_cycles_single_SVID = []
        for cycle in cycles_single_SVID:
            new_single_cycle = cycle[0:min_length]
            trim_cycles_single_SVID.append(new_single_cycle)
        return np.array(trim_cycles_single_SVID) # (# of cycles, length of a cycle) ex.(97,218)

    def get_freq_single_SVID(self, SVID_idx:int)->list:
        trim_cycles_single_SVID = self.get_trim_cycles_single_SVID(SVID_idx)
        freq_single_SVID = []
        for cycle in trim_cycles_single_SVID:
            cycle = np.array(cycle)
            freq_single_SVID.append(np.abs(np.fft.rfft(cycle))/len(cycle))
        return np.array(freq_single_SVID)


if __name__ == '__main__':
    df = ESD_container().get_trim_cycles_single_SVID(8)
    print(len(df[0]))

