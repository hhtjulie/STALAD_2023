import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses

from STALAD.pipeline.steps.step import Step
# from STALAD.pipeline.steps.pre_do import PreDo
# from STALAD.pipeline.steps.data_preprocess import DataPreprocess2CSST
# from STALAD.pipeline.steps.split_data import DataSplitter
# from STALAD.pipeline.steps.train_sae import SAETraining
# from STALAD.pipeline.steps.get_threshold import ThresholdGenerator


class SAETesting(Step):
    def process(self, SAE, test_data, threshold):
        MSE_val_list = self.get_difference_value(SAE, test_data)
        hypothesis_list = self.hypothesis_test(MSE_val_list, threshold)
        return MSE_val_list, hypothesis_list

    @staticmethod
    def get_difference_value(SAE, test_data) -> list:
        decoded_test_data = SAE.predict(test_data)
        MSE_val_list = []
        for i in range(decoded_test_data.shape[0]):
            MSE_val_list.append(np.mean((test_data[i] - decoded_test_data[i]) ** 2))
        # MSE_val_list = np.array(MSE_val_list)
        return MSE_val_list

    @staticmethod
    def hypothesis_test(MSE_val_list, threshold): # H0: test is normal
        hypothesis_list = []
        for i in range(len(MSE_val_list)):
            if MSE_val_list[i] > threshold:
                hypothesis_list.append(False)
            else:
                hypothesis_list.append(True)
        return hypothesis_list


if __name__ == '__main__':
    DATA_PATH = './../../../OfficialData/GetDate_20180604_RF_alarm.csv'
    SVID = 8
    STEP_NUM_COL_ID = 4
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    MAX_LATENT_SIZE = 9
    ALPHA = 0.2
    ACT_FUNC = lambda x: tf.keras.activations.relu(x, alpha=ALPHA)
    OPT_FUNC = tf.keras.optimizers.Adam(epsilon=1e-8)
    LOSS_FUNC = losses.MeanSquaredError()
    EPOCH_NUM = 200

    predo, datapreprocess2csst, datasplitter, saetraining, thresholdgenerator, saetesting = PreDo(), DataPreprocess2CSST(), DataSplitter(), SAETraining(), ThresholdGenerator(), SAETesting()
    raw_df = predo.process(DATA_PATH)
    time_cycle_series, freq_transformed_series = datapreprocess2csst.process(raw_df, STEP_NUM_COL_ID, SVID)
    train_data, val_data, test_data = datasplitter.process(time_cycle_series, TRAIN_RATIO, VAL_RATIO)
    SAE, history = saetraining.process(train_data, val_data, MAX_LATENT_SIZE, ACT_FUNC, OPT_FUNC, LOSS_FUNC, EPOCH_NUM)
    threshold = thresholdgenerator.process(SAE, val_data)
    MSE_val_list, hypothesis_list = saetesting.process(SAE, test_data, threshold)
    print(f'threshold:{threshold}' )
    print('MSE_val_list:\n', MSE_val_list)
    print('hypothesis_list\n:', hypothesis_list)
