import numpy as np
import tensorflow as tf
from tensorflow.keras import losses

from STALAD.pipeline.steps.step import Step
from STALAD.pipeline.steps.pre_do import PreDo
from STALAD.pipeline.steps.data_preprocess import DataPreprocess2CSST
from STALAD.pipeline.steps.split_data import DataSplitter
from STALAD.pipeline.steps.train_sae import SAETraining


class ThresholdGenerator(Step):
    def process(self, SAE, val_data):
        threshold = self.get_threshold(SAE, val_data)
        return threshold

    @staticmethod
    def get_threshold(SAE, val_data):
        # get threshold
        decoded_val_data = SAE.predict(val_data)
        MSE_val = []
        for i in range(decoded_val_data.shape[0]):
            MSE_val.append(np.mean((val_data[i] - decoded_val_data[i]) ** 2))
        MSE_val = np.array(MSE_val)
        threshold = MSE_val.mean() + MSE_val.std(ddof=1) * 3
        return threshold


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

    predo, datapreprocess2csst, datasplitter, saetraining, thresholdgenerator = PreDo(), DataPreprocess2CSST(), DataSplitter(), SAETraining(), ThresholdGenerator()
    raw_df = predo.process(DATA_PATH)
    time_cycle_series, freq_transformed_series = datapreprocess2csst.process(raw_df, STEP_NUM_COL_ID, SVID)
    train_data, val_data, test_data = datasplitter.process(time_cycle_series, TRAIN_RATIO, VAL_RATIO)
    SAE, history = saetraining.process(train_data, val_data, MAX_LATENT_SIZE, ACT_FUNC, OPT_FUNC, LOSS_FUNC, EPOCH_NUM)
    threshold = thresholdgenerator.process(SAE, val_data)
    print('threshold:', threshold)
