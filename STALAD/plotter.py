import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses
from STALAD.pipeline.steps.get_threshold import ThresholdGenerator
from STALAD.pipeline.steps.pre_do import PreDo
from STALAD.pipeline.steps.data_preprocess import DataPreprocess2CSST
from STALAD.pipeline.steps.split_data import DataSplitter
from STALAD.pipeline.steps.train_sae import SAETraining
from STALAD.pipeline.steps.test_sae import SAETesting
from STALAD.utils import Utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Plotter:
    def __init__(self, plot_path):
        self.plot_path = plot_path
        self.figure_size = (10, 6)
        Utils.create_dir(plot_path)

    @staticmethod
    def get_plot_setting(input_size, is_time_domain: bool):
        if is_time_domain:
            # cycle coordinate assumptions: time (sec); sampling rate = 1 second; start from 1 second
            input_cycle_ticks = np.linspace(1, input_size, input_size)
        else:
            # cycle coordinate assumptions: frequency (Hz); sampling rate = 1 second (even samples); start from 0 Hz
            input_cycle_ticks = np.linspace(0.0, 0.5, input_size)
        x_label_name = 'time (sec)' if is_time_domain else 'frequency (Hz)'
        y_label_name = 'value' if is_time_domain else 'magnitude'

        plot_setting = {
            'x_label_name': x_label_name,
            'y_label_name': y_label_name,
            'input_cycle_ticks': input_cycle_ticks,
        }
        return plot_setting

    def plot_Input_vs_Decode(self, SAE, input_data, is_time_domain, input_type: str):
        input_size = input_data.shape[1]
        plot_setting = self.get_plot_setting(input_size, is_time_domain)
        exp_domain_name = Utils.get_exp_domain_name(is_time_domain)
        plt.figure(figsize=self.figure_size)
        # plot input data
        plt.subplot(121)
        plt.title(f'Original {input_type} Cycles({exp_domain_name})', fontsize=18, fontweight='bold')
        plt.xlabel(plot_setting['x_label_name'], fontsize=18)
        plt.ylabel(plot_setting['y_label_name'], fontsize=18)
        for i in range(input_data.shape[0]):
            plt.plot(plot_setting['input_cycle_ticks'], input_data[i], '-')
        # plot decoded output
        output_data = SAE.predict(input_data)
        plt.subplot(122)
        plt.title(f'Decoded {input_type} Cycles({exp_domain_name})', fontsize=18, fontweight='bold')
        plt.xlabel(plot_setting['x_label_name'], fontsize=18)
        for i in range(len(output_data)):
            plt.plot(plot_setting['input_cycle_ticks'], output_data[i], '-')

        plt.savefig(f'{self.plot_path}/{input_type}_Input_vs_Decode.png')

    def plot_difference_value(self, MSE_val_list, threshold, is_time_domain):
        x_label_name = 'cycle'
        y_label_name = 'difference value'
        exp_domain_name = Utils.get_exp_domain_name(is_time_domain)
        plt.figure(figsize=self.figure_size)
        plt.subplot(111)
        plt.title(f'Difference Values of {exp_domain_name} Cycles', fontsize=20, fontweight='bold')
        plt.xlabel(x_label_name, fontsize=18)
        plt.ylabel(y_label_name, fontsize=18)
        plt.plot(MSE_val_list, '-b')
        plt.axhline(y=threshold, color='g', linestyle='-')
        plt.savefig(f'{self.plot_path}/{exp_domain_name}_difference_value.png')

    def plot_train_val_loss(self, history):
        plt.figure(figsize=self.figure_size)
        plt.subplot(111)
        plt.plot(history['loss'], '-b', label='training loss')
        plt.plot(history['val_loss'], '-r', label='validation loss')
        plt.title('Errors', fontsize=20, fontweight='bold')
        plt.xlabel('iterations', fontsize=18)
        plt.ylabel('loss', fontsize=18)
        plt.legend(loc='best')
        plt.savefig(f'{self.plot_path}/ErrorTrendWhenTraining.png')


if __name__ == '__main__':
    DATA_PATH = './../OfficialData/GetDate_20180604_RF_alarm.csv'
    SVID = 8
    STEP_NUM_COL_ID = 4
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    MAX_LATENT_SIZE = 9
    ALPHA = 0.2
    ACT_FUNC = lambda x: tf.keras.activations.relu(x, alpha=ALPHA)
    OPT_FUNC = tf.keras.optimizers.Adam(epsilon=1e-8)
    LOSS_FUNC = losses.MeanSquaredError()
    EPOCH_NUM = 50

    predo, datapreprocess2csst, datasplitter, saetraining, thresholdgenerator, saetesting = PreDo(), DataPreprocess2CSST(), DataSplitter(), SAETraining(), ThresholdGenerator(), SAETesting()
    raw_df = predo.process(DATA_PATH)
    time_cycle_series, freq_transformed_series = datapreprocess2csst.process(raw_df, STEP_NUM_COL_ID, SVID)
    train_data, val_data, test_data = datasplitter.process(time_cycle_series, TRAIN_RATIO, VAL_RATIO)
    SAE, history = saetraining.process(train_data, val_data, MAX_LATENT_SIZE, ACT_FUNC, OPT_FUNC, LOSS_FUNC, EPOCH_NUM)
    threshold = thresholdgenerator.process(SAE, val_data)
    MSE_val_list, hypothesis_list = saetesting.process(SAE, train_data, threshold)

    plotter = Plotter(plot_path='./plot')
    plotter.plot_Input_vs_Decode(SAE, train_data, is_time_domain=True, input_type='Train')
    plotter.plot_difference_value(MSE_val_list, threshold, is_time_domain=True)
    plotter.plot_train_val_loss(history.history)
