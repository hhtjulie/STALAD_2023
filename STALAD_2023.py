import time
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import math
import os
from ObjectDefs import ESD_container
###
import tensorflow as tf
from tensorflow.keras import layers, losses

print('hi')

class STALAD:
    def __init__(self, train_data=None, val_data=None, test_data=None, max_latent_size: int = 9):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.ActFunc = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.Opt = tf.keras.optimizers.Adam(epsilon=1e-8)  # specify the used optimizer
        self.input_size = self.train_data.shape[1]  # input data length per wafer
        self.layer_size_list = self.get_layer_size(max_latent_size)
        self.SAE = None
        self.build_SAE()
        self.root_path = f'{datetime.date.today().strftime("%Y%m%d")}_STALAD'
        self.plot_path = f'{self.root_path}/STALAD_plot'

    def get_layer_size(self, max_latent_size):
        encoder_layer_size_list = []
        layer_size = self.input_size
        while layer_size >= max_latent_size:
            encoder_layer_size_list.append(layer_size)
            layer_size = math.ceil(layer_size / 2)
        decoder_layer_size_list = encoder_layer_size_list[::-1]
        encoder_layer_size_list.append(layer_size)
        layer_size_list = encoder_layer_size_list+decoder_layer_size_list
        return layer_size_list

    def build_SAE(self):
        # encoder
        self.SAE = tf.keras.Sequential()
        self.SAE.add(layers.Dense(self.layer_size_list[1], activation=self.ActFunc,
                                  input_shape=(self.layer_size_list[0],)))
        for layer_size in self.layer_size_list[2:]:
            self.SAE.add(layers.Dense(layer_size, activation=self.ActFunc))
        self.SAE.compile(optimizer=self.Opt, loss=losses.MeanSquaredError())
        self.SAE.summary()

    def train_SAE(self, epoch_num: int = 200, is_save=True):
        checkpoint_filepath = '.\\log'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True, save_freq='epoch'
        )
        decoded = self.SAE.fit(self.train_data, self.train_data, validation_data=(self.val_data, self.val_data),
                               epochs=epoch_num, batch_size=len(self.train_data), shuffle=True,
                               callbacks=[model_checkpoint_callback])

        if is_save:
            tf.keras.utils.plot_model(self.SAE, to_file='./model.png', show_shapes=True)
            self.SAE.save("my_model")
        return decoded

    def get_threshold(self):
        # get threshold
        decoded_val_data = self.SAE.predict(self.val_data)
        MSE_val = []
        for i in range(decoded_val_data.shape[0]):
            MSE_val.append(np.mean((self.val_data[i] - decoded_val_data[i]) ** 2))
        MSE_val = np.array(MSE_val)
        threshold = MSE_val.mean() + MSE_val.std(ddof=1) * 3
        return threshold

    def get_decoded_data(self, input_data):
        decoded_test_data = self.SAE.predict(input_data)
        return decoded_test_data

    def get_difference_value(self):
        decoded_test_data = self.get_decoded_data(self.test_data)
        MSE_val = []
        for i in range(decoded_test_data.shape[0]):
            MSE_val.append(np.mean((self.test_data[i] - decoded_test_data[i]) ** 2))
        MSE_val = np.array(MSE_val)
        return MSE_val

    def plot_and_save(self, is_time_domain: bool = True):
        os.makedirs(self.plot_path)
        figure_size = (10, 6)  # the figure size of results
        input_cycle_ticks = None
        if is_time_domain:
            # cycle coordinate assumptions: time (sec); sampling rate = 1 second; start from 1 second
            input_cycle_ticks = np.linspace(1, self.input_size, self.input_size)
        else:
            # cycle coordinate assumptions: frequency (Hz); sampling rate = 1 second (even samples); start from 0 Hz
            input_cycle_ticks = np.linspace(0.0, 0.5, self.input_size)
        x_label_name = 'time (sec)' if is_time_domain else 'frequency (Hz)'
        y_label_name = 'value' if is_time_domain else 'magnitude'
        output_data = self.get_decoded_test_data()
        # plot decoded training cycles

        def plot_Input_vs_Decode(input_data, input_type: str = 'Train'):  # input can be val_data/train_data
            plt.figure(figsize=figure_size)
            plt.subplot(121)
            plt.title(f'Original {input_type} Cycles', fontsize=20, fontweight='bold')
            plt.xlabel(x_label_name, fontsize=18)
            plt.ylabel(y_label_name, fontsize=18)
            for i in range(input_data.shape[0]):
                plt.plot(input_cycle_ticks, self.train_data[i], '-')
            plt.subplot(122)
            plt.title(f'Decoded {input_type} Cycles', fontsize=20, fontweight='bold')
            plt.xlabel(x_label_name, fontsize=18)
            for i in range(len(output_data)):
                plt.plot(input_cycle_ticks, output_data[i], '-')
            plt.savefig(f'{self.plot_path}/{input_type}_Input_vs_Decode.png')

        def plot_train_val_loss():
            plt.figure(figsize=figure_size)
            plt.subplot(111)
            plt.plot(error_points, all_train_error_log10, '-b', label='training error')
            plt.plot(error_points, all_validate_error_log10, '-r', label='validation error')
            plt.title('Errors', fontsize=24, fontweight='bold')
            plt.xlabel('iterations', fontsize=18)
            plt.ylabel('log10(loss)', fontsize=18)
            plt.legend(loc='best')
            figure_filename = f'{saver_dir}/ErrorTrendWhenTraining.png'
            plt.savefig(figure_filename)

def get_data(data_path: str = './OfficialData/GetDate_20180604_RF_alarm.csv', SVID: int = 8,
             is_time_domain: bool = True, train_ratio: float = 0.7, val_ratio: float = 0.1):
    if is_time_domain:
        input_data = ESD_container(data_path).get_trim_cycles_single_SVID(SVID)
    else:
        input_data = ESD_container(data_path).get_freq_single_SVID(SVID)
    input_wafer_num = input_data.shape[0]
    train_num = round(train_ratio * input_wafer_num)
    val_num = round(val_ratio * input_wafer_num)
    train_data = input_data[:train_num]
    val_data = input_data[train_num:train_num + val_num]
    test_data = input_data[(train_num + val_num):]
    return train_data, val_data, test_data


if __name__ == '__main__':
    df = ESD_container().get_trim_cycles_single_SVID(8)
    train_data, val_data, test_data = get_data()
    print(type(train_data))
    print(type(val_data))
    print(type(test_data))
    print(isinstance(train_data,np.ndarray))
    STALAD = STALAD(train_data, val_data, test_data)
    print(STALAD.layer_size_list)
    print(STALAD.train_data)
    STALAD.train_SAE(epoch_num=10)
    a = STALAD.get_threshold()
    print(a)
    vec = STALAD.get_difference_value()
    print(vec)
