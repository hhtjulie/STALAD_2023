import math
import tensorflow as tf
from tensorflow.keras import layers, losses

from STALAD.pipeline.steps.step import Step
from STALAD.pipeline.steps.pre_do import PreDo
from STALAD.pipeline.steps.data_preprocess import DataPreprocess2CSST
from STALAD.pipeline.steps.split_data import DataSplitter
from STALAD.utils import Utils


class SAETraining(Step):
    def __init__(self, root_path):
        self.root_path = root_path
        Utils.create_dir(root_path)

    def process(self, train_data, val_data, max_latent_size, act_func, opt_func, loss_func, epoch_num):
        input_size = train_data.shape[1]
        layer_size_list = self.get_layer_size(input_size, max_latent_size)
        SAE = self.build_SAE(layer_size_list, act_func, opt_func, loss_func)
        train_history = self.train_SAE(SAE, train_data, val_data, epoch_num, is_save=True)
        return SAE, train_history

    @staticmethod
    def get_layer_size(input_size, max_latent_size):
        encoder_layer_size_list = []
        layer_size = input_size
        while layer_size >= max_latent_size:
            encoder_layer_size_list.append(layer_size)
            layer_size = math.ceil(layer_size / 2)
        decoder_layer_size_list = encoder_layer_size_list[::-1]
        encoder_layer_size_list.append(layer_size)
        layer_size_list = encoder_layer_size_list + decoder_layer_size_list
        return layer_size_list

    @staticmethod
    def build_SAE(layer_size_list, act_func, opt_func, loss_func):
        # encoder
        SAE = tf.keras.Sequential()
        SAE.add(layers.Dense(layer_size_list[1], activation=act_func, input_shape=(layer_size_list[0],)))
        for layer_size in layer_size_list[2:]:
            SAE.add(layers.Dense(layer_size, activation=act_func))
        SAE.compile(optimizer=opt_func, loss=loss_func)
        SAE.summary()
        return SAE

    def train_SAE(self, SAE, train_data, val_data, epoch_num: int, is_save=True):
        checkpoint_filepath = f'{self.root_path}/checkpoint_log'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True, save_freq=500
        )
        train_history = SAE.fit(train_data, train_data, validation_data=(val_data, val_data),
                                epochs=epoch_num, batch_size=len(train_data), shuffle=True,
                                callbacks=[model_checkpoint_callback])
        if is_save:
            tf.keras.utils.plot_model(SAE, to_file=f'{self.root_path}/model.png', show_shapes=True)
            SAE.save(f"{self.root_path}/my_model")
        return train_history


if __name__ == '__main__':
    DATA_PATH = './../../../OfficialData/GetDate_20180604_RF_alarm.csv'
    ROOT_PATH = './EXP'
    SVID = 8
    STEP_NUM_COL_ID = 4
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    MAX_LATENT_SIZE = 9
    ALPHA = 0.2
    ACT_FUNC = lambda x: tf.keras.activations.relu(x, alpha=ALPHA)
    OPT_FUNC = tf.keras.optimizers.Adam(epsilon=1e-8)
    LOSS_FUNC = losses.MeanSquaredError()
    EPOCH_NUM = 100

    predo, datapreprocess2csst, datasplitter, saetraining = PreDo(), DataPreprocess2CSST(), DataSplitter(), SAETraining(
        ROOT_PATH)
    raw_df = predo.process(DATA_PATH)
    time_cycle_series, freq_transformed_series = datapreprocess2csst.process(raw_df, STEP_NUM_COL_ID, SVID)
    train_data, val_data, test_data = datasplitter.process(time_cycle_series, TRAIN_RATIO, VAL_RATIO)
    SAE, train_history = saetraining.process(train_data, val_data, MAX_LATENT_SIZE, ACT_FUNC, OPT_FUNC, LOSS_FUNC,
                                             EPOCH_NUM)
    print(train_history.history)
