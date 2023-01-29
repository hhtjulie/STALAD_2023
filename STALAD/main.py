import tensorflow as tf
from tensorflow.keras import layers, losses

from STALAD.pipeline.steps.pre_do import PreDo
from STALAD.pipeline.steps.data_preprocess import DataPreprocess2CSST
from STALAD.pipeline.steps.split_data import DataSplitter
from STALAD.pipeline.steps.test_sae import SAETesting
from STALAD.pipeline.steps.train_sae import SAETraining
from STALAD.pipeline.steps.get_threshold import ThresholdGenerator
from STALAD.plotter import Plotter
from STALAD.logger import Logger

DATA_PATH = './../OfficialData/GetDate_20180604_RF_alarm.csv'
ROOT_PATH = './EXP'
SVID = 8
STEP_NUM_COL_ID = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
MAX_LATENT_SIZE = 9
ACT_FUNC = lambda x: tf.keras.activations.relu(x, alpha=0.2)
OPT_FUNC = tf.keras.optimizers.Adam(epsilon=1e-8)
LOSS_FUNC = losses.MeanSquaredError()
EPOCH_NUM = 200


def main():
    inputs = {
        'data_path': DATA_PATH,
        'SVID': SVID,
        'step_num_col_idx': STEP_NUM_COL_ID,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
    }
    param = {
        'max_latent_size': MAX_LATENT_SIZE,
        'act_func': ACT_FUNC,
        'opt_func': OPT_FUNC,
        'loss_func': LOSS_FUNC,
        'epoch_num': EPOCH_NUM,
    }
    pre_do, data_preprocess2csst, data_splitter, SAE_training, threshold_generator, SAE_testing = PreDo(), DataPreprocess2CSST(), DataSplitter(), SAETraining(ROOT_PATH), ThresholdGenerator(), SAETesting()

    raw_df = pre_do.process(inputs['data_path'])
    time_cycle_series, freq_transformed_series = data_preprocess2csst.process(raw_df, inputs['step_num_col_idx'],
                                                                              inputs['SVID'])
    # time
    time_train_data, time_val_data, time_test_data = data_splitter.process(time_cycle_series, inputs['train_ratio'],
                                                                           inputs['val_ratio'])
    time_SAE, time_history = SAE_training.process(time_train_data, time_val_data, param['max_latent_size'], param['act_func'],
                                    param['opt_func'], param['loss_func'], param['epoch_num'])
    time_threshold = threshold_generator.process(time_SAE, time_val_data)
    time_MSE_val_list, hypothesis_list = SAE_testing.process(time_SAE, time_test_data, time_threshold)
    # plot time
    plotter = Plotter(plot_path='./plot_time')
    plotter.plot_Input_vs_Decode(time_SAE, time_train_data, is_time_domain=True, input_type='Train')
    plotter.plot_difference_value(time_MSE_val_list, time_threshold, is_time_domain=True)
    plotter.plot_train_val_loss(time_history.history)
    # save setting log
    logger = Logger()
    logger.save_log(ROOT_PATH, True, inputs, param)

    # # frequency
    # freq_train_data, freq_val_data, freq_test_data = data_splitter.process(freq_transformed_series,
    #                                                                        inputs['train_ratio'], inputs['val_ratio'])
    # freq_SAE, freq_history = SAE_training.process(freq_train_data, freq_val_data, param['max_latent_size'], param['act_func'],
    #                                 param['opt_func'], param['loss_func'], param['epoch_num'])
    # freq_threshold = threshold_generator.process(freq_SAE, freq_val_data)
    # freq_MSE_val_list, hypothesis_list = SAE_testing.process(freq_SAE, freq_test_data, freq_threshold)
    # # plot frequency
    # plotter = Plotter(plot_path='./plot_freq')
    # plotter.plot_Input_vs_Decode(freq_SAE, freq_train_data, is_time_domain=False, input_type='Train')
    # plotter.plot_difference_value(freq_MSE_val_list, freq_threshold, is_time_domain=True)
    # plotter.plot_train_val_loss(freq_history.history)


if __name__ == '__main__':
    main()
