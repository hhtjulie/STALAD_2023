import tensorflow as tf
from tensorflow.keras import losses

from STALAD.utils import Utils


class Logger:
    def __init__(self):
        pass

    @staticmethod
    def save_log(root_path, is_time_domain, inputs, param):
        # Save all constant settings.
        Utils.create_dir(root_path)
        exp_domain_name = Utils.get_exp_domain_name(is_time_domain)
        with open(f'{root_path}/ConstantSettings_{exp_domain_name}.txt', 'w') as text_file:
            text_file.write("Data info: ==========\n")
            text_file.write(f"DATA_PATH: {inputs['data_path']}\n")
            text_file.write(f"STEP_NUM_COL_ID: {inputs['step_num_col_idx']}\n")
            text_file.write("Properties: ==========\n")
            text_file.write(f"Time or Frequency signal: {'Time' if is_time_domain else 'Frequency'}\n")
            text_file.write(f"SVID_name: {inputs['SVID']}\n")
            text_file.write(f"train_ratio = {inputs['train_ratio']}  # the ratio of training data\n")
            text_file.write(f"EncActFunc = {param['act_func']}  # the activation function of the encoder part\n")
            text_file.write(f"DecActFunc = {param['act_func']}  # the activation function of the decoder part\n")
            text_file.write(f"optimizer = {param['opt_func']}  # the optimizer for training\n")
            text_file.write(f"max_bottleneck_length = {param['max_latent_size']}  "
                            f"# the maximum length of the bottleneck hidden layer\n")
            text_file.write(f"epoch_num = {param['epoch_num']}\n")


if __name__ == '__main__':
    DATA_PATH = './../OfficialData/GetDate_20180604_RF_alarm.csv'
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
    EPOCH_NUM = 50

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

    logger = Logger()
    logger.save_log(ROOT_PATH, True, inputs, param)
