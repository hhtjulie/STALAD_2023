from STALAD.pipline.steps.read_csv_file import read_csv_file


DATA_PATH = './OfficialData/GetDate_20180604_RF_alarm.csv'
MAX_LATENT_SIZE = 9
SVID = 8

def main():
    inputs = {
        'data_path': DATA_PATH,
        'SVID': SVID,
        'max_latent_size': MAX_LATENT_SIZE,
    }

    steps = [
        read_csv_file()
    ]