import os


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def create_dir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def dir_exist(dir_path):
        return os.path.exists(dir_path) and os.path.getsize(dir_path) > 0

    @staticmethod
    def get_exp_domain_name(is_time_domain):
        if is_time_domain:
            exp_domain_name = 'Time'
        else:
            exp_domain_name = 'Frequency'
        return exp_domain_name
