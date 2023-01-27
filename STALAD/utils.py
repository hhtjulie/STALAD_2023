import os


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def create_dir(path):
        os.makedirs(path, exist_ok=True)