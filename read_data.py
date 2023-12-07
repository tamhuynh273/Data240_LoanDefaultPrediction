import pandas as pd


class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.load_csv()

    def load_csv(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("Read file successfully")
        except Exception as e:
            print(f"Error loading file: {str(e)}")

    def get_data(self):
        return self.data

