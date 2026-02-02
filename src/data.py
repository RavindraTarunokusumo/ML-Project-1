from sklearn.datasets import fetch_openml


class DataLoader:
    @staticmethod
    def load_data_from_csv(file_path):
        import pandas as pd
        data = pd.read_csv(file_path)
        return data
    
    @staticmethod
    def load_data_from_openml(dataset_name):
        data = fetch_openml(name=dataset_name, as_frame=True)
        return data.frame