from datasets import load_dataset
class DataTransformer:

    def __init__(self, root_path):
        self.root_path = root_path
        self.data = load_dataset("huggan/pokemon")

    def transform_data(self):
        pass
