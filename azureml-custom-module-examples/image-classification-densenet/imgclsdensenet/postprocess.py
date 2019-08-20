import os
import json
import fire
import pandas as pd
import pyarrow.parquet as pq  # noqa: F401
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.common.datatable.data_table import DataTable

from .smt_fake import smt_fake_file


class Postprocess:
    def __init__(self, file_path, meta={}):
        file_name = os.path.join(file_path, 'index_to_label.json')

        self.classes = []
        with open(file_name) as f:
            self.classes = json.load(f)

    def run(self, input, meta=None):
        my_list = []

        for i in range(input.shape[0]):
            index = input.iloc[i]['index']
            probability_string = input.iloc[i]['probability']
            result = [self.classes[index], probability_string]
            my_list.append(result)

        df = pd.DataFrame(my_list, columns=['label', 'probability'])
        return df

    def evaluate(self, data_path='test_data', save_path='outputs'):
        os.makedirs(save_path, exist_ok=True)
        input = pd.read_parquet(os.path.join(data_path, 'data.dataset.parquet'), engine='pyarrow')
        df = self.run(input)
        dt = DataTable(df)
        OutputHandler.handle_output(data=dt, file_path=save_path,
                                    file_name='data.dataset.parquet', data_type=DataTypes.DATASET)


def test(file_path='script', data_path='script/outputs2', save_path='script/outputs3'):
    postprocess = Postprocess(file_path)
    postprocess.evaluate(data_path=data_path, save_path=save_path)

    smt_fake_file(save_path)


if __name__ == '__main__':
    fire.Fire(test)
