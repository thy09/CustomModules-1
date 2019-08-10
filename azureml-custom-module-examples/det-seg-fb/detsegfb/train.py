import os
import json
import fire
from shutil import copyfile


def test(model_folder, config_filename, out_model_folder):
    os.makedirs(out_model_folder, exist_ok=True)
    src_path = os.path.join(model_folder, config_filename)
    tgt_path = os.path.join(out_model_folder, config_filename)
    copyfile(src_path, tgt_path)

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        'Id': 'ILearnerDotNet',
        'Name': 'ILearner .NET file',
        'ShortName': 'Model',
        'Description': 'A .NET serialized ILearner',
        'IsDirectory': False,
        'Owner': 'Microsoft Corporation',
        'FileExtension': 'ilearner',
        'ContentType': 'application/octet-stream',
        'AllowUpload': False,
        'AllowPromotion': False,
        'AllowModelPromotion': True,
        'AuxiliaryFileExtension': None,
        'AuxiliaryContentType': None
    }
    with open(os.path.join(out_model_folder, 'data_type.json'), 'w') as f:
        json.dump(dct, f)
    # Dump data.ilearner as a work around until data type design
    visualization = os.path.join(out_model_folder, 'data.ilearner')
    with open(visualization, 'w') as file:
        file.writelines('{}')
    print('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(test)
