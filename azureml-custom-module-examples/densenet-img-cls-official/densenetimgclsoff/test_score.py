from azureml.designer.model.io import load_generic_model
from azureml.studio.core.io.image_directory import ImageDirectory


if __name__ == '__main__':
    # Test inference
    print("Testing inference.")
    loaded_generic_model = load_generic_model(
        path='/mnt/chjinche/projects/saved_model')
    loader_dir = ImageDirectory.load('/mnt/chjinche/data/test_data/')
    result_dfd = loaded_generic_model.predict(loader_dir)
    print(f'result_dfd: {result_dfd}')
