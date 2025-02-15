from typing import List
from torch.utils.data import Dataset

from .image_dataset import CustomDataset

__dataset_mapper__ = {
    "HAM1000": CustomDataset,
}

def list_datasets() -> List[str]:
    """Returns a list of available dataset names.

    Returns:
        List[str]: List of dataset names as strings.

    Example:
        >>> from datasets import list_datasets
        >>> list_datasets()
        ['image', 'emodb']
    """
    return sorted(__dataset_mapper__.keys())

def get_dataset_by_name(dataset: str, *args, **kwargs) -> Dataset:
    """Returns the Dataset class using the given name and arguments.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        Dataset: The requested dataset instance.

    Example:
        >>> from datasets import get_dataset_by_name
        >>> dataset = get_dataset_by_name("emodb", root_path="./data/emodb")
        >>> type(dataset)
        <class 'datasets.audio_dataset.EmodbDataset'>
    """
    assert dataset in __dataset_mapper__, f"Dataset '{dataset}' not found in the mapper."
    return __dataset_mapper__[dataset](*args, **kwargs)