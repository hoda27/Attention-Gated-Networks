import json

from dataio.loader.pancreas_seg import PanSegNetDataset
from dataio.loader.test_dataset import TestDataset


def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'pancreas_seg': PanSegNetDataset,
        'test_sax': TestDataset,
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
