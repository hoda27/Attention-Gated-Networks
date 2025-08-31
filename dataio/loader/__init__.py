import json

from dataio.loader.ukbb_dataset import UKBBDataset
from dataio.loader.test_dataset import TestDataset


def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'acdc_sax': UKBBDataset,
        'test_sax': TestDataset,
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
