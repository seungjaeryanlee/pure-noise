"""
Script to download and format data to consume as PyTorch dataset.
"""
import json
import os
from types import SimpleNamespace
import zipfile

import gdown

from convert_from_tfrecords import convert_from_tfrecords


TFRECORDS_DIRPATH = "data/tfrecords/"
JSON_DIRPATH = "data/json/"


def download_tfrecords_from_gdrive():
    """
    Download .tfrecords file provided by Cui et al. (2019)
    
    Will create a `tfrecords.zip` file and `data/tfrecords/` directory.
    
    https://github.com/richardaecn/class-balanced-loss/blob/master/README.md#datasets

    """
    # Download from Google Drive
    url = "https://drive.google.com/uc?id=1NY3lWYRfsTWfsjFPxJUlPumy-WFeD7zK"
    output = "tfrecords.zip"
    gdown.download(url, output, quiet=False)

    # Unzip to `tfrecords/data/`
    with zipfile.ZipFile("tfrecords.zip", "r") as zip_ref:
        zip_ref.extractall("tfrecords")

    # Rename directories and remove zip file
    os.makedirs("data/", exist_ok=True)
    os.rename("tfrecords/data/", TFRECORDS_DIRPATH)
    os.remove("tfrecords.zip")
    os.rmdir("tfrecords/")


def convert_tfrecords_to_json():
    """
    Convert .tfrecords file to JSON format with script by Lio et al. (2022)

    https://github.com/bazinga699/ncl#prepare-datasets

    """

    cifar10_im50 = {
        'dir': 'cifar-10-data-im-0.02',
        'json': 'cifar10_imbalance50',
        'class': 10,
    }
    cifar10_im100 = {
        'dir': 'cifar-10-data-im-0.01',
        'json': 'cifar10_imbalance100',
        'class': 10,
    }
    cifar100_im50 = {
        'dir': 'cifar-100-data-im-0.02',
        'json': 'cifar100_imbalance50',
        'class':100,
    }
    cifar100_im100 = {
        'dir': 'cifar-100-data-im-0.01',
        'json': 'cifar100_imbalance100',
        'class': 100,
    }

    modes = ['train', 'valid']
    variants = [
        cifar10_im50,
        cifar10_im100,
        cifar100_im50,
        cifar100_im100,
    ]
    for mode in modes:
        for variant in variants:
            convert_from_tfrecords(
                TFRECORDS_DIRPATH,
                variant['dir'],
                variant['class'],
                mode,
                JSON_DIRPATH,
                variant['json'],
            )


def main():
    download_tfrecords_from_gdrive()
    convert_tfrecords_to_json()


if __name__ == "__main__":
    main()
