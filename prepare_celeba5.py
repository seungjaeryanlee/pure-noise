"""
Script to download CelebA-5 dataset.
"""
import os
import zipfile
import gdown

def download_from_gdrive():
    """
    Download CelebA-5 dataset from M2m (Kim et al., 2020), which is based on the CelebA dataset (Liu et al., 2015).

    These are In-The-Wild Images from CelebA.

    https://github.com/alinlab/M2m
    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """
    # Download from Google Drive
    zip_file_name = "CelebA5_64x64.zip"
    url = "https://drive.google.com/uc?id=1l3FoP44bd9U0xtq74OLKcF0JTF_XSecF"
    gdown.download(url, zip_file_name, quiet=False)

    # Unzip
    os.makedirs("data/", exist_ok=True)
    with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
        zip_ref.extractall("data/")
    os.remove(zip_file_name)

def main():
    download_from_gdrive()

if __name__ == "__main__":
    main()
