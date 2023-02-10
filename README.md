# Reproduction of "Pure Noise to the Rescue of Insufficient Data: Improving Imbalanced Classification by Training on Random Noise Images"

## Installation

Use the provided `requirements.txt` to install required packages.

```bash
pip install -r requirements.txt
```



## Run Multiple Results

To reproduce the tables in our paper, check the `bash_scripts/` directory and run the appropriate script.

- `datasets_and_ir_ratios.sh` for Table 3, 10
- `darbn_ablation.sh` for Table 4
- `augmentation.sh` for Table 5
- `balanced_cifar10.sh` for Table 6
- `resnet.sh` for Table 7, 8
- `compare_dataset.sh` for Table 10
- `search_inputnorm.sh` for Table 14
- `batch_size.sh` for Table 15
- `delta.sh` for Table 16



## Run Individual Experiments

You can also run individual experiments by directly calling `train.py`.

```
python train.py
```

By default, it uses the config values saved in `default_cifar10lt.yaml`. You can specify different default config files.

```
python train.py config_filepath=default_cifar10.yaml
```

You can also change individual config values through the command line.

```
python train.py batch_size=64
```



## Generate Figures

To generate figures for analysis, refer to the notebooks in the root directory.

- `analysis_tsne.ipynb` for Figure 2, 8
- `analysis_confusion_matrix.ipynb` for Figure 3
- `dataset_comparison.ipynb` for Figure 4
- `analysis_group_accuracy.ipynb` for Figure 6
- `analysis_histogram.ipynb` for Figure 9, 10

Each notebook downloads a model from Google Drive using a link specified in `storage.py` unless it already exists in local filepath.
