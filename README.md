# ASU-Net: Attention to Scale with U-Net for Semantic Segmentation
This is a TensorFlow implementation of our ASU-Net. We train Multi-scale U-Net model first, and then the ASU-Net.
For more details, check our paper. (Link coming soon)

1. [Prerequisites](#prerequisites)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Testing](#testing)
5. [Visualizing](#visualizing)
6. [CRF](#crf)
7. [BFSCORE](#bfscore)

## Directory Structure

```bash
└── __init__.py
└── .gitignore
└── ASUNet.py
└── BatchDatasetReader.py
└── bfscore.py
└── CalculateUtil.py
└── denseCRF.py
└── EvalMetrics.py
└── function_definitions.py
└── LICENSE
└── read_10k_data.py
└── read_CFPD_data.py
└── read_LIP_data.py
└── README.md
└── requirements.txt
└── TensorflowUtils.py
└── test_human.py
└── UNetMSc.py

```

## Prerequisites
 - For required packages installation, run `pip install -r requirements.txt`
 - pydensecrf installation in windows with conda: `conda install -c conda-forge pydensecrf`. For linux, use pip: `pip install pydensecrf`.
 - Check dataset directory in `read_dataset` function of corresponding data reading script, for example, for LIP dataset, check paths in `read_LIP_data.py` and modify as necessary.

## Dataset
 - Right now, there are dataset supports for 3 datasets. Set your directory path in the corresponding dataset reader script.
 - [CFPD](https://github.com/hrsma2i/dataset-CFPD) (For preparing CFPD dataset, you can visit here: https://github.com/minar09/dataset-CFPD-windows)
 - [LIP](http://www.sysu-hcp.net/lip/)
 - 10k (Fashion)
 - If you want to use your own dataset, please create your dataset reader. (Check `read_CFPD_data.py` for example, on how to put directory and stuff)

## Training
 - To train model simply execute `python UNetMSc.py` and then `python ASUNet.py`
 - You can add training flag as well, for example: `python UNetMSc.py --mode=train` and `python ASUNet.py --mode=train`
 - `debug` flag can be set during training to add information regarding activations, gradients, variables etc.
 - Set your hyper-parameters in the corresponding model script

## Testing
 - To test and evaluate results use flag `--mode=test`, e.g., `python ASUNet.py --mode=test`
 - After testing and evaluation is complete, final results will be printed in the console, and the corresponding files will be saved in the "logs" directory.
 - Set your hyper-parameters in the corresponding model script

## Visualizing
 - To visualize results for a random batch of images use flag `--mode=visualize`
 - Set your hyper-parameters in the corresponding model script

## CRF
 - Running testing will apply CRF by default.
 - If you want to run standalone, run `python denseCRF.py`, after setting your paths.

## BFSCORE
 - Run `python bfscore.py`, after setting your paths.
 - For more details, visit https://github.com/minar09/bfscore_python
