# EIR-auto-GP

<p align="center">
  <img src="docs/source/_static/img/eir-auto-gp-logo.svg" alt="Eir Auto GP Logo">
</p>

<p align="center">
  <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-APGL-5B2D5B.svg" />
  </a>
  
  <a href="https://www.python.org/downloads/" alt="Python">
        <img src="https://img.shields.io/badge/python-3.10-blue.svg" />
  </a>
  
  <a href="https://pypi.org/project/eir-auto-gp/" alt="Python">
        <img src="https://img.shields.io/pypi/v/eir-auto-gp.svg" />
  </a>
  
  
  <a href="https://codecov.io/gh/arnor-sigurdsson/EIR-auto-GP" > 
        <img src="https://codecov.io/gh/arnor-sigurdsson/EIR-auto-GP/branch/master/graph/badge.svg?token=PODL2J83Y0"/> 
  </a>
  
  <a href='https://eir-auto-gp.readthedocs.io'>
    <img src='https://readthedocs.org/projects/eir-auto-gp/badge/?version=latest' alt='Documentation Status' />
  </a>
      
  
</p>

`EIR-auto-GP`: Automated genomic prediction (GP) using deep learning models with EIR.

**WARNING**: This project is in alpha phase. Expect backwards incompatible changes and API changes.

## Overview

EIR-auto-GP is a comprehensive framework for genomic prediction (GP) tasks, built on top of the [EIR](https://github.com/arnor-sigurdsson/EIR) deep learning framework. EIR-auto-GP streamlines the process of preparing data, training, and evaluating models on genomic data, automating much of the process from raw input files to results analysis. Key features include:

- Support for `.bed/.bim/.fam` PLINK files as input data.
- Automated data processing and train/test splitting.
- Takes care of launching a configurable number of deep learning training runs.
- SNP-based feature selection based on GWAS, deep learning-based attributions, and a combination of both.
- Ensemble prediction from multiple training runs.
- Analysis and visualization of results.

## Installation

First, ensure that [plink2](https://www.cog-genomics.org/plink/2.0/) is installed and available in your `PATH`. 

Then, install `EIR-auto-GP` using `pip`:

`pip install eir-auto-gp`

## Usage

Please refer to the [Documentation](https://eir-auto-gp.readthedocs.io/en/latest/) for examples and information.

## Workflow

1. Data processing: EIR-auto-GP processes the input `.bed/.bim/.fam` PLINK files and `.csv` label file, preparing the data for model training and evaluation.
2. Train/test split: The processed data is automatically split into training and testing sets, with the option of manually specifying splits.
3. Training: Configurable number of training runs are set up and executed using EIR's deep learning models.
4. SNP feature selection: GWAS based feature selection, deep learning-based feature selection with Bayesian optimization, and mixed strategies are supported.
5. Test set prediction: Predictions are made on the test set using all training run folds.
6. Ensemble prediction: An ensemble prediction is created from the individual predictions.
7. Results analysis: Performance metrics, visualizations, and analysis are generated to assess the model's performance.

## Citation

If you use `EIR-auto-GP` in a scientific publication, we would appreciate if you could use the following citation:

```
@article{sigurdsson2021deep,
  title={Deep integrative models for large-scale human genomics},
  author={Sigurdsson, Arnor Ingi and Westergaard, David and Winther, Ole and Lund, Ole and Brunak, S{\o}ren and Vilhjalmsson, Bjarni J and Rasmussen, Simon},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
