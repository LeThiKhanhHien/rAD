# Anomaly detection with semi-supervised classification based on risk estimators (rAD)

This is the official repository for "Anomaly detection with semi-supervised classification based on risk estimators" using a deep neural network.

## Reproducing Results for deep rAD

### Requirements

Execute the following steps to set up the environment

1. Go to the root directory: ```cd rAD_deep```
2. Create an environment: ```conda env create -f requirements.yml```
3. Activate the environment: ```conda activate rAD```

### Training rAD

Execute the following steps to train the rAD model

1. Go to the root directory: ```cd rAD_deep```
2. Go to the source directory: ```cd src```
3. Activate the environment: ```conda activate rAD```
4. <span style="color:red">**[Optional]**</span> Update the experimental config: The paths can be modified within the config files in ```rAD_deep/configs```
5. Run the main script with the experiment config: ```python main.py --exp_config ../configs/<config_filename>```

## Misc.

If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```
@article{hien2023anomaly,
  title={Anomaly detection with semi-supervised classification based on risk estimators},
  author={Hien, Le Thi Khanh and Patra, Sukanya and Taieb, Souhaib Ben},
  journal={arXiv preprint arXiv:2309.00379},
  year={2023}
}
```
