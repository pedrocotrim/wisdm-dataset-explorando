# WISDM Dataset Biometrics and HAR

This repository contains code written for my Final Project for the Bachelor's degree in Computer Sciences, in the Institute of Computing of Universidade Federal Fluminense (UFF). The project uses the smartphone and smartwatch data from the WISDM dataset and is inspired by [this paper](https://doi.org/10.1109/ACCESS.2019.2940729) published by the researchers responsible for creating the dataset. In addition to the biometrics authentication and person classification, this project also uses the data for Human Activity Recognition, using the same machine learning algorithms with slight modifications.

## Prerequisites
To ensure the correct execution of the modules, it is necessary to download the dataset from the [UCI Machine Learning Repository](https://doi.org/10.24432/C5HK59) and extract the contents of the innermost `wisdm-dataset` folder to the root of this project.

All external modules required to run this project are present in the `requirements.txt` file and can be installed by running the following command from the root of this project. 
```bash
pip install -r requirements.txt
```


## Organization

The dataset processing and each of the classification routines are separated in a module each. To run the algorithm, import the module and call the `main()` function, with the boolean `voting` param to enable voting, that is disabled by default.
