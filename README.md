# HUST Recommender System On Steam

Mini project submission for Machine Learning - IT3190E, by Group 10.

| Group member     | Student ID |
| :--------------- | :--------: |
| Doi Sy Thang     |  20225528  |
| Ngo Duy Dat      |  20225480  |
| Ta Ho Thanh Dat  |  20225482  |
| Nguyen Nhat Minh |  20225510  |
| Nguyen Minh Quan |  20225520  |

Guided by Prof. Than Quang Khoat.

This project is marked as final as of **31/05/2024**. No further contribution is accepted.

## Description

### Abstract

Recommender systems represent a specialized field within machine learning, characterized by its unique attributes and evaluation methods. In this report, we aim to present fundamental approaches and initial perspectives on constructing a recommender system. We will employ methods specifically tailored to address this problem, such as the Content-Based Model (CB), which focuses on comparing the similarity between users or items, inspired by the human tendency for imitation. Additionally, we will explore basic machine learning models, such as the linear model, to experiment with recommender systems. Furthermore, we propose the use of Collaborative Filtering, featuring two main models: the Neighborhood-based model (NB) and the Latent-factor model (LF). While the NB model leverages user-user and item-item similarities based on ratings rather than attributes, as in the CB model, the Latent-factor model will find latent features based on observations, some approaches are implemented such as matrix factorization models, factorization machines, and certain deep learning techniques.

Our project will utilize game data from Steam, instead of widely known datasets like MovieLens or Netflix films. This dataset includes features that can be considered labels, such as "is recommended"(implicit feedback) features and "hours" features. We will propose a strategy to combine these features to generate the most reasonable ratings possible, which we term explicit feedback. Furthermore, for simplicity, our project will not address the cold-start problem but will focus on resolving issues using a warm-start approach.

### Project Structure

    .
    ├── data                            # Main database (not really)
    ├── CF                              # Collaborative Filtering models
    |   ├── Latent_Factor_Model
    |   |   ├── Deep_based              # Model-based CF using Deep Learing
    |   |   └── MatrixFactorization     # Model-based CF using Matrix Factorization
    |   └── Neighborhood_Based_Model    # Memory-based CF using top k neighbours from similarity
    ├── SourceCode
    |   ├── CB                          # Content Based model
    |   └── UI                          # App UI
    |       └── start.py
    ├── rating_trans.py                 # Generate explicit ratings data
    ├── README.md
    └── requirement.txt

## Setting up

### Requirement

The requirement for this project is listed in [requirement.txt](requirement.txt), please use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install -r requirement.txt
```

### Dataset

Due to the the disorganized nature of our project, the dataset is for each models must be downloaded to its corresponding destination as listed below (despite having a separate data folder). The dataset can be found [here](https://husteduvn-my.sharepoint.com/:f:/g/personal/dat_tht225482_sis_hust_edu_vn/Ev0Vpc1zpzxMvNQ0ZLgRJI0BrLqyQ0uWEBUq8udKeVZGjA?e=1x14QL).

`Steam Recommendation System`

| File                  | Destination                                   | Description                                                                      |
| :-------------------- | :-------------------------------------------- | :------------------------------------------------------------------------------- |
| kaggle.zip            | ./data                                        | The original dataset that we used for this project, citation is provided.        |
| data.zip              | ./SourceCode/CB                               | Dataset used for Content-based model and UI.                                     |
| data_process.zip      | ./CF/Latent_Factor_Model/Deep_based           | Dataset used for Latent Factor Colaborative Filtering with Deep learning.        |
| preprocessed_data.zip | ./CF/Latent_Factor_Model/MatrixFactorization/ | Dataset used for Latent Factor Colaborative Filtering with Matrix Factorization. |

## Usage

### [LF Deep-based](CF/Latent_Factor_Model)
    .
    |__ predict_demo.py         # implement predict with three models, NCF (implicit), NeuMF (implicit), NCF_Features (explicit)
    |__ NCF.py                  # structure of NCF 
    |__ NeuMF.py                # structure of NeuMF
    |__ NCF_Feature.py          # structure of NCF_Feature
    |__ Data_processing.ipynb   # implement data process for deep, note that data after process saving in pickle file because of having vector in data frame, and some 
                                 feature selection explains in process_data of matrix_factorization 
    |__ NCF.ipynb               # implement training NCF model 
    |__ NeuMF.ipynb             # implement training NeuMF model 
    |__ NCF_Feature.ipynb       # implemnt training NCF_feature 
### [LF Matrix Factorization](CF/Latent_Factor_Model/MatrixFactorization)
    .
    |__ EDA.py                 # analysis data and explain why to transform from implicit feedback rating to explicit feedback 
    |__ data_processing.py     # process data for training model of matrix_factorization
    
    |__ MFnoSideinfo __|-- BRR_MF      # implement model matrix factorization BRR
                       |-- other file # implement 4 simple algorithms of matrix factorization 
                       
    |__ MFSideinfo   __|-- FactorizationMachine folder # implement the fm model 
                       |-- CMF.ipynb # implement collective matrix factorization model,  with file utility_cmf.py is  a helper modulus filefile
### [CF Neighborhood-based](CF/Neighborhood_Based_Model)

This folder contain our implementation of memory-based Content Filtering system using neighborhood method with users / items similarity.

    .
    ├── preprocessing.py                    # Preprocessor class used for generating ratings data from source (./data/kaggle) and data splitter
    ├── mapping.py                          # Mapper class using json data to map users and items
    ├── similarity.py                       # Similarity class containing methods to calculate similarity between users or items.
    ├── topk_cf.py                          # TopKNeighborCF system itself
    ├── TopKCF_Testing.ipynb                # Notebook for testing TopKNeighborCF
    └── TopKCF_Demo.ipynb                   # Notebook containing the full module usage

Initially, this system is implemented as a module; however, it is not used for the final UI demonstration. It's usage can be found in [TopKCF_Demo.ipynb](CF/Neighborhood_Based_Model/TopKCF_Demo.ipynb).

### [Content-based and UI](SourceCode)
