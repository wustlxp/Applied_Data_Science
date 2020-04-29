# Predict Future Sales
##### Final Project for Applied Data Science for Practitioners

Predict Future Sales
Kaggle: https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data

The main goal of this kaggle competition is to predict the monthly sales for the next month based on the historical daily sales of each store and each product for 34 months.So it's a time-series problem.

#### Project difficulties:

Data is scattered in multiple excel files
Predictors are not suitable for direct use and require more feature engineering
The information brought by the time series should be fully exploited
Evaluation method: Submissions are evaluated by root mean squared error (RMSE). True target values are clipped into [0,20] range.


    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. project and data analysis, data cleaning, visualization, and hyperparameters tuning
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │   │
    │   ├── Dictionaries_origin.ipynb  <- Dictionaries for original data set
    │   └── Dictionaries_model.ipynb   <- Dictionaries for columns appears during the procedure
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── functions      <- Scripts of predifined functions for pipeline
    │   │   └── functions.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions for test set and Kaggle test set
    │   │   └── pipeline.ipynb
    │   │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org