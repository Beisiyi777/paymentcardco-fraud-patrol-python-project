# About PaymentCardCo Fraud Patrol Python Project

This repository contains the coding documents of my final capstone project for the Springboard Data Analytics Program.

The goal of this supervised learning project was to use Python to design an algorithm for a payment card company to identify fraudsters and encode the algorithm logic into a Patrol class that takes in a transaction ID as a string, computes the prediction of whether the transaction is fraudulent, and then returns an action ("PASS" if no action or "LOCK" to lock the transaction and the associated user's account).

Original data files and instructions were obtained from Zubal', Andrej. (2022 or earlier).  _Revolut Assignment - Customer Fraud_ [Online forum post]. Kaggle. [https://www.kaggle.com/datasets/andrejzuba/revolutassignment](https://www.kaggle.com/datasets/andrejzuba/revolutassignment)  ***Due to size limitations, original data files are not available in this repository but can be downloaded from Kaggle.***

The introduction in the Jupyter Notebook was reproduced from instructions available as a separate file in the original Kaggle post (Data Challenge FinCrime.pdf).

## Additional References

The requirements of this capstone project also includes three separate presentations (each for a different audience) about the algorithm designed in these coding documents. All three presentations are available on Tableau Public and can be viewed in a browser:

* [For Technical Audiences](https://public.tableau.com/views/ProjectFraudPatrolTechnicalAudience/Project_Fraud_Patrol?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

* [For Executive Audiences](https://public.tableau.com/views/ProjectFraudPatrolExecutiveAudience/Project_Fraud_Patrol?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

* [For All Audiences (non-technical)](https://public.tableau.com/views/ProjectFraudPatrolAllAudiences/Project_Fraud_Patrol?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Software
Code in the notebook and python script was written and tested with Python 3.12.  Python packages used in this project are listed below in alphabetical order:
* [joblib](https://joblib.readthedocs.io/)
* [matplotlib](https://matplotlib.org)
* [numpy](https://numpy.org/)
* [optuna](https://optuna.org/)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [scipy](https://scipy.org/)
* [seaborn](https://seaborn.pydata.org/)
* [shap](https://shap.readthedocs.io/en/stable/)
* [xgboost](https://xgboost.readthedocs.io/en/stable/)

## License

The code in this repository is released under the [MIT license](LICENSE-CODE). Read more at the [Open Source Initiative](https://opensource.org/licenses/MIT).
