# Breast Cancer Survival Classification
Project of Machine Learning (ML) Course for Master in Data Science Program of Universitat Politècnica de Catalunya (UPC)
***
## Instructions for Executing Analysis Notebooks

* Include in a single folder the files `01.EDA-feature-extraction.ipynb`, `02.Modeling.ipynb` and `Breast_Cancer.csv` included in the repository.
* Execute the notebooks with the following order:
	* `01.EDA-feature-extraction.ipynb`
		* This notebook generates in the end the updated dataset, which includes all the changes introduced during the EDA.
		* The name of the updated dataset is set to be `breast_cancer_new.csv`. This file is being used from the second notebook in order to complete the modeling part.
	* `02.Modeling.ipynb`
* Both notebooks generate some extra csv files, containing information used during the analysis
	* E.g.: `chi-2.csv` and `chi-2-2.csv` files mentioned in the report are generated during the execution of the above-mentioned notebooks. They can also be found in this repository.
* Uncomment the first cell of the notebooks in order to install missing libraries.
* Click on the `Run all` button of the notebooks to reproduce the results of the whole project.

## Analysis Includes
1. Exploratory Data Analysis (EDA)
    * Univariate Exploratory Analysis
    * Univariate Outliers Analysis
    * Bivariate Exploratory Analysis
    * Multivariate Outlier Analysis
    * Feature Selection/Extraction
2. Modeling
    * Dataset Splits
    * Preprocessing
    * Learning Algorithms
    * Model Comparison & Hyper Parameter Tuning
    	* 5-fold Cross Validation on the following models, with multiple hyper-parameter values:
     		* Logistic Regression
       		* Random Forest
       		* SVM
       		* Gradient Boosting
       		* kNN
       		* Decision Tree
       		* Naive Bayes
    * Final Model Performance Analysis (Generalization, bias, variance analysis)
        * Training-Test Error Analysis on the max_depth Parameter
        * Training-Test Error Analysis on the n_estimators Parameter
        * Training-Test Error Analysis on the Training Data Size
        * Final Performance Metrics
        * Interpretability of the Final Model
3. Limitations & Future Work

## Executed Notebooks
The following two `html` files, present the notebooks of the project executed.
* `EDA-feature-extraction.html`
* `Modeling.html`
