# Heart Disease Prediction Analysis

## Overview
This project aims to predict the risk of heart failure based on various medical indicators using R. It utilizes logistic regression, decision trees, and random forest models to analyze heart disease data and predict fatal myocardial infarction (fatal MI).

## Data Source
The dataset used in this analysis is the Heart Failure dataset, which can be accessed directly via the following URL: `https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv`. This dataset includes medical indicators such as age, sex, blood pressure, cholesterol levels, and more.

## Getting Started

### Prerequisites
Ensure you have R installed on your system. You can download it from [CRAN](https://cran.r-project.org/).

### Installation
1. Clone the repository to your local machine:
``` 
git clone https://github.com/JKQL/Classification-Summative-Assessment-
``` 
2. Open R and set the working directory to the cloned repository:
``` 
R
setwd("/path/to/heart-disease-prediction")
``` 
3. Install the required R packages by running the following command in R:
``` 
R
install.packages(c("ncvreg", "tree", "randomForest", "ROCR", "pROC"))
``` 
### Running the Analysis
To run the analysis, simply open the `heart_disease_analysis.R` script in R or RStudio and execute the script.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
