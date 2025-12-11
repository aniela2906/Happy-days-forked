# ITU BDS MLOPS'25 - Project

This repository contains our implementation of a fully reproducible end-to-end MLOps pipeline developed for the ITU BDS MLOps 2025 course.
The goal was to transform the original exploratory notebook into a modular, automated ML system using DVC for data management, MLflow for experiment tracking, and GitHub Actions + Dagger for CI/CD automation.
  
  
## Project Structure
```bash
Happy-days-forked/  
│   
├── artifacts/                     # Model artifacts, metrics & outputs  
│   ├── model_selection.json       # Summary of the best model  
│   ├── lr_metrics.json            # Metrics from the Linear Regression run  
│   ├── xgboost_metrics.json       # Metrics from the XGBoost run  
│   ├── columns_list.json          # Column names used for training  
│   └── date_limits.json           # Date filtering boundaries  
│  
├── notebooks/                     # Original notebooks provided for the project  
│   ├── main.ipynb  
│   ├── model_inference.py  
│   └── requirements.txt  
│  
├── src/  
│   ├── data/                      # Data loading, cleaning, preprocessing  
│   ├── models/                    # Training scripts for LR and XGBoost  
│   └── pipeline/                  # Pipeline orchestration & model selection  
│  
├── .gitignore  
├── README.md  
└── requirements.txt               # Project dependencies needed to run the pipeline  
  ```
  
  
### The pipeline performs the following steps:
- Load & preprocess data (cleaning, feature generation, train–test split)
- Train two models: Linear Regression and XGBoost
- Log experiments to MLflow, including parameters, metrics, and artifacts
- Select the best model based on evaluation metrics
- Export artifacts (model selection results, metrics, columns used, date limits)
- Automate training via GitHub Actions and Dagger workflows NOT DONE YET
- Ensure reproducibility using DVC for data versioning and pipeline tracking NOT DONE YET


### HOW TO run the pipeline:
1. Clone the repository
  ```bash
  git clone https://github.com/aniela2906/Happy-days-forked.git
  cd Happy-days-forked
  ```
2. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
3. Run the code ( data preprocessing → model training → model selection) 
  ```bash
  python -m src.pipeline.train
  ```
  This command runs:
- preprocessing  
- feature extraction  
- LR training  
- XGBoost training  
- model selection  
- MLflow logging  
- artifact export  
