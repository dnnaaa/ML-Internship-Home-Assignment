from pathlib import Path

PARENT_PATH = Path(__file__).parent.parent

DATA_PATH = PARENT_PATH / "data/"
RAW_DATASET_PATH = DATA_PATH / "raw/resume.csv"
PROCESSED_DATASET_PATH = DATA_PATH / "processed/resume.csv"

PROCESSED_DATA_PATH = PARENT_PATH / "processed/"

MODELS_PATH = PARENT_PATH / "models/"
NAIVE_BAYES_PIPELINE_PATH = MODELS_PATH / "naive_bayes_pipeline.joblib"
REPORTS_PATH = PARENT_PATH / "reports/"
CM_PLOT_PATH = REPORTS_PATH / "cm_plot.png"

DATABASE_URL = f"sqlite:///{PARENT_PATH}/predictions.db"

BASE_URL = "http://localhost:9003"
API_BASE_URL = BASE_URL + "/api"
PREDICTIONS_BASE_URL = API_BASE_URL + "/predictions"

INFERENCE_ENDPOINT = "/inference"
INFERENCE_API_URL = API_BASE_URL + INFERENCE_ENDPOINT


SAMPLES_PATH = PARENT_PATH / "samples"
LABELS_MAP = {0: '.Net Developer', 1: 'Business Analyst', 2: 'Business Intelligence', 3: 'Help Desk and Support', 4: 'Informatica Developer', 5: 'Java Developer', 6: 'Network and System Administrator', 7: 'Oracle DBA', 8: 'Project Manager', 9: 'Quality Assurance', 10: 'SAP', 11: 'SQL Developer', 12: 'Sharepoint Developer', 13: 'Web Developer'}