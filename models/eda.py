import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

titanic_df = pd.read_csv("data/processed/train.csv")

profile = ProfileReport(titanic_df, title="Titanic dataset EDA", explorative=True)

profile.to_file("titanic_eda_report.html")
