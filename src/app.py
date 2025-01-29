from utils import db_connect
engine = db_connect()

# your code here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functions import create_univariate_charts

pd.set_option('display.max_columns', None)

 #%%

customer_data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', sep=';')
print(customer_data.head())

# STEP 1: PERFORM EDA

# Gain first insights into dataset
print(customer_data.info())
print(customer_data.describe())

# Handle duplicates
duplicates = customer_data.duplicated().sum()
if duplicates !=0:
    customer_data.drop_duplicates(inplace=True)

# Remove dimensions irrelated to business problem

customer_data.drop(['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1, inplace=True)

# Convert categorical values to numerical
cat_dimensions = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']

# Dictionary to store mapping
factorize_mappings = {}

for col in cat_dimensions:
    encoded_value, categories = pd.factorize(customer_data[col])
    factorize_mappings[col] = dict(enumerate(categories))
    customer_data[col] = encoded_value
print(customer_data.head())
print(customer_data.info())
print(factorize_mappings)

# Visualize univariate values
create_univariate_charts(customer_data)

plt.tight_layout()
plt.show()