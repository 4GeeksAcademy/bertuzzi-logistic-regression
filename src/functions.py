
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 #%%

def create_univariate_charts(customer_data):
    fig, axis = plt.subplots(9, 2, figsize=(18, 35))

    # Age distribution (Histogram with KDE)
    sns.histplot(ax=axis[0, 0], data=customer_data, x='age', kde=True)
    axis[0, 0].set_title('Age Distribution')
    axis[0, 0].set_xlabel('Age')
    axis[0, 0].set_ylabel('Frequency')

    # Marital status distribution
    sns.countplot(ax=axis[0, 1], data=customer_data, x='marital')
    axis[0, 1].set_title('Marital Status Distribution')
    axis[0, 1].set_xlabel('Marital Status')
    axis[0, 1].set_ylabel('Count')

    # Education distribution
    sns.countplot(ax=axis[1, 0], data=customer_data, x='education')
    axis[1, 0].set_title('Education Distribution')
    axis[1, 0].set_xlabel('Education Level')
    axis[1, 0].set_ylabel('Count')

    # Default status distribution
    sns.countplot(ax=axis[1, 1], data=customer_data, x='default')
    axis[1, 1].set_title('Default Status Distribution')
    axis[1, 1].set_xlabel('Default')
    axis[1, 1].set_ylabel('Count')

    # Housing distribution
    sns.countplot(ax=axis[2, 0], data=customer_data, x='housing')
    axis[2, 0].set_title('Housing Distribution')
    axis[2, 0].set_xlabel('Housing')
    axis[2, 0].set_ylabel('Count')

    # Loan distribution
    sns.countplot(ax=axis[2, 1], data=customer_data, x='loan')
    axis[2, 1].set_title('Loan Distribution')
    axis[2, 1].set_xlabel('Loan')
    axis[2, 1].set_ylabel('Count')

    # Contact communication method distribution
    sns.countplot(ax=axis[3, 0], data=customer_data, x='contact')
    axis[3, 0].set_title('Contact Communication Method')
    axis[3, 0].set_xlabel('Contact Method')
    axis[3, 0].set_ylabel('Count')

    # Month distribution
    sns.countplot(ax=axis[3, 1], data=customer_data, x='month')
    axis[3, 1].set_title('Month Distribution')
    axis[3, 1].set_xlabel('Month')
    axis[3, 1].set_ylabel('Count')

    # Day of the week distribution
    sns.countplot(ax=axis[4, 0], data=customer_data, x='day_of_week')
    axis[4, 0].set_title('Day of Week Distribution')
    axis[4, 0].set_xlabel('Day of Week')
    axis[4, 0].set_ylabel('Count')

    # Duration distribution (Histogram with KDE)
    sns.histplot(ax=axis[4, 1], data=customer_data, x='duration', kde=True)
    axis[4, 1].set_title('Duration Distribution')
    axis[4, 1].set_xlabel('Duration')
    axis[4, 1].set_ylabel('Frequency')

    # Campaign distribution (Histogram with KDE)
    sns.histplot(ax=axis[5, 0], data=customer_data, x='campaign', kde=True)
    axis[5, 0].set_title('Campaign Distribution')
    axis[5, 0].set_xlabel('Campaign')
    axis[5, 0].set_ylabel('Frequency')

    # Pdays distribution (Histogram with KDE)
    sns.histplot(ax=axis[5, 1], data=customer_data, x='pdays', kde=True)
    axis[5, 1].set_title('Pdays Distribution')
    axis[5, 1].set_xlabel('Pdays')
    axis[5, 1].set_ylabel('Frequency')

    # Previous distribution (Histogram with KDE)
    sns.histplot(ax=axis[6, 0], data=customer_data, x='previous', kde=True)
    axis[6, 0].set_title('Previous Distribution')
    axis[6, 0].set_xlabel('Previous')
    axis[6, 0].set_ylabel('Frequency')

    # Poutcome distribution (Histogram with KDE)
    sns.histplot(ax=axis[6, 1], data=customer_data, x='poutcome', kde=True)
    axis[6, 1].set_title('Poutcome Distribution')
    axis[6, 1].set_xlabel('Poutcome')
    axis[6, 1].set_ylabel('Frequency')

    # Y distribution (Target variable count plot)
    sns.countplot(ax=axis[7, 0], data=customer_data, x='y')
    axis[7, 0].set_title('Target Variable: Y Distribution')
    axis[7, 0].set_xlabel('Y')
    axis[7, 0].set_ylabel('Count')

    # Boxplot for age
    sns.boxplot(ax=axis[7, 1], data=customer_data, x='age')
    axis[7, 1].set_title('Age Boxplot')
    axis[7, 1].set_xlabel('Age')

    # Boxplot for duration
    sns.boxplot(ax=axis[8, 0], data=customer_data, x='duration')
    axis[8, 0].set_title('Duration Boxplot')
    axis[8, 0].set_xlabel('Duration')

    # Remove any empty subplots (if applicable)
    fig.delaxes(axis[8, 1])