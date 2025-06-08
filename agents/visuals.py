# agents/visuals.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_numeric_distribution(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax, color='skyblue')
    ax.set_title(f'Distribution of {column}')
    return fig

def plot_boxplot(df, column):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column], ax=ax, color='lightgreen')
    ax.set_title(f'Boxplot of {column}')
    return fig

def plot_barplot(df, column):
    fig, ax = plt.subplots()
    value_counts = df[column].value_counts().nlargest(10)
    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='viridis')
    ax.set_title(f'Barplot of {column}')
    return fig

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig
