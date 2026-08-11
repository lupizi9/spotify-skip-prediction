# Spotify Skip Prediction

Machine learning project focused on predicting whether a user will skip a song based on information about the user, the track, and the listening context.

## Overview

This project addresses the problem of predicting song skips on Spotify using **supervised machine learning** techniques.

The objective is to estimate the probability that a user will skip a song based on information related to the user, the track, and the context in which the song is played.

The project follows the main stages of a typical machine learning workflow, including data cleaning and preparation, feature engineering, model training, and evaluation.

An iterative approach was followed, starting with simpler models such as **Decision Trees**, and progressively incorporating new variables and feature combinations to improve predictive performance. More advanced ensemble methods, including **Random Forest** and **XGBoost**, were then explored.

Finally, a **blend of Random Forest and XGBoost** models was developed to optimize predictive performance, with the models evaluated using the **ROC-AUC** metric.

To ensure robust evaluation and avoid **data leakage**, the project uses a validation strategy based on users, ensuring that information from the same user is not improperly shared between training and validation data.

## Academic Context

This project was developed as part of **TD6 – Artificial Intelligence** at Universidad Torcuato Di Tella.
