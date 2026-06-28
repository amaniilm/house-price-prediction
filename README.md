# house-price-prediction
House Price Prediction App | Python | Scikit-learn | Streamlit | Random Forest Regression
# 🏠 House Price Prediction

This is one of my machine learning projects where I built a model to predict house prices based on different property features like location, size, number of bedrooms, bathrooms, and the age of the house.

I trained the model using **Random Forest Regression** and created a simple **Streamlit** web app so users can enter house details and get a predicted price.

## Features

* Data cleaning and preprocessing
* Handling missing values using SimpleImputer
* Feature engineering (House Age)
* Label Encoding for location
* Random Forest Regression model
* Streamlit web application
* Model saved using Pickle

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

## Project Structure

```
house-price-prediction/
│── app.py
│── house_price_prediction.pkl
│── labelencoder.pkl
│── simpleimputer.pkl
│── requirements.txt
│── README.md
```

## How to Run

1. Clone the repository.

```
git clone https://github.com/your-username/house-price-prediction.git
```

2. Install the required libraries.

```
pip install -r requirements.txt
```

3. Start the Streamlit app.

```
streamlit run app.py
```

## What I Learned

While building this project, I learned how to:

* Clean and preprocess real-world datasets
* Handle missing values
* Perform feature engineering
* Train and evaluate a machine learning model
* Save models using Pickle
* Deploy a Streamlit application

## Future Improvements

* Tune the model for better accuracy
* Try XGBoost and LightGBM
* Deploy the project with Docker

## Author

Aman kumar


