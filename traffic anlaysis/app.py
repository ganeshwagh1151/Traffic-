import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('Car.csv')

# Initialize session_state
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploration", "Linear Regression"])

# Home page
if page == "Home":
    st.title("Traffic Data Dashboard")
    st.write("Welcome to the Road Accident Analysis Dashboard!")

    # Add image
    st.image("https://miro.medium.com/max/1076/1*UR95UCqXZrVAg8iIsaPn7g.jpeg", use_column_width=True)

# Exploration page
elif page == "Exploration":
    st.title("Data Exploration")

    # Display the data
    st.subheader("Raw Data")
    st.write(df)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

# Linear Regression page
elif page == "Linear Regression":
    st.title("Linear Regression Model")

    # Prepare the data for modeling
    y = df["Total Road Accidents Cases"]
    x = df.drop(["Total Road Accidents Cases", "State/UT/City"], axis=1)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)

    # Identify NaN values in y_train
    nan_indices = np.isnan(y_train)

    # Remove rows with NaN values
    x_train_imputed = x_train_imputed[~nan_indices]
    y_train = y_train[~nan_indices]

    # Train the Linear Regression model
    lr = LinearRegression()
    lr.fit(x_train_imputed, y_train)

    # User input for prediction
    st.subheader("Enter Features for Prediction")
    features = {}

    for column in x.columns:
        features[column] = st.number_input(f"Enter {column}", value=x[column].mean())

    # Convert user input to DataFrame
    user_input = pd.DataFrame([features])

    # Prediction function
    def predict():
        prediction = lr.predict(user_input)
        return prediction[0]

    # Display prediction
    if st.button("Submit"):
        st.subheader("Prediction")
        result = predict()
        st.write(f"Predicted Total Road Accidents Cases: {result:.2f}")

    # Make predictions for evaluation
    y_lr_train_pred = lr.predict(x_train_imputed)
    y_lr_test_pred = lr.predict(x_test_imputed)

    # Evaluate the model
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    # Display model performance metrics
    st.subheader("Model Performance Metrics")
    st.write(f"Training MSE: {lr_train_mse:.2f}")
    st.write(f"Training R^2: {lr_train_r2:.2f}")
    st.write(f"Testing MSE: {lr_test_mse:.2f}")
    st.write(f"Testing R^2: {lr_test_r2:.2f}")
