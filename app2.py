# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
# import pickle
# import os

# print("Loading car price prediction app...")

# # Function to load or train the model
# @st.cache_resource
# def load_or_train_model():
#     print("Inside load_or_train_model function")
#     model_path = 'car_price_prediction_model.pkl'
#     encoders_path = 'label_encoders.pkl'
    
#     if os.path.exists(model_path) and os.path.exists(encoders_path):
#         print("Loading existing model and encoders")
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
#         with open(encoders_path, 'rb') as file:
#             label_encoders = pickle.load(file)
#     else:
#         print("Training new model")
#         # Load the data
#         url = 'https://raw.githubusercontent.com/rashida048/Datasets/master/USA_cars_datasets.csv'
#         df = pd.read_csv(url)
        
#         # Preprocess the data
#         df = df[['price', 'brand', 'model', 'year', 'mileage', 'state']]
        
#         # Encode categorical variables
#         label_encoders = {}
#         for column in ['brand', 'model', 'state']:
#             label_encoders[column] = LabelEncoder()
#             df[column] = label_encoders[column].fit_transform(df[column])
        
#         # Prepare the features (X) and target (y)
#         X = df.drop('price', axis=1)
#         y = df['price']
        
#         # Create and train the model
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X, y)
        
#         # Save the model and encoders
#         with open(model_path, 'wb') as file:
#             pickle.dump(model, file)
#         with open(encoders_path, 'wb') as file:
#             pickle.dump(label_encoders, file)
    
#     return model, label_encoders

# # Load or train the model
# model, label_encoders = load_or_train_model()

# # Streamlit app
# st.title('Car Price Prediction App')

# # Input fields for car details
# brand = st.selectbox('Brand', label_encoders['brand'].classes_)
# car_model = st.selectbox('Model', label_encoders['model'].classes_)
# year = st.number_input('Year', min_value=1900, max_value=2023, value=2020)
# mileage = st.number_input('Mileage', min_value=0, value=50000)
# state = st.selectbox('State', label_encoders['state'].classes_)

# # Make prediction when the user clicks the button
# if st.button('Predict Price'):
#     input_data = pd.DataFrame({
#         'brand': [brand],
#         'model': [car_model],
#         'year': [year],
#         'mileage': [mileage],
#         'state': [state]
#     })
    
#     # Encode categorical variables
#     for column in ['brand', 'model', 'state']:
#         input_data[column] = label_encoders[column].transform(input_data[column])
    
#     prediction = model.predict(input_data)
#     st.success(f'Predicted Car Price: ${prediction[0]:,.2f}')

# # Optional: Add a section to show model details
# st.sidebar.header('Model Details')
# st.sidebar.write('This app uses a Random Forest Regressor to predict car prices based on various features.')
# st.sidebar.write('Features used: Brand, Model, Year, Mileage, State')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pickle
import os

print("Loading optimized car price prediction app with Random Forest...")

@st.cache_resource
def load_or_train_model():
    print("Inside load_or_train_model function")
    model_path = 'car_price_prediction_model_rf_optimized.pkl'
    encoders_path = 'label_encoders_rf_optimized.pkl'
    scaler_path = 'scaler_rf_optimized.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(scaler_path):
        print("Loading existing model, encoders, and scaler")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(encoders_path, 'rb') as file:
            label_encoders = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
    else:
        print("Training new optimized Random Forest model")
        url = 'https://raw.githubusercontent.com/rashida048/Datasets/master/USA_cars_datasets.csv'
        df = pd.read_csv(url)
        
        df = df[['price', 'brand', 'model', 'year', 'mileage', 'state']]
        
        # Feature engineering
        df['age'] = 2023 - df['year']
        df['mileage_per_year'] = df['mileage'] / df['age']
        
        label_encoders = {}
        for column in ['brand', 'model', 'state']:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
        
        X = df.drop(['price', 'year'], axis=1)
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(random_state=42))
        ])
        
        # Define the parameter grid
        param_grid = {
            'rf__n_estimators': [100, 200, 300, 400, 500],
            'rf__max_depth': [10, 20, 30, 40, 50, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__max_features': ['auto', 'sqrt', 'log2']
        }
        
        # Perform randomized search
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        
        # Get the best model
        model = random_search.best_estimator_
        
        # Evaluate the model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
        print(f"Train R2: {train_r2}, Test R2: {test_r2}")
        
        # Feature importance
        feature_importance = model.named_steps['rf'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        print("Feature Importance:")
        print(feature_importance_df)
        
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        with open(encoders_path, 'wb') as file:
            pickle.dump(label_encoders, file)
        with open(scaler_path, 'wb') as file:
            pickle.dump(model.named_steps['scaler'], file)
    
    return model, label_encoders

model, label_encoders = load_or_train_model()

st.title('Car Price Prediction App')

brand = st.selectbox('Brand', label_encoders['brand'].classes_)
car_model = st.selectbox('Model', label_encoders['model'].classes_)
year = st.number_input('Year', min_value=1900, max_value=2023, value=2020)
mileage = st.number_input('Mileage', min_value=0, value=50000)
state = st.selectbox('State', label_encoders['state'].classes_)

if st.button('Predict Price'):
    age = 2023 - year
    mileage_per_year = mileage / age if age > 0 else 0
    
    input_data = pd.DataFrame({
        'brand': [brand],
        'model': [car_model],
        'mileage': [mileage],
        'state': [state],
        'age': [age],
        'mileage_per_year': [mileage_per_year]
    })
    
    for column in ['brand', 'model', 'state']:
        input_data[column] = label_encoders[column].transform(input_data[column])
    
    prediction = model.predict(input_data)
    st.success(f'Predicted Car Price: ${prediction[0]:,.2f}')

st.sidebar.header('Model Details')
st.sidebar.write('This app uses an optimized Random Forest Regressor to predict car prices.')
st.sidebar.write('Features used: Brand, Model, Mileage, State, Age, Mileage per Year')
st.sidebar.write('The model is optimized using RandomizedSearchCV and includes feature engineering.')

# Display feature importance if available
if hasattr(model.named_steps['rf'], 'feature_importances_'):
    feature_importance = model.named_steps['rf'].feature_importances_
    feature_importance_df = pd.DataFrame({'feature': ['brand', 'model', 'mileage', 'state', 'age', 'mileage_per_year'], 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    st.sidebar.subheader('Feature Importance')
    st.sidebar.dataframe(feature_importance_df)