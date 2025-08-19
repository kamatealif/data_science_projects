import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

if __name__ == "__main__":
    df = pd.read_csv('salary_data.csv')

    # feature selection 

    X = df[['YearsExperience']]
    y = df['Salary']
    
    # model training
    model = LinearRegression()
    model.fit(X, y)

    st.title('Salary Prediction App')
    st.write("This app predicts salary based on years of experience.")

    st.write("### Input your years of experience:")
    years_input = st.number_input('Years of Experience', min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    if years_input:
        print(f"Years of Experience: {years_input}")
        prediction = model.predict([[years_input]])[0]
        st.success(f'Predicted Salary: ${prediction:,.2f}')

    # Plotting the data
    st.write("### Data Visualization")
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
    plt.title('Years of Experience vs Salary')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    st.pyplot(plt)

    # Display the dataframe
    st.write("### Data Overview")
    st.dataframe(df.head())

    # Display model coefficients
    st.write("### Model Coefficients")
    st.write(f"Coefficient: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}")

    # Display the model's score
    st.write("### Model Performance")
    st.write(f"Model R^2 Score: {model.score(X, y):.2f}")

    # Save the model if needed
    # import joblib
    # joblib.dump(model, 'salary_model.pkl')
    st.write("Model saved as 'salary_model.pkl' (commented out for now).")
    st.write("You can uncomment the saving part to save the model for future use.")
    st.write("Enjoy predicting salaries based on years of experience!")
    st.write("This app is built using Streamlit and scikit-learn.")
    st.write("Make sure to have the required libraries installed: pandas, numpy, matplotlib, scikit-learn, streamlit.")