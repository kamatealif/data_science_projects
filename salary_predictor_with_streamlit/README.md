# Salary Predictor with Streamlit

## Project Overview
This project is a simple web application that predicts a person's salary based on their years of experience using a linear regression model. The app provides an interactive interface for users to input their years of experience and instantly see the predicted salary, along with data visualizations and model performance metrics.

## Technologies Used
- **Python**
- **Streamlit**: For building the interactive web app
- **scikit-learn**: For training the linear regression model
- **pandas**: For data manipulation
- **numpy**: For numerical operations
- **matplotlib**: For data visualization

## Installation
This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management. Make sure you have `uv` installed. If not, you can install it with:

```powershell
pip install uv
```

Then, install all dependencies:

```powershell
uv pip install -r pyproject.toml
# or
uv sync
```

Or, to add new packages:

```powershell
uv add package_name
```

## Running the Project
1. Make sure you are in the project directory.
2. Start the Streamlit app using uv's latest update:

```powershell
uv pip install -r pyproject.toml  # Ensure all dependencies are installed
streamlit run main.py
```

3. The app will open in your browser. Enter your years of experience to get a salary prediction and explore the data visualizations.

## Notes
- The app expects a `salary_data.csv` file in the project directory with columns `YearsExperience` and `Salary`.
- You can modify or extend the model as needed.
- The model saving functionality is included but commented out; uncomment if you wish to persist the trained model.

---
Enjoy predicting salaries with this interactive Streamlit app!
