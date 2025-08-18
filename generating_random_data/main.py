import pandas as pd
import numpy as np

def main():
    np.random.seed(42)  # For reproducibility
    years = np.random.uniform(0.5, 10, size=1000).round(2)
    salary = (30000 + years * 5000 + np.random.normal(0, 5000, size=1000) ).round(2)

    df = pd.DataFrame({
        'Years of Experience': years,
        'Salary': salary
    })
    df.to_csv('salary_data.csv', index=False)

    print("Data generation complete. 'salary_data.csv' created.")
    print("have a look at the first few rows of the dataset:")
    print(df.head())

if __name__ == "__main__":
    main()
