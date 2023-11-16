# Sam Cullen
# C00250093
import matplotlib.pyplot as plt
import pandas as pd

# from tensorflow import tf
# from fbprohet import Prophet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Load data from the CSV file
data = pd.read_csv("carAccidentsData.csv")

print(data.head)

"""
Accidents per Year Bar Graph
This code generates a bar graph showing the number of accidents per year
It utilizes the 'Accident year' column from the provided dataset, displaying the data in a bar chart
"""
# Graph to display Accidents per year
data["Accident year"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Accident Year")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Year")
plt.show()

# Process the data
data["Accident year"] = pd.to_datetime(data["Accident year"], format="%Y")
data.set_index("Accident year", inplace=True)
# Resample data to yearly frequency and count the number of accidents per year
accidents_per_year = data.resample("Y").size()

# Plot the number of accidents per year
accidents_per_year.plot(style="o-")
plt.xlabel("Year")
plt.ylabel("Number of Accidents")
plt.title("Number of Accidents per Year")
plt.show()

"""
Monthly Accidents Line Graph
This code creates a line graph to illustrate the number of accidents per month
It utilizes the 'Accident month' column from the provided dataset, sorting the data by month
The graph is labeled with months and displays the trend of accidents throughout the year
"""
# Graph to display Accidents per month
monthly_data = data["Accident month"].value_counts().sort_index()
monthly_data.plot(kind="line", marker="o")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Month")
plt.xticks(
    range(1, 13),
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
)
plt.show()

"""
Distribution for amount of male and female drivers
and Distribution of crashes caused by each gender
"""
# Creating a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Pie chart for number of male and female drivers
gender_counts = data["Casualty sex"].value_counts()
axes[0].pie(
    gender_counts,
    labels=gender_counts.index,
    autopct="%1.1f%%",
    colors=["blue", "pink"],
    startangle=90,
)
axes[0].set_title("Gender Distribution of Drivers")

# Pie chart of crashes caused by each gender
crash_counts = data.groupby("Casualty sex")["Killed or seriously injured"].sum()
axes[1].pie(
    crash_counts,
    labels=crash_counts.index,
    autopct="%1.1f%%",
    colors=["pink", "blue"],
    startangle=90,
)
axes[1].set_title("Distribution of Crashes by Gender")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

"""
Weather Condition Distribution Graph
This code generates a pie chart to visualize the distribution of weather conditions in accidents.
It uses the 'Weather condition' column from the dataset and displays the percentages for each category.
"""
# Graph for weather condition
data["Weather condition"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Distribution of Weather Conditions in Accidents")
plt.show()

"""
Weather Condition Distribution Graph
Same as above, this graph shows the updated weather condition categories and makes a horizontal bar chart
It calculates the percentage distribution of specified weather conditions in the provided dataset
The resulting chart visualizes the distribution of accidents based on different weather conditions
"""
# Updated Weather Condition Categories
weather_labels = [
    "Fine no high winds",
    "Raining no high winds",
    "Snowing no high winds",
    "Fine + high winds",
    "Raining + high winds",
    "Snowing + high winds",
    "Fog or mist",
]

# Distribution of Weather Conditions Bar Chart
sizes = (
    data["Weather condition"].value_counts(normalize=True).loc[weather_labels].fillna(0)
    * 100
)

# Create horizontal bar chart
plt.barh(
    weather_labels,
    sizes,
    color=[
        "skyblue",
        "lightcoral",
        "lightgreen",
        "gold",
        "orange",
        "cyan",
        "lightgray",
    ],
)
plt.xlabel("Percentage (%)")
plt.title("Distribution of Weather Conditions in Accidents")
plt.show()

"""
ARIMA Time Series Forecasting
This code uses an ARIMA model to forecast yearly accident counts. It splits data into
training and testing sets, fits the model, evaluates performance with MSE, and plots results
"""
# Split the data into features and target variable
X = data.drop("Killed or seriously injured", axis=1)
y = data["Killed or seriously injured"]

# Handle categorical variables with one-hot encoding
X = pd.get_dummies(X, columns=["Casualty sex", "Weather condition", "Accident month"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Choose a different model (e.g., Random Forest)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for Killed/Seriously injured:", mse)


"""
Linear Regression Prediction of Car Accidents

This code processes car accident data, resamples it to yearly frequency, and uses linear regression
to predict the number of accidents in future years (2023 and 2024). It visualizes the actual data and
future predictions using a line plot
"""
# Process the data
# Remove leading and trailing spaces from all column names
data.columns = data.columns.str.strip()

# Resample data to yearly frequency and count the number of accidents per year
accidents_per_year = data.resample("Y").size()

# Assuming 'Accident year' is your predictor variable
# and 'Killed or seriously injured' is the target variable
X = accidents_per_year.index.year.values.reshape(-1, 1)  # Features
y = accidents_per_year.values  # Target

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Generate future years for prediction (2023 and 2024)
future_years = np.array([[2023], [2024]])

# Make predictions for future years
future_predictions = model.predict(future_years)

# Display the predictions
for year, prediction in zip(future_years.flatten(), future_predictions):
    print(f"Year {year}: Predicted Number of Accidents = {prediction:.2f}")

# Plotting the results
plt.plot(X, y, color="black", label="Actual data")
plt.plot(
    future_years.flatten(),
    future_predictions,
    color="red",
    marker="o",
    linestyle="dashed",
    label="Future predictions",
)
plt.xlabel("Year")
plt.ylabel("Number of Accidents")
plt.title("Linear Regression Prediction of Car Accidents")
plt.legend()
plt.show()

"""
Linear Regression with Categorical Variables

This code shows the use of one-hot encoding for categorical variables,
handling missing values through imputation, and training a linear regression model
The model is evaluated using mean squared error on a test set
"""
# Handle categorical variables withh one-hot encoding
data["Casualty age"] = pd.to_numeric(data["Casualty age"], errors="coerce")
data["Casualty age"].fillna(data["Casualty age"].median(), inplace=True)

# Feature Engineering: Explore additional features or transformations

# Split the data into features and target variable
X = data.drop("Killed or seriously injured", axis=1)
y = data["Killed or seriously injured"]

# Handle categorical variables with one-hot encoding
#X = pd.get_dummies(X, columns=["Casualty sex", "Weather condition", "Accident month"])
X = pd.get_dummies(X, columns=["Casualty sex", "Weather condition", "Accident month"])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Choose a different model (e.g., Random Forest)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for Killed/Seriously injured:", mse)

"""
Calculate Averages

This script calculates the average values for the "Killed or seriously injured" and "Casualty age" columns
in the given dataset and prints the results.
"""
# Calculate the average of the "Killed or seriously injured" column
average_killed_or_injured = data["Killed or seriously injured"].mean()
print(f"Average Killed or Seriously Injured: {average_killed_or_injured}")
average_casualty_age = data["Casualty age"].mean()
print(f"Average Casualty Age: {average_casualty_age}")