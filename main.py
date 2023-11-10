# Sam Cullen
# C00250093
import matplotlib.pyplot as plt
import pandas as pd
#from tensorflow import tf
#from fbprohet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


# Load data from the CSV file
data = pd.read_csv('carAccidentsData.csv')


"""
# Calculate the average of the "Casualty Age" column
average_casualty_age = data["Casualty Age"].mean()

print(f"Average Casualty Age: {average_casualty_age}")
"""

# Graph to display Accidents per year
print(data.head())
data['Accident year'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Accident Year')
plt.ylabel('Number of Accidents')
plt.title('Accidents by Year')
plt.show()

# Calculate the average of the "Killed or seriously injured" column
average_killed_or_injured = data["Killed or seriously injured"].mean()
print(f"Average Killed or Seriously Injured: {average_killed_or_injured}")

average_casualty_age = data["Casualty age"].mean()
print(f"Average Casualty Age: {average_casualty_age}")

# Graph to display Accidents per month
monthly_data = data['Accident month'].value_counts().sort_index()
monthly_data.plot(kind='line', marker='o')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.title('Accidents by Month')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

# Graph to display Casualty sex
data['Casualty sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Casualty Sex Distribution')
plt.show()


# Graph for weather condition
data['Weather condition'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Weather Conditions in Accidents')
plt.show()

# Handle categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=["Casualty sex", "Weather condition", "Accident month"])

# Handle missing values by replacing "Unknown or missing" with NaN and using median imputation
data["Casualty age"] = pd.to_numeric(data["Casualty age"], errors='coerce')
data["Casualty age"].fillna(data["Casualty age"].median(), inplace=True)

# Split the data into features and target variable
X = data.drop("Killed or seriously injured", axis=1)
y = data["Killed or seriously injured"]

# Split the data into training and testing sets                     20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a machine learning model (e.g., Linear Regression)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for Killed/Seriously injured:", mse)



# Process the data
data['Accident year'] = pd.to_datetime(data['Accident year'], format='%Y')
data.set_index('Accident year', inplace=True)

# Resample data to yearly frequency and count the number of accidents per year
accidents_per_year = data.resample('Y').size()

# Plot the number of accidents per year
accidents_per_year.plot(style='o-')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents per Year')
plt.show()

# Split the data into training and testing sets
train_size = int(len(accidents_per_year) * 0.8)
train, test = accidents_per_year[0:train_size], accidents_per_year[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))

# Evaluate the model
mse = mean_squared_error(test, predictions)
print("Mean Squared Error for Number of Accidents:", mse)

# Plot the predictions
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, predictions, label='Predictions', linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Accident predictor per Year')
plt.legend()
plt.show()