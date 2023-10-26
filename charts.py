# Sam Cullen
# C00250093

import matplotlib.pyplot as plt
import pandas as pd

# Load data from the CSV file
data = pd.read_csv('carAccidentsData.csv')

# Graph to display Accidents per year
print(data.head())
data['Accident year'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Accident Year')
plt.ylabel('Number of Accidents')
plt.title('Accidents by Year')
plt.show()

# Graph to display Accidents per month
monthly_data = data['Accident month'].value_counts().sort_index()
monthly_data.plot(kind='line', marker='o')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.title('Accidents by Month')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

# Graph for weather condition
data['Weather condition'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Weather Conditions in Accidents')
plt.show()


# Graph to display Casualty sex
data['Casualty sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Casualty Sex Distribution')
plt.show()
