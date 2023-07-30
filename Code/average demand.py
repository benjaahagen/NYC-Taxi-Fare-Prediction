import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

main_sample = pd.read_csv('Main Sample.csv')

# Convert tpep_pickup_datetime to datetime object
main_sample['tpep_pickup_datetime'] = pd.to_datetime(main_sample['tpep_pickup_datetime'])

# Extract the day of the week from the pickup datetime and create a new column 'day_of_week'
main_sample['day_of_week'] = main_sample['tpep_pickup_datetime'].dt.day_name()

# Group the data by day of the week and count the number of trips
daily_demand = main_sample.groupby('day_of_week').size()

# Calculate the average demand for each day
average_daily_demand = daily_demand / len(main_sample['day_of_week'].unique())

# Sort the average daily demand by day order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
average_daily_demand = average_daily_demand.loc[day_order]

# Create a DataFrame for the average daily demand
average_daily_demand_df = pd.DataFrame(average_daily_demand).reset_index()
average_daily_demand_df.columns = ['Day of the Week', 'Average Daily Demand']

# Days with the highest and lowest demand
highest_demand_day = average_daily_demand.idxmax()
lowest_demand_day = average_daily_demand.idxmin()

print("The average daily demand for taxis is:\n")
print(average_daily_demand_df)
print("\n")
print(f"The day with the highest demand is {highest_demand_day} with an average of {average_daily_demand[highest_demand_day]:,.0f} trips.")
print(f"The day with the lowest demand is {lowest_demand_day} with an average of {average_daily_demand[lowest_demand_day]:,.0f} trips.")

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
bar_plot = sns.barplot(x="Day of the Week", y="Average Daily Demand", data=average_daily_demand_df, palette="coolwarm")

plt.xlabel('Day of the Week', fontsize=13)
plt.ylabel('Average Daily Demand', fontsize=13)
plt.title('Average Daily Demand for Taxis', fontsize=16)
plt.xticks(rotation=45)

# Annotate the bars with the average daily demand values
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.0f'), 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', 
                      va = 'center', 
                      xytext = (0, 10), 
                      textcoords = 'offset points',
                      fontsize = 10)

# Days to index map
day_to_index = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

# Highlight the day with highest demand
bar_plot.patches[day_to_index[highest_demand_day]].set_facecolor('green') 

# Highlight the day with lowest demand
bar_plot.patches[day_to_index[lowest_demand_day]].set_facecolor('red') 

# Save the plot as an image
plt.savefig('average_daily_demand.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

