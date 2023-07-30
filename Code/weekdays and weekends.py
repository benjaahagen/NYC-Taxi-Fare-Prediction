import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

main_sample = pd.read_csv('Main Sample.csv')

# Convert the pickup datetime column to a datetime format
main_sample['tpep_pickup_datetime'] = pd.to_datetime(main_sample['tpep_pickup_datetime'])

# Extract day of the week from the tpep_pickup_datetime column
main_sample['day_of_week'] = main_sample['tpep_pickup_datetime'].dt.day_name()

# Define a function to categorize the day type
def day_type(day):
    if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        return 'Weekday'
    else:
        return 'Weekend'

# Create a new column 'day_type' to categorize the pickup day as 'Weekday' or 'Weekend'
main_sample['day_type'] = main_sample['day_of_week'].apply(day_type)

# Group the data by day type and calculate the total revenue and the number of unique days
day_type_stats = main_sample.groupby('day_type').agg({'total_amount': 'sum', 'tpep_pickup_datetime': lambda x: x.dt.date.nunique()})

# Calculate the average revenue per day for weekdays and weekends
day_type_stats['average_revenue_per_day'] = day_type_stats['total_amount'] / day_type_stats['tpep_pickup_datetime']

# Style for the graph
sns.set(style="whitegrid")

# Print the day type statistics
print("The average revenue generated on weekdays and weekends is:\n")
print(day_type_stats['average_revenue_per_day'])

# Create the bar plot
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=day_type_stats.index, y=day_type_stats['average_revenue_per_day'], palette='viridis')

plt.xlabel('Day Type', fontsize=14)
plt.ylabel('Average Daily Revenue', fontsize=14)
plt.title('Average Daily Revenue for Weekdays vs. Weekends', fontsize=16)

# Customize the y-axis to show the values in millions
bar_plot.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x/1_000_000)}M"))

# Annotate the bars with their respective heights (average daily revenue)
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height()/1_000_000, '.2f') + 'M', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

# Save the plot as an image
plt.savefig('weekdays_vs_weekends.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
