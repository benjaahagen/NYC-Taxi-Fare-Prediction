import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

main_sample = pd.read_csv('Main Sample.csv')

# Define time periods
time_periods = {
    'Morning':   (datetime.strptime('04:00:00', '%H:%M:%S').time(), datetime.strptime('11:59:59', '%H:%M:%S').time()),
    'Afternoon': (datetime.strptime('12:00:00', '%H:%M:%S').time(), datetime.strptime('17:59:59', '%H:%M:%S').time()),
    'Evening':   (datetime.strptime('18:00:00', '%H:%M:%S').time(), datetime.strptime('23:59:59', '%H:%M:%S').time()),
    'Night':     (datetime.strptime('00:00:00', '%H:%M:%S').time(), datetime.strptime('03:59:59', '%H:%M:%S').time())
}

# Extract the pickup time from the pickup datetime and create a new column 'pickup_time'
main_sample['pickup_time'] = pd.to_datetime(main_sample['tpep_pickup_datetime']).dt.time

# Assign the time period for each pickup time
def get_time_period(pickup_time):
    for period, time_range in time_periods.items():
        if time_range[0] <= pickup_time <= time_range[1]:
            return period
    return 'Unknown'

main_sample['time_period'] = main_sample['pickup_time'].apply(get_time_period)

# Group the data by time period and count the number of trips
time_period_demand = main_sample.groupby('time_period').size()

# Time period with the highest demand
peak_period = time_period_demand.idxmax()

print(f"The demand for taxis in different time periods is:\n{time_period_demand}\n")
print(f"The peak period for taxi operation is {peak_period} with {time_period_demand[peak_period]} trips.")

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
bar_plot = sns.barplot(x=time_period_demand.index, y=time_period_demand.values, palette='viridis')

plt.xlabel('Time Period', fontsize=12)
plt.ylabel('Demand', fontsize=12)
plt.title('Demand for Taxis in Different Time Periods', fontsize=15)

# Annotate the bars with the demand values
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

# Save the plot as an image
plt.savefig('time_period_demand.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
