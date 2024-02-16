import csv
from datetime import datetime, timedelta

# Define parameters
window_minutes = 2
frequency_threshold = 5 # e.g., 10 entries/exits within the defined window


def read_csv(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        data = [(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f"), int(row[1])) for row in reader]
    return data


def find_crowd_times_by_frequency_and_occupancy(data, window_minutes, frequency_threshold):
    window_seconds = window_minutes * 60
    entrance_times, exit_times = [], []

    for i in range(len(data) - 1):
        current_time = data[i][0]
        potential_end_time = current_time + timedelta(seconds=window_seconds)
        events_within_window = [x for x in data if current_time <= x[0] <= potential_end_time]

        # If the frequency threshold is met
        if len(events_within_window) >= frequency_threshold:
            # Check for entrance (occupancy increase) or exit (occupancy decrease)
            if events_within_window[-1][1] > events_within_window[0][1]:
                entrance_times.append(events_within_window[-1][0])
            elif events_within_window[-1][1] < events_within_window[0][1]:
                exit_times.append(events_within_window[-1][0])

    return entrance_times, exit_times


data = read_csv("raw_sensing.csv")  # Replace with your file path
entrance_times, exit_times = find_crowd_times_by_frequency_and_occupancy(data, window_minutes, frequency_threshold)

print("Crowd Entrance Times:")
for time in entrance_times:
    print(time)

print("\nCrowd Exit Times:")
for time in exit_times:
    print(time)
