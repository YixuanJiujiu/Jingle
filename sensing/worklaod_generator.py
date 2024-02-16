# body = "code=#include <iostream>%0A  using namespace std; %0Aint main (){%0Aint n,s,ans,a[4],i;%0A while(cin>>n){	%0As=0;%0A		while(n--){	%0Aans=0;%0A 	for(i=0;i<3;i%2B%2B) {%0A	cin>>a[i];%0A	if (a[i]==1)%0A	ans%2B%2B; } %0A	if (ans>=2)	s%2B%2B;  %0A } cout<<s<<endl;  } return 0;}"
# headers = {"Content-Type": "application/x-www-form-urlencoded"}
# response = requests.post(url, data=body, headers=headers)

import csv
import time
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

CSV_FILENAME = 'test_requests.csv'
OUTPUT_CSV_FILENAME = 'output_latency.csv'
BASE_URL = 'http://localhost:6000/submit'


def send_request(_):
    try:
        start_time = time.time()
        body = "code=#include <iostream>%0A  using namespace std; %0Aint main (){%0Aint n,s,ans,a[4],i;%0A while(cin>>n){	%0As=0;%0A		while(n--){	%0Aans=0;%0A 	for(i=0;i<3;i%2B%2B) {%0A	cin>>a[i];%0A	if (a[i]==1)%0A	ans%2B%2B; } %0A	if (ans>=2)	s%2B%2B;  %0A } cout<<s<<endl;  } return 0;}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(BASE_URL, data=body, headers=headers)
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        return (start_time, latency)
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None


def main():

    with open(CSV_FILENAME, 'r') as csv_file, open(OUTPUT_CSV_FILENAME, 'w', newline='') as output_file:
        csv_reader = csv.reader(csv_file)
        csv_writer = csv.writer(output_file)

        # Write header for the output file
        csv_writer.writerow(["timestamp", "load", "p99_latency"])

        next(csv_reader)  # Skip header row

        prev_timestamp = None
        for row in csv_reader:
            latencies = []
            timestamp_str, load = row
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")  # Adjust the format if necessary

            # Calculate sleep interval based on the difference with the previous timestamp
            if prev_timestamp:
                interval = (timestamp - prev_timestamp).total_seconds()
                time.sleep(interval)

            # Send concurrent requests based on the load
            with ThreadPoolExecutor(max_workers=int(load)) as executor:
                results = list(executor.map(send_request, [None] * int(load)))
                real_timestamps_latencies = list(filter(None, results))
                for real_timestamp, latency in real_timestamps_latencies:
                    latencies.append(latency)

            # Calculate p99 latency
            sorted_latencies = sorted(latencies)
            p99_index = int(0.99 * len(sorted_latencies))
            p99_latency = sorted_latencies[p99_index] if sorted_latencies else 0

            # Log to CSV
            csv_writer.writerow([real_timestamp, load, p99_latency])
            print(f"all latency {latencies}")
            print(f"Requests sent at {real_timestamp}. Load: {load}. p99 Latency: {p99_latency:.2f}ms")

            prev_timestamp = timestamp


if __name__ == "__main__":
    main()