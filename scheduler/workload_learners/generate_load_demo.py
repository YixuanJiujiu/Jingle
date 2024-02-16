import csv
from random import randint

with open('synthentic.csv', 'w+', newline='') as file:
    writer = csv.writer(file)
    field = ["timestamp", "load"]
    load_variation = [1, 1, 1, 2, 4, 4, 4, 2, 1, 1, 1,
                      1, 1, 1, 2, 7, 9, 10, 10, 9, 1, 1,
                      1, 1, 1, 2, 6, 8, 7, 8, 2, 1, 1]
    idx = 1
    duration = 180
    timestamp = 0
    writer.writerow(field)

    while idx <= duration:
        timestamp = timestamp + 1
        load = load_variation[idx%len(load_variation)] * 30 * 2
        load = randint(load-20, load+20)
        print([timestamp*1000, load])
        writer.writerow([timestamp*1000, load])
        idx = idx + 1
