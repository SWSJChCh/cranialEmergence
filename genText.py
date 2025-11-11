import numpy as np

#Generate 10 linearly spaced numbers between 1e2 and 1e3
values = np.linspace(1e2, 1e3, 10)

#Round to 3 decimal places
values = np.round(values, 3)

#VEGF degradation parameter
z = 3.25e2

#Open output file
with open("output.txt", "w") as f:
    for _ in range(20):
        for x in values:
            for y in values:
                f.write(f"{x:.3f}\t{y:.3f}\t{z:.3f}\n")

