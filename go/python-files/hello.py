# Write to a file in a new directory called "output"
import os

os.makedirs("output", exist_ok=True)

with open("output/hello-from-python.txt", "w") as f:
    f.write("Hello from Python in Dagger!\n")