import time
import psutil
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

# Initialize lists to store timestamp and memory values
timestamps = []
memory_values = []

# Create the plot
plt.xlabel('Time')
plt.ylabel('Free Memory (MB)')
plt.title('Free Memory Monitoring')

# Continuous plot update
i = 0
while True:
    # Get current timestamp and free memory
    timestamp = time.strftime('%H:%M:%S')
    memory = psutil.virtual_memory().available / (1024 * 1024)  # in MB

    # Add values to lists
    timestamps.append(timestamp)
    memory_values.append(memory)

    # Update the plot
    plt.plot(timestamps, memory_values, color='blue')
    plt.draw()
    plt.pause(2)  # Pause for 2 seconds
    print(i)
    i += 1
    if i==2: break

# Keep the plot window open
plt.show()