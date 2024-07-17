import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('nn_descent_results.csv')

# Plot execution time vs data dimension
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(data['Dimension'], data['ExecutionTime'], marker='o', linestyle='-', color='b')
plt.title('Execution Time of NNDescent vs Data Dimension for 10k points')
plt.xlabel('Data Dimension')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)

# Plot correct percentage vs data dimension
plt.subplot(1, 2, 2)
plt.plot(data['Dimension'], data['CorrectPercentage'], marker='o', linestyle='-', color='g')
plt.title('Correct Neighbors Percentage vs Data Dimension for 10k points')
plt.xlabel('Data Dimension')
plt.ylabel('Correct Percentage (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('nn_descent_results_20NN_10K_K20_T20.png')  # Save the figure as a file
plt.show()