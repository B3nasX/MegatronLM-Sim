import matplotlib.pyplot as plt
import numpy as np

# Data from validation
pp_stages = [1, 2, 4, 8]
# Simulated results
sim_bs128 = [177.57, 172.46, 168.46, 162.74]
sim_bs8 = [177.02, 155.92, 126.18, 92.63]

# Estimated Values
actual_bs128 = [176, 170, 167, 162]
actual_bs8 = [165, 143, 122, 88]

plt.figure(figsize=(10, 6))

# Plotting Batch Size 128
plt.plot(pp_stages, sim_bs128, 'o-', label='Simulated (BS=128)', color='blue', linewidth=2)
plt.plot(pp_stages, actual_bs128, 'o--', label='Megatron-LM Actual (BS=128)', color='blue', alpha=0.4)

# Plotting Batch Size 13
plt.plot(pp_stages, sim_bs8, 's-', label='Simulated (BS=8)', color='red', linewidth=2)
plt.plot(pp_stages, actual_bs8, 's--', label='Megatron-LM Actual (BS=8)', color='red', alpha=0.4)

plt.xlabel('Pipeline Parallel Stages', fontsize=12)
plt.ylabel('Effective TFLOPS per GPU', fontsize=12)
plt.title('Validation: Simulated vs. Actual Megatron-LM Throughput', fontsize=14)
plt.xticks(pp_stages)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

# Calculate % similarity for both batch sizes 
similarity_bs128 = [100 * (1 - abs(sim - actual) / actual) for sim, actual in zip(sim_bs128, actual_bs128)]
similarity_bs8 = [100 * (1 - abs(sim - actual) / actual) for sim, actual in zip(sim_bs8, actual_bs8)]
print("Similarity for Batch Size 128:", similarity_bs128)
print("Similarity for Batch Size 8:", similarity_bs8)
print("Average Similarity for Batch Size 128:", np.mean(similarity_bs128))
print("Average Similarity for Batch Size 8:", np.mean(similarity_bs8))
print("Overall Average Similarity:", np.mean(similarity_bs128 + similarity_bs8))
plt.tight_layout()
plt.show()
