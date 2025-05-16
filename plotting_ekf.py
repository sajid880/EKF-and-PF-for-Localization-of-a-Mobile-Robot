import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Define test configurations
scale_factors = [1/64, 1/16, 1/4, 1, 4, 16, 64]
num_trials = 10

# Run a single EKF localization with specified noise settings
def run_localization_trial(data_noise, filter_noise, seed_value):
    command = [
        'python', 'localization.py', 'ekf',
        '--data-factor', str(data_noise),
        '--filter-factor', str(filter_noise),
        '--seed', str(seed_value)
    ]
    
    process = subprocess.run(command, capture_output=True, text=True)
    output = process.stdout

    try:
        pos_error = float(output.split("Mean position error:")[1].split("\n")[0].strip())
        anees_score = float(output.split("ANEES:")[1].strip())
        return pos_error, anees_score
    except Exception as e:
        print("Error while parsing output:", e)
        print(output)
        return None, None

# Conduct experiments across different noise scales
def evaluate_filter_performance(lock_data_noise=True):
    all_errors = []
    all_anees = []

    for r in scale_factors:
        trial_errors = []
        trial_anees = []
        print(f"\nRunning tests with scale factor r = {r}")

        for trial_id in range(num_trials):
            data_var = 1 if lock_data_noise else r
            filter_var = r

            err, anees = run_localization_trial(data_var, filter_var, trial_id)
            if err is not None:
                trial_errors.append(err)
                trial_anees.append(anees)

        all_errors.append(np.mean(trial_errors))
        all_anees.append(np.mean(trial_anees))

    return all_errors, all_anees

# ---- Main Experiment Flow ----
print("Experiment 3(b): Changing both input and filter noise")
error_b, anees_b = evaluate_filter_performance(lock_data_noise=False)

print("\nExperiment 3(c): Fixing input noise, changing only filter noise")
error_c, anees_c = evaluate_filter_performance(lock_data_noise=True)

# ---- Visualization ----
plt.figure(figsize=(12, 5))

# Subplot for mean position error
plt.subplot(1, 2, 1)
plt.plot(scale_factors, error_b, 'o-', label='Exp 3(b): Input & Filter')
plt.plot(scale_factors, error_c, 's--', label='Exp 3(c): Filter Only')
plt.xlabel('Noise Scaling (r)')
plt.ylabel('Mean Position Error')
plt.title('Effect of Noise Scale on Position Error')
plt.legend()
plt.show()

# Subplot for ANEES values
plt.subplot(1, 2, 2)
plt.plot(scale_factors, anees_b, 'o-', label='Exp 3(b): Input & Filter')
plt.plot(scale_factors, anees_c, 's--', label='Exp 3(c): Filter Only')
plt.xlabel('Noise Scaling (r)')
plt.ylabel('ANEES')
plt.title('Effect of Noise Scale on ANEES')
plt.legend()

plt.tight_layout()
plt.show()
