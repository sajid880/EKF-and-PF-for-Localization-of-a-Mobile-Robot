import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Experiment configuration
noise_scales = [1/64, 1/16, 1/4, 1, 4, 16, 64]
seeds_to_test = 10
particle_variants = [20, 50, 500]  # For Section 4(d)

# Executes one particle filter run
def run_particle_filter(data_noise, filter_noise, seed, num_particles=100):
    args = [
        'python', 'localization.py', 'pf',
        '--data-factor', str(data_noise),
        '--filter-factor', str(filter_noise),
        '--seed', str(seed),
        '--num-particles', str(num_particles)
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    stdout = result.stdout
    try:
        error = float(stdout.split("Mean position error:")[1].split("\n")[0].strip())
        anees = float(stdout.split("ANEES:")[1].strip())
        return error, anees
    except Exception as e:
        print("Could not extract results:", e)
        print(stdout)
        return None, None

# Perform multiple runs across scale values
def run_experiment_set(fix_input_noise=True, num_particles=100):
    mean_errors = []
    mean_anees = []

    for r in noise_scales:
        collected_errors = []
        collected_anees = []
        print(f"\nRunning tests with scale factor r = {r} with {num_particles} particles")
        for i in range(seeds_to_test):
            data_factor = 1 if fix_input_noise else r
            filter_factor = r
            err, anees = run_particle_filter(data_factor, filter_factor, i, num_particles)
            if err is not None:
                collected_errors.append(err)
                collected_anees.append(anees)
        mean_errors.append(np.mean(collected_errors))
        mean_anees.append(np.mean(collected_anees))

    return mean_errors, mean_anees

# ---------------- Execute Experiments ----------------
print("Section 4(b): Varying both data and filter noise levels")
errors_b, anees_b = run_experiment_set(fix_input_noise=False)

print("\nSection 4(c): Fixed data noise, varying filter noise")
errors_c, anees_c = run_experiment_set(fix_input_noise=True)

# Section 4(d): Particle count analysis
error_trends_by_particles = {}
anees_trends_by_particles = {}

for p in particle_variants:
    print(f"\nSection 4(d): Using {p} particles with variable r")
    e_vals, a_vals = run_experiment_set(fix_input_noise=True, num_particles=p)
    error_trends_by_particles[p] = e_vals
    anees_trends_by_particles[p] = a_vals

# ---------------- Visualization ----------------
plt.figure(figsize=(15, 5))

# Subplot 1: Position Error comparison
plt.subplot(1, 3, 1)
plt.plot(noise_scales, errors_b, 'o-', label='4(b): Vary Data & Filter')
plt.plot(noise_scales, errors_c, 's--', label='4(c): Filter Only')
plt.xscale('log')
plt.xlabel('Scaling Factor (r)')
plt.ylabel('Avg. Position Error')
plt.title('Position Error Across Noise Scales')
plt.legend()
plt.show()

# Subplot 2: ANEES comparison
plt.subplot(1, 3, 2)
plt.plot(noise_scales, anees_b, 'o-', label='4(b): Vary Data & Filter')
plt.plot(noise_scales, anees_c, 's--', label='4(c): Filter Only')
plt.xscale('log')
plt.xlabel('Scaling Factor (r)')
plt.ylabel('ANEES')
plt.title('ANEES vs Noise Scaling')
plt.legend()
plt.show()


# Subplot 3: Effect of particle count
plt.subplot(1, 3, 3)
for particle_count in particle_variants:
    plt.plot(noise_scales, error_trends_by_particles[particle_count], label=f'{particle_count} particles')
plt.xscale('log')
plt.xlabel('Scaling Factor (r)')
plt.ylabel('Avg. Position Error')
plt.title('Impact of Particle Count on Error')
plt.legend()

plt.tight_layout()
plt.show()
