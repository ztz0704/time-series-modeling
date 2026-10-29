import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=2026)

sample_size = 120
time_index = np.arange(1, sample_size + 1)

true_frequency = 1 / 11
amplitude = 2.0
phase = 0.7
noise_std = 0.8

signal = amplitude * np.cos(
    2 * np.pi * true_frequency * time_index + phase
)

noise = rng.normal(
    loc=0.0,
    scale=noise_std,
    size=sample_size, 
)

series = signal + noise

centered_series = series - series.mean()

frequency_indices = np.arange(
    1,
    sample_size // 2 + 1,
)

fourier_frequencies = frequency_indices/ sample_size

rows = []

for frequency in fourier_frequencies:
    cosine_wave = np.cos(
        2 * np.pi * frequency * time_index
    )

    sine_wave = np.sin(
        2 * np.pi * frequency * time_index
    )

    a_f = (
        2 
        / sample_size
        * np.sum(centered_series * cosine_wave)
    )

    b_f = (
        2/
        sample_size
        * np.sum(centered_series * sine_wave)
    )

    intensity = (
        sample_size
        / 2
        * (a_f**2 + b_f**2)
    )

    rows.append(
        {
            'frequency': frequency,
            'period': 1 / frequency,
            'a_f': a_f,
            'b_f': b_f,
            'I_f': intensity,
        }
    )

periodogram_table = pd.DataFrame(rows)

highest_frequency_mask = np.isclose(
    periodogram_table['frequency'],
    0.5,
)

highest_frequency = 0.5

highest_cosine_wave = np.cos(
    2 * np.pi * highest_frequency * time_index
)

highest_a_f = (
    1
    / sample_size
    * np.sum(centered_series * highest_cosine_wave)
)  

highest_intensity = (
    sample_size 
    * highest_a_f**2
)

periodogram_table.loc[
    highest_frequency_mask,
    'a_f',
] = highest_a_f

periodogram_table.loc[
    highest_frequency_mask,
    'b_f',
] = 0.0

periodogram_table.loc[
    highest_frequency_mask,
    'I_f',
] = highest_intensity

total_sum_of_squares = np.sum(centered_series**2)

frequency_sum_of_squares = periodogram_table['I_f'].sum()

anova_check = pd.DataFrame(
    {
        "quantity": [
            "时间域总平方和",
            "频率强度之和",
            "两者之差",
        ],
        "value": [
            total_sum_of_squares,
            frequency_sum_of_squares,
            total_sum_of_squares
            - frequency_sum_of_squares,
        ],
    }
)

print("\n频域 ANOVA 验证：")
print(anova_check.to_string(index=False))

top_frequencies = (
    periodogram_table
    .sort_values(
        by='I_f',
        ascending=False,
    )
    .head(10)
    .reset_index(drop=True)
)

print("\n强度最大的 10 个频率：")
print(top_frequencies.to_string(index=False))

plt.figure(figsize=(10, 4))

plt.plot(
    time_index,
    series,
    label= 'Observed series',
)

plt.plot(
    time_index,
    signal,
    label='True periodic signal',
)

plt.xlabel('Time')
plt.ylabel('Value')
plt.title("Periodic signal plus white noise")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.stem(
    periodogram_table['frequency'],
    periodogram_table['I_f'],
    basefmt=" "
)

plt.axvline(
    true_frequency,
    linestyle='--',
    label=f'True frequency = {true_frequency: .4f}'
)

plt.xlabel("Frequency f")
plt.ylabel("Intensity I(f)")
plt.title("Box periodogram")
plt.legend()
plt.tight_layout()
plt.show()