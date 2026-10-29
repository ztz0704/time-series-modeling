import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def box_periodogram(series):

    values = np.asarray(series, dtype=float)

    if values.ndim != 1:
        raise ValueError("series 必须是一维数组。")

    if len(values) < 3:
        raise ValueError("series 至少需要包含 3 个观测。")

    sample_size = len(values)

    centered_values = values - values.mean()

    time_index = np.arange(
        1,
        sample_size + 1,
    )

    frequency_indices = np.arange(
        1,
        sample_size // 2 + 1,
    )

    rows = []

    for frequency_index in frequency_indices:
        frequency = frequency_index / sample_size

        cosine_wave = np.cos(
            2 * np.pi * frequency * time_index
        )

        sine_wave = np.sin(
            2 * np.pi * frequency * time_index
        )

        is_highest_frequency = (
            sample_size % 2 ==0
            and frequency_index == sample_size // 2
        )

        if is_highest_frequency:
            a_f = (
                1
                / sample_size
                * np.sum(centered_values * cosine_wave)
            )

            b_f = 0.0

            intensity = sample_size * a_f ** 2

        else:
            a_f = (
                2
                / sample_size
                * np.sum(centered_values * cosine_wave)
            )

            b_f = (
                2
                / sample_size
                * np.sum(centered_values * sine_wave)
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

    time_domain_sum_of_squares = np.sum(
        centered_values**2
    )

    frequency_domain_sum_of_squares = (
        periodogram_table['I_f'].sum()
    )

    anova_error = (
        time_domain_sum_of_squares
        - frequency_domain_sum_of_squares
    )

    return periodogram_table, anova_error

def simulate_ar(
        coefficients,
        sample_size,
        noise_std,
        rng,
        burn_in=1000,
):
    coefficients = np.asarray(
        coefficients,
        dtype=float,
    )

    order = len(coefficients)

    total_size = sample_size + burn_in + order

    innovations = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=total_size,
    )

    values = np.zeros(total_size)

    for time in range(order, total_size):
        autoregressive_part = 0.0

        for lag, coefficient in enumerate(
            coefficients,
            start=1,
        ):
            autoregressive_part += (
                coefficient * values[time - lag]
            )   

        values[time] = (
            autoregressive_part
            + innovations[time]
        )

    return values[burn_in + order:]

def box_ar_spectrum(
    frequencies,
    coefficients,
    noise_variance,
):
    frequencies = np.asarray(
        frequencies,
        dtype=float,
    )

    coefficients = np.asarray(
        coefficients,
        dtype=float,
    )

    ar_polynomial = np.ones(
        len(frequencies),
        dtype=complex,
    )

    for lag, coefficient in enumerate(
        coefficients,
        start=1,
    ):
        ar_polynomial -= (
            coefficient
            * np.exp(
                -2j
                * np.pi
                * lag
                * frequencies
            )
        )

    spectrum = (
        2
        * noise_variance
        / np.abs(ar_polynomial) ** 2
    )

    return spectrum

rng = np.random.default_rng(seed=2026)

sample_size = 400
noise_std = 1.0
noise_variance = noise_std**2

time_index = np.arange(
    1,
    sample_size + 1,
)

white_noise = rng.normal(
    loc=0.0,
    scale=noise_std,
    size=sample_size,
)

positive_ar1_coefficient =0.8

positive_ar1 = simulate_ar(
    coefficients=[positive_ar1_coefficient],
    sample_size=sample_size,
    noise_std=noise_std,
    rng=rng,
)

negative_ar1_coefficient = -0.8

negative_ar1 = simulate_ar(
    coefficients=[negative_ar1_coefficient],
    sample_size=sample_size,
    noise_std=noise_std,
    rng=rng,
)

ar2_coefficients = [1.4, -0.8]

oscillating_ar2 = simulate_ar(
    coefficients=ar2_coefficients,
    sample_size=sample_size,
    noise_std=noise_std,
    rng=rng,
)

true_frequency = 1 / 12
amplitude = 2.0
phase = 0.7

periodic_signal = amplitude * np.cos(
    2 
    * np.pi
    * true_frequency
    * time_index
    + phase
)

sine_plus_noise = (
    periodic_signal
    + rng.normal(
        loc=0.0,
        scale=noise_std,
        size=sample_size,
    )
)

series_dictionary = {
    "White noise": {
        "series": white_noise,
        "coefficients": [],
    },
    "AR(1), phi = 0.8": {
        "series": positive_ar1,
        "coefficients": [0.8],
    },
    "AR(1), phi = -0.8": {
        "series": negative_ar1,
        "coefficients": [-0.8],
    },
    "Oscillating AR(2)": {
        "series": oscillating_ar2,
        "coefficients": ar2_coefficients,
    },
    "Sine plus noise": {
        "series": sine_plus_noise,
        "coefficients": None,
    },
}

periodogram_results = {}
summary_rows = []

for name, information in series_dictionary.items():
    table, anova_error = box_periodogram(
        information['series']
    )

    periodogram_results[name] = table

    strongest_row = (
        table
        .sort_values(
            by='I_f',
            ascending=False,
        )
        .iloc[0]
    )

    summary_rows.append(
        {
            'process': name,
            'peak_frequency': strongest_row[
                'frequency'
            ],
            'peak_period': strongest_row[
                'period'
            ],
            "peak_intensity": strongest_row[
                "I_f"
            ],
            "anova_error": anova_error,
        }
    )

summary_table = pd.DataFrame(summary_rows)

print("\n各过程的最强样本频率：")
print(    
    summary_table.to_string(
        index=False,
    )
)

figure, axes = plt.subplots(
    nrows=5,
    ncols=1,
    figsize=(11, 12),
    sharex=True,
)

for axis, (name, information) in zip(
    axes,
    series_dictionary.items(),
):
    axis.plot(
        time_index[:120],
        information['series'][:120],
    )

    axis.set_title(name)
    axis.set_ylabel(r'$z_t$')

axes[-1].set_xlabel('Time')

figure.suptitle(
    'time-domain paths',
)

figure.tight_layout()
plt.show()

figure, axes = plt.subplots(
    nrows=5,
    ncols=1,
    figsize=(11, 14),
    sharex=True,
)

for axis, (name, information) in zip(
    axes,
    series_dictionary.items(),
):
    table = periodogram_results[name]

    frequencies = table['frequency'].to_numpy()

    axis.plot(
        frequencies,
        table['I_f'],
        linewidth=0.8,
        label='Sample periodogram I(f)',
    )

    coefficients = information['coefficients']

    if coefficients is not None:
        theoretical_spectrum = box_ar_spectrum(
            frequencies=frequencies,
            coefficients=coefficients,
            noise_variance=noise_variance,
        )

        axis.plot(
            frequencies,
            theoretical_spectrum,
            linewidth=2.0,
            label="Population spectrum p(f)",
        )

    else:
        axis.axhline(
            2 * noise_variance,
            linestyle='--',
            label='White-noise background',
        )    

        axis.axvline(
            true_frequency,
            linestyle='--',
            label='Spectral line f = 1/12',
        )

    axis.set_title(name)
    axis.set_ylabel('Intensity')
    axis.legend()

axes[-1].set_xlabel("Frequency f")

figure.suptitle(
    "Box periodograms and population spectra",
)

figure.tight_layout()
plt.show()



