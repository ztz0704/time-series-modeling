import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import periodogram, welch

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

    total_size = (
        sample_size
        + burn_in 
        + order
    )

    innovations = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=total_size,
    )

    values = np.zeros(total_size)

    for time in range(
        order,
        total_size,
    ):
        autogressive_part = 0.0

        for lag, coefficient in enumerate(
            coefficients,
            start=1
        ):
            autogressive_part += (
                coefficient
                * values[time - lag]
            )

        values[time] = (
            autogressive_part
            + innovations[time]
        )

    return values[
        burn_in + order:
    ]

rng = np.random.default_rng(seed=2026)

sample_size = 1024

time_index = np.arange(
    1,
    sample_size + 1,
)

white_noise = rng.normal(
    loc=0.0,
    scale=1.0,
    size=sample_size,
)

positive_ar1 = simulate_ar(
    coefficients=[0.8],
    sample_size=sample_size,
    noise_std=1.0,
    rng=rng,
)

frequency_1 = 0.10
frequency_2 = 0.13

two_frequencies = (
    2.0
    * np.cos(
        2
        * np.pi
        * frequency_1
        * time_index
    )
    +
    1.5
    * np.cos(
        2
        * np.pi
        * frequency_2
        * time_index
        + 0.8
    )
    +
    rng.normal(
        loc=0.0,
        scale=1.0,
        size=sample_size,
    )
)

series_dictionary = {
    'White noise': white_noise,
    'AR(1), phi = 0.8': positive_ar1,
    'Two periodic signals': two_frequencies,
}

spectral_results = {}

for name, series in series_dictionary.items():
    raw_frequencies, raw_intensity = periodogram(
        series,
        fs=1.0,
        window='boxcar',
        detrend='constant',
        return_onesided=True,
        scaling='density',
    )

    short_frequencies, short_welch = welch(
        series,
        fs=1.0,
        window='hann',
        nperseg=128,
        noverlap=64,
        detrend='constant',
        return_onesided=True,
        scaling='density',
    )

    long_frequencies, long_welch = welch(
        series,
        fs=1.0,
        window="hann",
        nperseg=512,
        noverlap=256,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )

    spectral_results[name] = {
        "raw_frequencies": raw_frequencies,
        "raw_intensity": raw_intensity,
        "short_frequencies": short_frequencies,
        "short_welch": short_welch,
        "long_frequencies": long_frequencies,
        "long_welch": long_welch,        
    }

figure, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(11, 12),
    sharex=True,
)

for axis, name in zip(
    axes,
    series_dictionary,
):
    result = spectral_results[name]

    axis.plot(
        result["raw_frequencies"],
        result["raw_intensity"],
        linewidth=0.7,
        alpha=0.55,
        label="Raw periodogram I(f)",
    )

    axis.plot(
        result["short_frequencies"],
        result["short_welch"],
        linewidth=2.0,
        label="Welch: nperseg = 128",
    )

    axis.plot(
        result["long_frequencies"],
        result["long_welch"],
        linewidth=1.5,
        label="Welch: nperseg = 512",
    )

    if name == "White noise":
        axis.axhline(
            2.0,
            linestyle="--",
            label="Population spectrum p(f) = 2",
        )

    if name == "Two periodic signals":
        axis.axvline(
            frequency_1,
            linestyle="--",
            label="True frequency 0.10",
        )

        axis.axvline(
            frequency_2,
            linestyle=":",
            label="True frequency 0.13",
        )

    axis.set_title(name)
    axis.set_ylabel("Spectral intensity")
    axis.legend()

axes[-1].set_xlabel("Frequency f")

figure.suptitle(
    "Raw periodogram and Welch spectral estimates"
)

figure.tight_layout()
plt.show()