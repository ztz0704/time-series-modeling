# Box–Jenkins Series G: Day 1 Replication

## 1. Data

- Number of observations:
- Start date:
- End date:
- Frequency:
- Unit:

## 2. Raw-series evidence

Describe:

- trend;
- seasonality;
- changing seasonal amplitude;
- stationarity or nonstationarity.

## 3. Log transformation

Define:

Z_t = log(Y_t)

Explain:

- why the logarithm is considered;
- which feature is stabilized;
- which nonstationary features remain.

## 4. Baseline results

| model | MAE | RMSE | MASE |
|---|---:|---:|---:|
| seasonal-naive | | | |
| naive | | | |
| mean | | | |

## 5. Interpretation

### Why seasonal-naive is the best baseline

Series G exhibits strong annual seasonality. Observations from the same month in adjacent years tend to occupy similar positions within the seasonal cycle. For example, summer months tend to be relatively high, whereas observations near the beginning of the year tend to be lower.

The seasonal-naive forecast is defined by

$$
\widehat{Y}_{T+h\mid T}
=======================

Y_{T+h-12k},
$$

where the observation on the right-hand side is the most recently observed value from the same month of the year. In this application, the model uses the previous year's value for a given month to forecast the corresponding month in the next year.

Seasonal-naive therefore preserves the annual seasonal pattern of the series. In contrast:

* the mean forecast assigns the same historical average to every future month and captures neither trend nor seasonality;
* the ordinary naive forecast assigns the final training observation to every future period and therefore ignores differences across calendar months;
* the seasonal-naive forecast distinguishes between months and reproduces the most recently observed annual seasonal profile.

Because annual seasonality is a dominant feature of Series G, seasonal-naive produces lower MAE, RMSE, and MASE than the mean and ordinary naive forecasts. It is therefore the strongest of the three baseline models.

### Why seasonal-naive still systematically underpredicts

Although seasonal-naive captures the shape of the seasonal pattern, it implicitly assumes that the overall annual level of the series does not continue to change systematically. Its underlying benchmark can be written approximately as

$$
Y_t
===

Y_{t-12}
+
\varepsilon_t.
$$

Series G, however, contains a pronounced upward movement in addition to annual seasonality. Passenger counts in most months of 1959 and 1960 are higher than the corresponding observations in 1958.

The seasonal-naive model merely copies the monthly values from the last observed year into the forecast period. For example,

$$
\widehat{Y}_{1959,m}
====================

Y_{1958,m}.
$$

It does not extrapolate the continuing increase in the level of the series. Consequently, for most months,

$$
Y_{1959,m}

>

\widehat{Y}_{1959,m}.
$$

The discrepancy generally becomes larger in 1960.

The forecast errors therefore do not fluctuate randomly around zero. Instead, many satisfy

$$
Y_t-\widehat{Y}_t>0.
$$

This pattern indicates systematic underprediction. The seasonal-naive model captures seasonality but fails to represent the persistent increase in the level of the series.

### Why regular and seasonal differencing should be investigated next

The original series contains at least two important nonstationary features:

1. a persistent change in the long-run level;
2. a recurring seasonal pattern with a period of 12 months.

The regular first difference is defined as

$$
\Delta Y_t
=

Y_t-Y_{t-1},
$$

or, on the logarithmic scale,

$$
\Delta\log Y_t
=

\log Y_t-\log Y_{t-1}.
$$

Regular differencing is intended to remove or reduce nonstationarity associated with persistent changes in the level of the series.

The seasonal difference is defined as

$$
\Delta_{12}Y_t
=

Y_t-Y_{t-12},
$$

or, on the logarithmic scale,

$$
\Delta_{12}\log Y_t
=

\log Y_t-\log Y_{t-12}.
$$

Seasonal differencing compares observations from the same month in adjacent years and is intended to remove or reduce the recurring annual seasonal structure.

For Series G, the combined transformation

$$
W_t
=

(1-B)(1-B^{12})\log Y_t
$$

should also be investigated, where $B$ denotes the backshift operator and satisfies

$$
BY_t
=

Y_{t-1}.
$$

Expanding the expression gives

$$
W_t
=

\log Y_t

- \log Y_{t-1}

-\log Y_{t-12}
+
\log Y_{t-13}.
$$

These transformations address different features of the original data:

* the logarithm stabilizes fluctuations whose magnitude increases with the series level;
* the regular difference addresses persistent changes in the level;
* the seasonal difference addresses the annual seasonal pattern.

However, the use of all three transformations should not be imposed automatically. Their necessity should be evaluated using the transformed time-series plots, the ACF and PACF, and subsequent residual diagnostics.

## 6. Day 1 Conclusion

The original Series G data do not appear to be covariance stationary.

First, the level of the series increases substantially over time, so the mean cannot reasonably be treated as constant. Second, the series exhibits a strong 12-month seasonal pattern, indicating systematic differences across calendar months. Third, the absolute magnitude of the seasonal fluctuations increases as the level of the series rises, suggesting that the variance may also be nonconstant.

The original series therefore exhibits:

* a changing level over time;
* strong annual seasonal structure;
* possible level-dependent variability.

However, the raw plot alone cannot determine whether the long-run movement is generated by a deterministic trend or a unit root. It also cannot establish whether the seasonal behavior is caused by deterministic seasonality, stationary seasonal dependence, or a seasonal unit root.

The next stage should first examine the logarithmic transformation

$$
Z_t
=

\log Y_t,
$$

which may stabilize the seasonal variation as the series level increases. The following transformed series should then be studied.

The regular first difference is

$$
\Delta Z_t
=
(1-B)Z_t=

Z_t-Z_{t-1}.
$$

The seasonal difference is

$$
\Delta_{12}Z_t
=

 (1-B^{12})Z_t

=Z_t-Z_{t-12}.
$$

The combined regular and seasonal difference is

$$
(1-B)(1-B^{12})Z_t.
$$

The objective of the next stage is not to specify a SARIMA model immediately. It is first necessary to determine whether an appropriate transformation, deterministic detrending, or differencing operation produces an approximately stationary series.

Further analysis should determine:

* whether the long-run movement is a deterministic trend or an ordinary unit root;
* whether the seasonal pattern is deterministic, stationary seasonal dependence, or a seasonal unit root;
* whether regular and seasonal differencing are actually required.

Only then should the ACF, PACF, parameter estimates, and residual diagnostics be used to identify a suitable ARMA or seasonal ARIMA structure.
