**Rainfall Frequency Analysis**

Extreme rainfall frequency analysis is a cornerstone of hydrological design, particularly for infrastructure such as dams, embankments, drainage systems, and urban stormwater networks. Several statistical methods are commonly employed to estimate design rainfall associated with specified return periods. These methods differ in their assumptions regarding distributional form, tail behavior, data requirements, and suitability for extrapolation. The following provides a comparative overview of four key approaches.

<img width="1979" height="1180" alt="image" src="https://github.com/user-attachments/assets/68f8069c-20ac-4318-9b0a-83599b989cef" />

<img width="1980" height="1180" alt="image" src="https://github.com/user-attachments/assets/68e832f4-981a-47b9-a642-252f7a2c98d0" />



**Gumbel Distribution (Extreme Value Type I, EV1)**

Concept and Characteristics
The Gumbel distribution represents the simplest member of the extreme value family. It is particularly designed to model annual maxima (block maxima approach), such as the largest rainfall depth recorded in each year. The probability of exceedance is derived using a closed-form cumulative distribution function (CDF):


Strengths

Simplicity and widespread use in hydrological engineering practice.

Analytical tractability of return period calculations.

Reasonable performance for moderate extremes (e.g., 10–25 year events).

Limitations

Fixed light-tailed behavior due to the absence of a shape parameter.

Tendency to underestimate very rare extremes (e.g., 100-year or 500-year rainfall).

May not capture regional variability in rainfall extremes adequately.

Recommended Applications

Gumbel (EV1) remains the practical choice in engineering due to its simplicity, though its fixed tail behavior can be problematic for extrapolating very rare events.



**Generalized Extreme Value (GEV) Distribution**

Concept and Characteristics
The GEV distribution is a unified framework encompassing three subtypes: Gumbel (Type I), Fréchet (heavy-tailed, Type II), and Weibull (bounded upper tail, Type III). Its probability distribution function includes three parameters—location (
𝜇
μ), scale (
𝜎
σ), and shape (
𝜉
ξ):


Strengths

Flexibility to represent light, heavy, or bounded tails.

Suitable for capturing rare, extreme events more accurately than Gumbel.

Increasingly favored in climate change studies where non-stationarity and extremes matter.

Limitations

Estimation of the shape parameter is statistically challenging, particularly for short data records.

Requires long-term, high-quality datasets for robust fitting.

Recommended Applications

GEV offers the most flexibility and is increasingly regarded as a robust framework for extreme value analysis, especially under non-stationary climate conditions.



**Log-Pearson Type III (LP3) Distribution**

Concept and Characteristics
Widely used in flood and rainfall frequency studies in the United States (mandated by the US Water Resources Council guidelines), the LP3 distribution assumes that the logarithms of the annual maxima follow a Pearson Type III distribution. Parameters are derived from the log-transformed series: mean, standard deviation, and skewness.

Strengths

Handles skewed data effectively, which is common in hydrological extremes.

Well-established regulatory standard in North America.

Familiar to practitioners and supported by extensive hydrological literature.

Limitations

Assumes positive data values and suitability for log-transformation.

Sensitive to skew coefficient estimates, which can be unstable for short datasets.

Less commonly applied outside the US regulatory framework.

Recommended Applications

Log-Pearson Type III retains relevance in regulatory environments where skewed flood or rainfall distributions must be considered.



**Weibull Plotting Position / Empirical Method**

Concept and Characteristics
The Weibull formula is a non-parametric approach that avoids the assumption of a specific theoretical distribution. The empirical exceedance probability is assigned based on rank:


Strengths

Straightforward and intuitive, requiring no distribution fitting.

Useful for exploratory analysis and visualization.

Applicable even with short data series.

Limitations

Reliability limited by dataset length; extrapolation to rare return periods (e.g., 100 years) is statistically weak.

Provides descriptive rather than inferential power.

Does not capture underlying stochastic structure of extreme events.

Recommended Applications

Weibull plotting positions provide a simple, empirical perspective but are unsuitable for reliable long-term design projections.


**Key Insights**

| **Method**                      | **Type**       | **Tail Flexibility**      | **Data Requirement**        | **Best Use Case**                                     |
| ------------------------------- | -------------- | ------------------------- | --------------------------- | ----------------------------------------------------- |
| Gumbel (EV1)                    | Parametric     | Fixed (light)             | Moderate                    | Standard hydrological design; moderate return periods |
| Generalized Extreme Value (GEV) | Parametric     | Flexible (light to heavy) | Long record needed          | Rare/extreme events; climate impact studies           |
| Log-Pearson Type III            | Parametric/log | Moderate (skewness)       | Moderate                    | Skewed flood/rainfall data; US regulatory practice    |
| Weibull / Empirical             | Non-parametric | Dataset-dependent         | Any length (short possible) | Quick, data-driven estimates; exploratory analysis    |


| Method                           | Most accurate when…                                                                                                                                       | Less reliable when…                                                                                |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Weibull (empirical plotting)** | You have **short records** and want **distribution-free visualization**; checking consistency of fits; **bridging** to regional methods.                  | **Extrapolating** beyond data range (e.g., 50–100-yr from 15–20 yrs).                              |
| **Gumbel (EV1)**                 | Tails are **light**, return periods are **modest** (≈10–25 yr); you need a **simple, stable** design number.                                              | **Heavy-tailed** climates; design for **rare extremes** (≥50–100 yr) → tends to **underestimate**. |
| **GEV**                          | You must model **tail behavior** (light/heavy/bounded); have **≥25–30 yrs** (ideally ≥40) or can **pool regionally**; need **non-stationary** capability. | Very **short series** with unstable $\xi$ estimates; poor pooling.                                 |
| **Log-Pearson III (LP3)**        | Data are **strongly skewed**; regulatory environments; **consistent** log-skew across sites.                                                              | When log-skew is poorly estimated (short N), zeros/negatives, or **non-stationarity** dominates.   |



**Method selection matrix**

| Situation                                   | Primary method                       | Cross-checks                                | Notes                                           |
| ------------------------------------------- | ------------------------------------ | ------------------------------------------- | ----------------------------------------------- |
| **Urban drainage (10–25 yr), limited data** | GEV (L-moments) **or** Gumbel        | Choose better tail diagnostics; report CI   | If similar, Gumbel is fine.                     |
| **Critical infrastructure (≥50–100 yr)**    | **GEV**                              | LP3 if skew strong                          | Always give CIs + sensitivity.                  |
| **Short record (<20–25 yr)**                | Weibull points + **GEV (L-moments)** | Regional pooling                            | Avoid hard claims beyond \~50 yr unless pooled. |
| **Regulated (LP3 required)**                | **LP3**                              | GEV cross-check                             | Include uncertainty to de-risk.                 |
| **Monsoon climates (e.g., Bangladesh)**     | **GEV (L-moments)**                  | Gumbel for quick checks; LP3 if strong skew | Expect occasional heavy tails.                  |


**Red-flag checklist**

Selecting Gumbel by habit when tail is heavy → underestimation of rare events.

Using LP3 with unstable log-skew from short records → volatile design depths.

Blind extrapolation (100-yr from 12–15 yrs) without RFA/uncertainty.

Ignoring non-stationarity → biased, time-inconsistent designs.

Reporting a single number without CIs or model sensitivity table.

