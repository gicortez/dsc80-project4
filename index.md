# Predicting Spotify Track Popularity

**By Gian Cortez** &nbsp;|&nbsp; DSC 80, UCSD &nbsp;|&nbsp; Spring 2026

---

## Introduction

Spotify hosts over 100 million tracks, but the vast majority of them are barely listened to. What actually makes a song popular? Is it the tempo? How danceable it is? The genre? Or is popularity essentially unpredictable from the audio itself?

This project uses the **Spotify Music Tracks dataset**, which contains 114,000 tracks spanning 114 genres. Each track comes with a set of audio features computed directly from the audio signal by Spotify — things like `danceability`, `energy`, `acousticness`, and `tempo` — along with metadata like release date, genre, and whether the track is explicit.

The `popularity` column is Spotify's own 0–100 score measuring how frequently and recently a track has been streamed. A score of 0 means the track is essentially unheard; a score near 100 means it's one of the most-streamed songs on the platform.

I approach this in two parts. First, I investigate a specific question: **do explicit songs tend to be more popular than non-explicit ones?** Then I build a regression model to **predict a track's popularity score** from its audio features and metadata.

---

## Data Cleaning and Exploratory Data Analysis

The dataset required minimal cleaning. I dropped a leftover index column (`Unnamed: 0`) and standardized the `release_date` column, which mixed formats — some rows stored just a year (`"1974"`) while others stored a full date (`"1974-05-14"`). I extract a consistent `release_year` integer from both.

The popularity distribution immediately reveals something interesting: a massive spike at 0. Most songs on Spotify are essentially undiscovered — only a small fraction have any meaningful listener base.

<iframe src="assets/fig_popularity_dist.html" width="100%" height="450" frameborder="0"></iframe>

Looking at popularity split by explicit content, explicit songs sit noticeably higher in the distribution. The median and interquartile range are both shifted upward compared to non-explicit songs — a pattern worth testing more rigorously.

<iframe src="assets/fig_popularity_by_explicit.html" width="100%" height="450" frameborder="0"></iframe>

---

## Assessment of Missingness

The only column with meaningful missingness is `tempo`, which is absent for **22,114 rows — about 19.4% of the dataset**. Every other column has at most 1 missing value.

**Could tempo be NMAR?** Possibly. Spotify computes tempo algorithmically from the audio signal. Tracks with very complex or freeform rhythmic structure — like certain classical, ambient, or jazz pieces — may not yield a reliable BPM estimate, meaning the missingness could depend on the tempo value itself. Without access to Spotify's internal pipeline, we can't confirm this.

To test for MAR, I ran two permutation tests. The first tests whether tempo missingness depends on `track_genre`. The bar chart below shows that genres with less defined rhythm — romance, ambient, new-age, classical — have missingness rates above 30%, while metal genres are below 10%.

<iframe src="assets/fig_missingness_genre.html" width="100%" height="430" frameborder="0"></iframe>

A permutation test on the TVD (total variation distance) between the genre distributions of missing vs. non-missing rows gives an observed TVD of **0.179** with a **p-value of 0.0000**. Tempo missingness clearly depends on genre.

<iframe src="assets/fig_perm_genre.html" width="100%" height="430" frameborder="0"></iframe>

A second permutation test checks whether tempo missingness depends on `popularity`. The observed difference in mean popularity between missing and non-missing rows is only 0.06, and the two-sided p-value is **0.74** — no significant relationship. **Conclusion: tempo's missingness is MAR on `track_genre`.**

---

## Hypothesis Testing

**Question**: Do explicit songs have a higher mean popularity than non-explicit songs?

- **Null hypothesis**: Explicit and non-explicit songs have the same mean popularity; any observed difference is due to random chance.
- **Alternative hypothesis**: Explicit songs have a higher mean popularity than non-explicit songs.
- **Test statistic**: Difference in group means — mean popularity of explicit songs minus mean popularity of non-explicit songs.

I used a permutation test with 5,000 shuffles of the `explicit` label.

<iframe src="assets/fig_perm_explicit.html" width="100%" height="430" frameborder="0"></iframe>

The observed difference is **3.52 popularity points**. Not a single one of the 5,000 permutations produced a difference that large under the null — the p-value is effectively **0.0000**. I reject the null hypothesis.

Explicit songs do appear to have statistically higher mean popularity. That said, the effect size is small (~3.5 points on a 0–100 scale), so while the difference is real, it isn't dramatic on its own.

---

## Framing a Prediction Problem

The prediction task is to **predict a track's `popularity` score (0–100)**. This is a **regression problem** — popularity is a continuous numeric value, not a category.

I chose popularity as the response variable because it's the most direct measure of a song's commercial reach on Spotify, and understanding what drives it is the central question of the project.

For evaluation, I use **RMSE (root mean squared error)**. It's interpretable in the same units as popularity — an RMSE of 5 means I'm off by about 5 popularity points on average. RMSE also penalizes large errors more heavily than MAE, which matters here since drastically mispredicting a popular song is a more costly mistake.

All features I use — audio features, genre, release year, and explicit status — are either derived from the audio itself or are metadata known at the time a song is released. There's no data leakage.

---

## Baseline Model

For the baseline I used a simple **linear regression** on just two features: `danceability` and `energy`. These two are the most intuitive audio features for predicting popularity, and starting here gives a clear floor to beat.

| Metric | Baseline |
|--------|----------|
| R² | 0.0012 |
| RMSE | 22.202 |

The baseline is essentially useless — an R² of 0.001 means it explains almost none of the variance in popularity. This is barely better than predicting the mean popularity for every single song. Two raw audio features clearly aren't enough.

---

## Final Model

To improve on the baseline, I made several changes:

**More features**: I included all available numeric audio features (`loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `explicit`) plus `release_year`.

**Feature engineering**: I added `song_age` (2025 − release_year) and an interaction term `energy_dance` (energy × danceability), since high-energy, danceable songs might jointly capture something neither feature captures alone.

**One-hot encoding `track_genre`**: Genre is probably the single strongest predictor of popularity. A classical piece and a trap song follow very different popularity dynamics, and raw audio features alone can't capture that. I encoded all 114 genres.

**Ridge regression**: With 100+ genre dummy variables, OLS risks overfitting. Ridge adds L2 regularization (alpha = 10) to keep coefficients stable.

**Median imputation for `tempo`**: Instead of dropping the ~19% of rows with missing tempo, I impute with the column median inside the pipeline so there's no leakage.

| Metric | Baseline | Final Model |
|--------|----------|-------------|
| R² | 0.0012 | **0.2673** |
| RMSE | 22.202 | **19.015** |

The final model's R² jumped from 0.001 to 0.267 and RMSE dropped from 22.2 to 19.0 — a substantial improvement. Genre encoding did most of the heavy lifting.

<iframe src="assets/fig_actual_vs_pred.html" width="100%" height="480" frameborder="0"></iframe>

The scatter plot shows the model captures the general trend, though it underestimates very high-popularity tracks. This is expected — extreme popularity often reflects factors outside audio features entirely, like artist fame, social media virality, or playlist placement.

---

## Fairness Analysis

To check whether the model is equally good across different groups, I split the test set into **older songs (released before 2000)** and **newer songs (released 2000 or later)** and compared RMSE for each group.

- **Null hypothesis**: The model has the same RMSE for pre-2000 and post-2000 songs; any observed difference is due to chance.
- **Alternative hypothesis**: The model performs differently on the two groups.
- **Test statistic**: Difference in RMSE (RMSE_old − RMSE_new).

| Group | RMSE |
|-------|------|
| Pre-2000 (n = 12,459) | 18.130 |
| 2000 and later (n = 10,341) | 20.030 |

The model performs notably better on older songs. A permutation test with 1,000 shuffles gives an observed difference of **−1.90** with a **p-value of 0.0000**.

<iframe src="assets/fig_fairness_perm.html" width="100%" height="430" frameborder="0"></iframe>

The model is **not fair** with respect to release era. It predicts older songs significantly better than newer ones. This likely reflects the fact that older songs' popularity has had more time to stabilize — they've either become classics or faded into obscurity, making patterns more learnable. Newer songs are in a more volatile phase where popularity depends heavily on current cultural momentum, which audio features can't capture.

---

*DSC 80 Final Project — Spring 2026, UCSD*
