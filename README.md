
# Early Game Advantage in Professional League of Legends
### Gian Cortez

## Introduction

This project looks at professional League of Legends match data from 2025 to understand how early-game advantages relate to winning. In pro play, people often talk about how important the first 15 minutes are, so I wanted to study which early-game factors actually seem most connected to match outcomes.

The main question for this project is: **which early-game factors are most associated with winning in professional League of Legends?**

## Data Cleaning and Exploratory Data Analysis

To make the analysis consistent with the question, I focused on team-level rows only so that each row represents one team in one game. I also kept the columns most relevant to early-game advantage, such as gold difference at 15 minutes, XP difference at 15 minutes, CS difference at 15 minutes, first blood, first dragon, side, game length, and result.

From the exploratory analysis, gold difference at 15 stood out as one of the clearest signals. Teams with a positive gold lead at 15 minutes won much more often, and teams that secured early objectives like first blood or first dragon also tended to have higher win rates.

## Assessment of Missingness

For missingness, I looked at `goldat25`, which had meaningful missing values. I do not think this column is MNAR, because the missingness can be reasonably explained by the structure of the games themselves. If a game ends before 25 minutes, then there is no value to record at that time.

My missingness analysis showed that the missingness of `goldat25` depends on `gamelength`, since shorter games are much more likely to have missing values. On the other hand, missingness did not appear to depend on side. This suggests the missingness is more consistent with MAR than MCAR.

## Hypothesis Testing

For Step 4, I tested whether Blue side has a higher win rate than Red side.

- **Null hypothesis:** Blue side and Red side have the same win rate, and any observed difference is due to random chance.
- **Alternative hypothesis:** Blue side has a higher win rate than Red side.

I used the difference in win rates, `winrate(Blue) - winrate(Red)`, as the test statistic and performed a permutation test. The observed difference was about 0.06, and the p-value was less than 0.001. Since this is below 0.05, I rejected the null hypothesis and concluded that Blue side has a statistically higher win rate in this dataset.

## Framing a Prediction Problem

I framed this as a **binary classification** problem. The goal is to predict whether a team wins or loses, so the response variable is `result`.

I used only information available by 15 minutes, since the main idea of the project is to study early-game advantage. This helps avoid data leakage from later-game information. I evaluated the models using ROC-AUC and accuracy, since ROC-AUC gives a good overall measure of classification performance and accuracy is easy to interpret.

## Baseline Model

My baseline model is a logistic regression model that predicts whether a team wins using simple early-game features such as `golddiffat15` and `xpdiffat15`. I chose these because they are straightforward measures of which team is ahead by 15 minutes.

The baseline model performed reasonably well, with a test ROC-AUC of about 0.81 and a test accuracy of about 73%. This suggests that even a simple model using only a small amount of early-game information can already predict match outcomes fairly well.

## Final Model

To improve the baseline model, I added more early-game features that still fit the 15-minute prediction setting, such as `csdiffat15`, `firstblood`, `firstdragon`, and side. I also created new features like gold per minute and XP per minute, and I tested different `C` values in logistic regression to find the version that worked best.

The final model improved on the baseline while still staying interpretable. Gold difference at 15 remained the strongest predictor, which makes sense because gold lead captures a lot of overall early-game pressure. Other features, such as first dragon and CS difference, also helped explain match outcomes.

## Fairness Analysis

For fairness, I compared how well the final model performed for Blue side teams versus Red side teams.

- **Group X:** Blue side teams
- **Group Y:** Red side teams

I used test-set accuracy as the evaluation metric and performed a permutation test on the difference in accuracy between the two groups. The observed difference was small, and the permutation test did not show strong evidence that the model performed substantially worse for one side than the other.

Overall, this suggests that the final model does not appear to unfairly favor Blue side or Red side in terms of prediction accuracy.
