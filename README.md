# Logistic regression vs. neural networks for heart disease prediction

Can a neural network beat plain logistic regression on a small clinical dataset? This project tests that question in MATLAB on the Heart Failure Prediction dataset: 918 patients, 11 clinical predictors, and a yes/no heart disease outcome.

The short answer is not reliably. A tuned network matched or slightly beat logistic regression, and its one real edge was recall: it caught a few more true cases of heart disease.

The full write-up, with background, methods, tuning and discussion, is in the [report (PDF)](report/ANN-vs-Log-Reg-Report.pdf).

## The data

`data/heart.csv` holds 918 records pooled from five patient cohorts. The 11 predictors are age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression (`Oldpeak`) and ST slope. There are no missing values, and the two classes are close to balanced (about 55/45), so accuracy stays meaningful without resampling.

## How the comparison works

1. Categorical columns are one-hot encoded and continuous ones are standardized.
2. The data is split 70/15/15 into training, validation and test sets, and every model sees the same splits.
3. Logistic regression is the baseline.
4. Nine network shapes, from a single hidden layer of 10 neurons up to three layers of 40, 20 and 10, are screened with 5-fold cross-validation.
5. The four strongest are tuned over learning rate and number of training epochs, using the validation set only.
6. Each finalist is run 10 times against logistic regression on identical splits, and the averages are compared.

Every model is scored on accuracy, precision, recall and F1. Accuracy alone would hide the trade-off that matters here: a missed case of heart disease costs more than a false alarm.

## Results

Averages over 10 runs, on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Network, layers [40 20 10] | 0.8708 | 0.8791 | 0.8908 | 0.8841 |
| Logistic regression, same runs | 0.8701 | 0.8818 | 0.8855 | 0.8825 |

The best single run was the [40 20 10] network at a learning rate of 0.01 for 150 epochs: 90.51% accuracy and 93.42% recall, with 5 missed cases against 10 for logistic regression's best run.

A few things stood out:

- **The margins are small.** The largest accuracy gain is about half a percentage point, on a test set of roughly 137 patients.
- **Depth didn't decide it.** The deepest network did best, but the two-layer [20 10] network lost to logistic regression on all four metrics.
- **Single runs mislead.** One architecture scored 91.97% in a single run and finished mid-field under cross-validation, which is why the final comparison uses 10-run averages.

## Running it

You'll need MATLAB with the Statistics and Machine Learning Toolbox (for `fitglm`) and the Deep Learning Toolbox (for `patternnet`).

The scripts load `heart.csv` by name, so set MATLAB's current folder to `data/` first, or copy the CSV next to the script you're running.

| Script | What it does |
| --- | --- |
| `src/v1.m` | Explores the data, preprocesses it, and runs a first comparison against a [20 10] network on one split |
| `src/v1eval.m` | 5-fold cross-validation across all nine network architectures |
| `src/v1parametertuning.m` | Learning rate and epoch tuning for the shortlisted networks |

## Limitations

- The 918 records come from five pooled cohorts, so populations and measurement practices are mixed.
- No significance test was run, so differences under a percentage point may be noise.
- The models were evaluated as classifiers only, with no probability calibration and no external validation cohort.
- This is a modeling study, not a clinical tool, and it shouldn't inform care.

Natural next steps would be dropout and early stopping for the deeper networks, paired significance tests across the 10-run results, and adding a random forest and an SVM to the comparison.

---

Chase Roby
