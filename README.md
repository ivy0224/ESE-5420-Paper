# Hotel Booking Cancellation Classification

**Course:** ESE 5420 – Statistics for Data Science, Fall 2026  
**Team Meber:** Shen Li, Yining Liang, Fenting Jin

## 1. Project Overview

Hotel booking cancellations create challenges for revenue management, room allocation, and operational planning. Being able to identify reservations with a high likelihood of cancellation can help hotels better understand booking patterns and prioritize reservations for further review.

This project investigates how machine learning classification models can predict hotel booking cancellations using reservation characteristics, including lead time, hotel type, distribution channel, length of stay, and previous booking history.

The primary objective is to compare different classification approaches and evaluate their ability to estimate cancellation probabilities.

## 2. Research Questions

- Which reservation characteristics are associated with hotel booking cancellations?
- How accurately can classification models estimate the probability of a booking being cancelled?
- How do different classification models compare in terms of predictive performance?

## 3. Dataset

We use the public [Hotel Booking Demand Datasets](https://www.sciencedirect.com/science/article/pii/S2352340918315191?via%3Dihub#s0010) published by Antonio, de Almeida, and Nunes (2019):
- **H1.csv:** Resort Hotel bookings
- **H2.csv:** City Hotel bookings

The datasets are combined into a unified dataset, with an additional `hotel_type` feature distinguishing resort and city hotels.

The original datasets contain 119,390 bookings and 32 variables after adding the hotel type indicator.

**Target Variable:**
- `is_canceled = 1`: Booking was cancelled.
- `is_canceled = 0`: Booking was not cancelled.

The dataset contains reservation information such as:

- Booking lead time
- Arrival date
- Length of stay
- Number of guests
- Distribution channel
- Market segment
- Previous cancellations and booking history
- Hotel type

## 4. Methodology

The project follows a structured predictive modeling workflow.

### Data Preparation
- Combine the city and resort hotel datasets.
- Standardize column names and inspect data types.
- Audit missing values, duplicate records, and unusual observations.
- Exclude records without any recorded guests.
- Review potential data leakage and exclude outcome-related variables.

### Feature Engineering
Four additional features are created:

- `log_lead_time`
- `total_nights`
- `total_guests`
- `is_family`

Categorical encoding, missing-value imputation, scaling, and rare-category grouping are handled within the modeling pipeline.

### Exploratory Data Analysis (EDA)

The exploratory analysis examines:
- Overall booking cancellation distribution.
- Differences in cancellation rates between city and resort hotels.
- Relationship between booking lead time and cancellation rates.
- Relationship between length of stay and cancellation rates.

### Model Development

The current baseline uses **Ridge Logistic Regression (L2-penalized Logistic Regression)**.

Model selection is performed using:
- Four expanding-window cross-validation splits.
- Quarterly validation windows in 2016.
- Mean validation log loss as the primary selection criterion.

The modeling pipeline includes preprocessing, categorical encoding, and regularization.

## 5. Evaluation Strategy

A chronological split is used to evaluate generalization to later arrival cohorts:

| Dataset | Arrival Period | Bookings |
|---|---|---:|
| Development | July 2015 – December 2016 | 78,590 |
| Held-out Test | January 2017 – August 2017 | 40,619 |

The held-out test cohort is reserved for final evaluation after model development and tuning are complete.

**Evaluation Metrics:**
- Log Loss (primary metric)
- ROC-AUC (secondary metric)
- Precision
- Recall
- F1 Score
- Accuracy
- Confusion Matrix

Threshold-based metrics are evaluated at a common probability threshold of 0.5.

## 6. Limitations (TBC)

- The chronological split is based on scheduled arrival dates rather than booking creation dates.
- The dataset does not fully reconstruct the information available at the original booking time, so some temporal leakage risk may remain.
- Identical rows are retained because the dataset does not provide unique booking identifiers.
- Descriptive associations should not be interpreted as causal relationships.
