# Workflow Predictive Analytics Project

## Project Objective
This project predicts workflow ticket resolution time using machine learning.

## Prediction Moment
The project predicts ticket resolution time **at ticket creation**.

## Prediction Rule
Only features known at the moment the ticket is created are allowed as model inputs.

## Target Variable
`task_duration_hours = resolved_at - opened_at`

## Safe Feature Principle
A feature may only be used if it is available when the ticket is first opened.

## Success Criterion
The final enhanced model must:
1. beat the median dummy baseline by at least 20% on MAE
2. maintain stable cross-validation error
3. avoid leakage by using only creation-time features