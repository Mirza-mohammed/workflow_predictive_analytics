# Validation Rules for Enhanced ETL

## Required Fields
1. `number` must not be missing
2. `opened_at` must not be missing
3. `resolved_at` must not be missing
4. `priority` must not be missing
5. `category` must not be missing

## Timestamp Rules
6. `opened_at` must be a valid datetime
7. `resolved_at` must be a valid datetime
8. `resolved_at` must be greater than or equal to `opened_at`

## Data Quality Rules
9. `task_duration_hours` should not be negative
10. Rows failing validation should be logged separately
11. Valid rows should be loaded into the enhanced database table only

## Logging Rules
12. Record total input rows
13. Record valid rows
14. Record invalid rows
15. Record ETL runtime