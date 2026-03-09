# Data Dictionary

## Dataset Overview

This dataset contains **24,918 incident records** and **36 columns**.  
It appears to be a ticket/incident management dataset suitable for predicting **task/incident resolution duration**.

For this project, the primary predictive target will be derived from the timestamp fields:

- `opened_at`
- `resolved_at`

The target variable will be:

`task_duration_hours = resolved_at - opened_at`

All columns currently show **0 missing values** based on the initial inspection.

---

## Column Dictionary

| Column Name | Description | Data Type | Missing Values | Role in Project | Notes |
|-------------|-------------|-----------|----------------|-----------------|-------|
| number | Unique identifier for the incident/ticket | str | 0 | Identifier | Primary ticket ID; should not be used directly as a predictive feature |
| incident_state | Current or final state of the incident | str | 0 | Input Feature | Useful categorical workflow state |
| active | Indicates whether the incident is active | bool | 0 | Input Feature | Boolean flag; may need conversion to integer |
| reassignment_count | Number of times the incident was reassigned | int64 | 0 | Input Feature | Useful operational behaviour feature |
| reopen_count | Number of times the incident was reopened | int64 | 0 | Input Feature | May indicate complexity or recurring issues |
| sys_mod_count | Number of times the incident record was modified | int64 | 0 | Input Feature | Could reflect process complexity |
| made_sla | Indicates whether the SLA was met | bool | 0 | Input Feature / Analysis Feature | May be useful, but check whether it leaks post-resolution information |
| caller_id | Identifier for the caller who raised the incident | str | 0 | Input Feature | High-cardinality categorical feature |
| opened_by | Identifier of the user who opened the incident | str | 0 | Input Feature | May be encoded if used |
| opened_at | Timestamp when the incident was opened | str | 0 | Timestamp / Target Source | Must be converted to datetime; used to calculate task duration |
| sys_created_by | User/system that created the record | str | 0 | Input Feature | Categorical metadata |
| sys_created_at | Timestamp when the record was created in the system | str | 0 | Timestamp Feature | Convert to datetime; similar to opened_at, check redundancy |
| sys_updated_by | User/system that last updated the record | str | 0 | Input Feature | May not be useful for early prediction if it contains future information |
| sys_updated_at | Timestamp when the record was last updated | str | 0 | Timestamp Feature | Likely post-event information; check for leakage |
| contact_type | Channel through which the incident was reported | str | 0 | Input Feature | Example: phone, email, self-service, etc. |
| location | Location associated with the incident or caller | str | 0 | Input Feature | Categorical location field |
| category | Main incident category | str | 0 | Input Feature | One of the key modelling features |
| subcategory | More detailed classification under category | str | 0 | Input Feature | Useful finer-grained categorical feature |
| u_symptom | Reported symptom associated with the incident | str | 0 | Input Feature | May be important for issue-type prediction |
| cmdb_ci | Configuration item linked to the incident | str | 0 | Input Feature | Likely high-cardinality categorical field |
| impact | Business/service impact level | str | 0 | Input Feature | Important priority-related operational feature |
| urgency | Urgency level of the incident | str | 0 | Input Feature | Important severity-related feature |
| priority | Overall incident priority | str | 0 | Input Feature | One of the most important predictive features |
| assignment_group | Team/group assigned to handle the incident | str | 0 | Input Feature | Useful organisational workflow feature |
| assigned_to | Specific agent/user assigned to the incident | str | 0 | Input Feature | Equivalent to assigned agent; high-cardinality categorical feature |
| knowledge | Indicates whether a knowledge article is linked/used | bool / str | 0 | Input Feature / Analysis Feature | Verify exact meaning and type during preprocessing |
| u_priority_confirmation | Indicates whether priority was confirmed | str | 0 | Input Feature / Analysis Feature | Check meaning and usefulness |
| notify | Notification setting or behaviour | str | 0 | Metadata / Optional Feature | Check whether this is useful for modelling |
| problem_id | Identifier of linked problem record | str | 0 | Input Feature / Metadata | May indicate relationship to broader issue; high-cardinality |
| rfc | Identifier or flag for related request for change | str | 0 | Input Feature / Metadata | Check actual values and whether it is sparse |
| vendor | Vendor associated with the incident or service | str | 0 | Input Feature | Useful if incidents depend on external systems/vendors |
| caused_by | Cause reference for the incident | str | 0 | Input Feature / Metadata | Check whether this is populated with meaningful categories or IDs |
| close_code | Code explaining how the incident was closed | str | 0 | Optional / Leakage Risk | Likely post-resolution information; may not be suitable for early prediction |
| resolved_by | Identifier of the user who resolved the incident | str | 0 | Optional / Leakage Risk | Post-outcome field; may cause target leakage |
| resolved_at | Timestamp when the incident was resolved | str | 0 | Timestamp / Target Source | Must be converted to datetime; used to calculate task duration |
| closed_at | Timestamp when the incident was formally closed | str | 0 | Timestamp Feature | Convert to datetime; may be used for alternate duration analysis |

---

## Project-Relevant Column Mapping

For this project, the following columns align closely with the proposed workflow prediction structure:

| Project Field | Dataset Column |
|---------------|----------------|
| ticket_id | `number` |
| created_at | `opened_at` |
| resolved_at | `resolved_at` |
| priority | `priority` |
| category | `category` |
| assigned_agent | `assigned_to` |

---

## Target Variable Definition

The main target variable for this project will be:

`task_duration_hours = resolved_at - opened_at`

This will be calculated after converting both timestamp columns to datetime format.

---

## Initial Observations

- The dataset contains **24,918 records** and **36 variables**
- There are **no missing values** in the initial inspection
- Several timestamp columns are currently stored as **string (`str`)** and must be converted to datetime
- Multiple high-cardinality categorical fields are present, such as:
  - `caller_id`
  - `opened_by`
  - `assigned_to`
  - `cmdb_ci`
  - `problem_id`
- Some fields may introduce **target leakage** if used directly for prediction, especially:
  - `resolved_by`
  - `resolved_at`
  - `closed_at`
  - `close_code`
  - `sys_updated_at`
  - `sys_updated_by`
- The strongest likely predictive features include:
  - `priority`
  - `impact`
  - `urgency`
  - `category`
  - `subcategory`
  - `assignment_group`
  - `assigned_to`
  - `reassignment_count`
  - `reopen_count`
  - `incident_state`

---

## Notes for Preprocessing

1. Convert these columns to datetime:
   - `opened_at`
   - `sys_created_at`
   - `sys_updated_at`
   - `resolved_at`
   - `closed_at`

2. Create the target column:
   - `task_duration_hours`

3. Exclude or carefully review leakage-prone columns before modelling:
   - `resolved_at`
   - `closed_at`
   - `resolved_by`
   - `close_code`
   - `sys_updated_at`
   - `sys_updated_by`

4. Encode categorical columns before training machine learning models

5. Convert boolean columns such as `active` and `made_sla` into numeric format if required by the modelling pipeline

---

## Version Note

This data dictionary is based on the first inspection of the currently loaded `workflow_tasks.csv` dataset and may be refined as preprocessing and feature engineering progress.