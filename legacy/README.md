# Legacy pipeline (superseded)

The repository's original analysis, kept unchanged for history. **Do not use its risk scores.**

`loan_recovery_system.py` clustered borrowers with K-Means, labelled two of the four clusters "high risk",
and trained a Random Forest to predict that label from the same ten features the clusters were built on.
The model re-learned the clustering (93.5% accuracy), not default: the flagged group's write-off rate
(41.9%) was *lower* than the unflagged group's (45.6%). Its inputs also mixed post-default collection
fields with pre-default behaviour.

The replacement in [`../src`](../src) predicts a future default event from information available at a
snapshot date, and models LGD, EAD and Expected Loss. See the [top-level README](../README.md).

To run the legacy code: `pip install -r legacy/requirements.txt`, then run from `legacy/src/`.
