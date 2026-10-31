# FORGETGRAPH

**Dependency-aware data deletion verification for an ML pipeline**
*Syntaxis 2026 Hackathon — PS09*

## The problem

Deleting a user from a database isn't enough to actually "forget" them. Once their
data has fed into derived features, cached predictions, and a trained model, simply
removing their row leaves their influence baked into everything downstream — in
violation of the spirit of "right to be forgotten" requirements like GDPR.

## What this does

FORGETGRAPH treats data as a dependency chain:

```
users (raw rows) → features (derived) → predictions (cached) → model (Ridge)
```

When a deletion request comes in, it doesn't just delete a row — it walks the
entire chain:

1. Removes the user's raw data
2. Recomputes derived features without them
3. Invalidates any cached predictions tied to them
4. Retrains the model on the remaining data
5. **Independently verifies** the retraining actually worked
6. Records a tombstone to block the same ID from re-entering

## The core idea: verify, don't just claim

The most important part of this system is step 5. Instead of just trusting that
retraining worked, a completely separate verification process:

- Opens its own fresh database connection (no shared state with the deletion pipeline)
- Independently re-queries the retained data and recomputes features from scratch
- Trains a second, fully independent **reference model**
- Compares its coefficients against the pipeline's **replacement model** using
  `numpy.allclose()`

If the two independently-trained models agree within tolerance, that's real evidence
the deletion pipeline worked correctly — not just a reported "success" message. If
a bug caused incomplete deletion (e.g. stale cached features), the two models would
diverge, and the system would flag it.

## Tech stack

- **Database:** SQLite (5 tables: users, features, predictions, tombstones, deletion_log)
- **Model:** `scikit-learn` Ridge regression (`alpha=1.0`, `random_state=42`, fixed
  everywhere for reproducibility)
- **Dashboard:** Streamlit
- **Tests:** pytest (5 tests covering deletion, model change, independent
  verification, idempotency, and re-ingestion blocking)

## Dataset

200 synthetic users (numpy, seed=42), with features `age`, `account_balance`,
`num_transactions`, `tenure_days`, predicting a continuous `risk_score`.

## Project structure

```
PS09/
├── app.py                  — Streamlit dashboard
├── train_original.py       — seeds DB, trains baseline model
├── test_forgetgraph.py     — pytest suite
├── requirements.txt
└── forgetgraph/
    ├── __init__.py         — shared constants (seed, alpha, tolerances)
    ├── schema.py            — DB schema + synthetic data generation
    ├── pipeline_core.py     — feature computation, deletion pipeline, ingestion guard
    └── verify.py            — independent verification (fresh connection, reference model)
```

## Running it

```bash
cd PS09
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
python -m forgetgraph.schema      # seed 200 synthetic users
python train_original.py          # train the baseline model
pytest test_forgetgraph.py -v     # run the test suite
streamlit run app.py              # launch the dashboard
```

## Dashboard walkthrough

- **Overview** — live KPIs (total users, deletions, blocked re-ingestions, features,
  verifications passed) and a dependency graph showing real row counts at each stage
- **Deletion Center** — trigger a deletion and watch the 6-step pipeline execute live
- **Model Comparison** — side-by-side coefficients (original / replacement /
  reference) with a pass/fail verification badge and a comparison chart
- **Tombstones** — list of every deleted user ID and when they were removed
- **Re-ingestion Guard** — attempt to re-insert a user ID and see it accepted
  (fresh ID) or blocked (tombstoned ID)

## What makes a tombstone different from a backup

A tombstone is purely a **ban on the same ID** re-entering through normal data
ingestion — it stores no data about the deleted person, only their ID and when they
were removed. If that person genuinely returns later, they sign up with a new ID and
start completely fresh, with no link to their old data. The tombstone exists to catch
accidental re-ingestion of stale data (e.g. an old backup or sync job), not to
recognize returning humans.
