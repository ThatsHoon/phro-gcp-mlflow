# Probe before training (all stages)

Train only after the input, labels and ceiling are understood. Decisive fixes usually come from these probes, not from training curves. `domain.md` names the tool for each stage; if none exists, write a small read-only script that prints counts only.

| Step | Question | What to check |
|---|---|---|
| 1 Data check | normalization, leakage, unit-level split, label statistics (counts, empty or odd rows) printed as numbers; the user inspects actual labels (for example in the MLflow UI), the agent never prints them | text normalization consistent between refs and outputs; eval-only sources absent from training; split by `resample_unit`; labels of excluded categories dropped |
| 2 Ceiling | best achievable with this input | oracle or best-candidate metric; label (teacher) quality vs the target; per-condition buckets |
| 3 Information | does the input carry the signal, does it complement other inputs | how often the needed information is present in the input; recoverable vs missing share; distribution of model scores vs random |
| 2b Loss split | how much of the gap to target comes from the upstream stage vs this stage | train and test this stage on perfect upstream input; the difference to real input = upstream share |
| 4 Smoke | does the path run | a handful of steps on a handful of samples (tag `status=smoke`); too little data learns only base rates, so smoke proves the path, not the method |

Rules
- Probe results are runs too: write them in the standard result folder (`references/contract.md`) so they reach MLflow, one line per measured case.
- If a probe shows a problem, stop and give the user a method change (what, why, the number behind it) before any training.
- Weak items found by a probe (for example low recall on one class) are the first targets for the next sweep.
- A stage that does not exist yet gets no probes until it does.
