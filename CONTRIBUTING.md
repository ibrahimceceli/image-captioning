**English** · [Türkçe](CONTRIBUTING.tr.md)

# Team workflow

This file collects the working rules of the three-person team. The project itself, the fixed decisions and the reproduction steps are in [README.md](README.md).

## Branches

- `main`: working shared code
- `data`: Member A
- `attention`: Member B
- `transformer-demo`: Member C

Before starting, pull `main` into your own branch. When you are done, commit to your branch and open a pull request into `main`.

## File ownership

- `src/train.py`, `src/data/` and the demo skeleton (`app/app.py`, the shared layer in `app/captioners.py`): Member A
- `src/decode.py` and the attention model (`src/models/attention_lstm.py`): Member B
- `src/evaluate.py` and the demo's model connections (`BlipCaptioner` and the Model 3 adapter in `app/captioners.py`, the model entries in `app/config.yaml`): Member C

If a change is needed in someone else's file, the owner of that file merges the pull request.

## Pull request rules

- Every change reaches `main` through a pull request from your own branch.
- A pull request that touches someone else's file is merged by that file's owner.
- Values marked as "locked" in the project overview ([README: Fixed decisions](README.md#fixed-decisions)) are not changed.
- Data, the feature cache, checkpoints and `kaggle.json` are never committed (`.gitignore`).
- If `src/data/artifacts/` shows changes after running `python -m src.data.prepare`, do not commit them; revert with `git checkout -- src/data/artifacts`.
