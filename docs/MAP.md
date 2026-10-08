# Map of the project

How the files connect, and the order things run in. Updated after every step.

## Run order

```
 one-off setup
 -------------
 check_gpu.py              proves PyTorch trains on the GPU


 Phase 1: data             (being built)
 Phase 2: the model        (not started)
 Phase 3: the simulation   (not started)
 Phase 4: the website      (not started)
```

## Files

| File | What it is for | Reads | Produces |
|---|---|---|---|
| `check_gpu.py` | Proves training runs on the GPU | nothing | a printed report |
| `download_data.py` | Downloads race lap times with FastF1 | FastF1 (internet) | `data/raw/` |
| `chart_style.py` | Shared chart colours | nothing | nothing |
| `test_*.py` | Tests, one file per part | nothing | pass/fail |
| `requirements.txt` | Python libraries | | |
| `pytest.ini` | Tells pytest which folders to skip | | |
| `CLAUDE.md` | The rules for how this project is built | | |
| `docs/DECISIONS.md` | Each significant decision and why | | |

## Folders

- `data/` is downloaded data. Not in git (large; rebuilt by the download scripts).
- `archive/` is the earlier strategy simulator project. Nothing uses it.
- `.venv/` is the Python environment. Not in git.
