# F1START

An F1 race strategy simulator with an AI strategist. It answers "what if the team
had pitted differently?" and "what would an AI strategist have done?".

Public GitHub repository. Public data only (FastF1). Nothing private, ever: no
secrets, no personal data, no API keys in the repo.

## The owner

A beginner at coding who wants to genuinely learn machine learning from this
project, and needs it finished in about 3 weeks at 1 to 2 hours a day for a CV
(AI/ML graduate roles, for example Williams F1). They must be able to explain
every important design decision in an interview.

## How to work with me (follow exactly)

- Keep the project small and flat: as few files as possible, one folder until a
  split is truly needed. No clever patterns or abstractions that are not needed yet.
- Every file starts with a plain English comment: what it does, what it reads,
  what it produces, and which other files use it.
- Short functions with clear names. Comments explain WHY, not just what.
  Beginner friendly code that is still correct and well tested.
- Keep docs/MAP.md up to date: a simple diagram of how the files connect and the
  order things run in.
- Working mode (changed by the owner on 2026-10-08): finish the project as
  fast as possible. Do NOT ask questions and do NOT wait for approval between
  steps. Make reasonable decisions and log each one in docs/DECISIONS.md.
  No `TODO(human)` lines: write all the code. Only stop for things that
  cannot be done without the owner (creating the GitHub repository, logging
  in to Streamlit Community Cloud), and then give exact click by click steps.
- Commit after each step, and push once the GitHub remote exists.
- Never spend money: free tools and free hosting only.
- The owner still has to explain this project in interviews, so the core ML
  code keeps plain English comments that explain the idea next to the code,
  and docs/INTERVIEW_GUIDE.md explains each part and the likely questions.
- Keep docs/DECISIONS.md with each significant decision and why.
- Show results as tables and simple charts, and keep a results section in
  README.md updated with real, measured numbers, including weaknesses.
- Never make big changes across many files at once.

## The four phases (all required)

1. Race model (supervised ML): FastF1 data with Polars; LightGBM baseline; a
   PyTorch lap time network on the GPU that predicts a range (uncertainty); pit
   loss per circuit; safety car / VSC likelihood; validated by replaying held
   out races. STOP and show replay validation results before Phase 2.
2. Simulator: Gymnasium style lap by lap environment, vectorised in PyTorch so
   thousands of races run in parallel on the GPU; brute force best fixed strategy.
3. AI strategist: PPO trained on the GPU in the vectorised environment,
   evaluated on held out races against the best fixed strategy and the real team.
4. Website: Streamlit app (not Next.js) that must also run CPU only on a small
   free server.

Version one scope: pit lap, tyre choice, one versus two stops, safety cars and
VSCs, undercut and overcut. Rain, double stacking and extras only if time allows.

## Machine and environment

- Windows 11, Intel Core Ultra 7 255HX, 32 GB RAM, NVIDIA RTX 5070 Ti Laptop
  GPU (12 GB VRAM). Shell is PowerShell.
- Python 3.12 in a virtual environment at `.venv`. Run things with
  `.venv\Scripts\python.exe`, or activate first with `.venv\Scripts\Activate.ps1`.
- RTX 50 series cards need a PyTorch build made for CUDA 12.8 or newer. Older
  builds install fine but cannot run on this GPU. See docs/DECISIONS.md.
- Check the GPU works: `.venv\Scripts\python.exe check_gpu.py`
- Run tests: `.venv\Scripts\python.exe -m pytest`

## Honesty rules for results

- Split by time: train on 2022 to 2023, validate on 2024 (ALL tuning and model
  choices use 2024 only), test on 2025. Do not look at 2025 results until the
  end; then retrain on 2022 to 2024 with fixed settings and evaluate on 2025
  once. Never let validation or test races leak into training. 2026 is out of
  scope for version one.
- Report weaknesses and any simulator flaws the agent exploits.
- Every number in the README must come from a script in this repo that can be
  re-run.
