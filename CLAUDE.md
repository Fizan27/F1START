# F1-EQUALIZER

An F1 "same car" simulator. If every driver in history drove the same car, who
would be fastest? A validated statistical model separates driver skill from car
performance, and a website lets people race drivers from any era against each
other in identical cars.

Public GitHub repository. Public data only (Jolpica F1 API and FastF1). Nothing
private, ever: no secrets, no personal data, no API keys in the repo.

This repository used to hold an F1 race strategy simulator. That project is
kept, unchanged, in `archive/`. Nothing outside `archive/` may import from it.

## The owner

A beginner at coding who needs this finished in 2 to 3 weeks at 1 to 2 hours a
day, for a CV (AI and ML graduate roles). They must understand the core model
well enough to explain it confidently in an interview.

## How to work with me (follow exactly)

- Write the code yourself. No `TODO(human)` lines.
- Keep the project small and flat: as few files as possible. Python files live
  in the top folder, the website lives in `web/`. No clever patterns or
  abstractions that are not needed yet.
- Every file starts with a plain English comment: what it does, what it reads,
  what it produces, and which other files use it.
- Short functions with clear names. Comments explain WHY, not just what.
  Beginner friendly code that is still correct and well tested.
- Keep docs/MAP.md up to date: a simple diagram of how the files connect and
  the order things run in.
- Log every significant decision and the reason in docs/DECISIONS.md.
- Do NOT ask questions and do NOT wait for approval between steps. Make
  reasonable decisions and log them. Only stop for:
  1. Things that cannot be done without the owner (logging in to GitHub or
     Vercel). Then give exact click by click steps.
  2. The end of each phase: list the 2 or 3 functions the owner must
     understand, ask them to explain those back in their own words, and
     correct them.
- After each step, say exactly how to run it and what the owner should expect
  to see.
- Commit after each step. Push after each phase.
- Never spend money: free tools and free hosting only.
- Never make big changes across many files at once.

## The four phases (all required)

1. Data: every race and qualifying result since 1950 (Jolpica), lap times for
   2018 onwards (FastF1), circuit outlines for the website. Cleaned: retirements
   classified as driver caused or car caused, teammates marked, results
   normalised across eras.
2. The model: a statistical model with uncertainty that separates driver skill
   from car performance, using teammate comparisons linked across eras. Skill
   changes over a career. Validated by training up to 2023 and testing on 2024
   and 2025 against simple baselines.
3. The simulation: equal car races, thousands in parallel on the GPU. Exports
   compact JSON for the website.
4. The website: Next.js static site in `web/`, hosted free on Vercel. Races are
   simulated in the browser from the exported ratings. No copyrighted images or
   logos: driver names, simple colours and track outlines only.

Deliverables at the end: README with validation results, docs/INTERVIEW_GUIDE.md
and three honest CV bullet points with real numbers.

## Machine and environment

- Windows 11, Intel Core Ultra 7 255HX, 32 GB RAM, NVIDIA RTX 5070 Ti Laptop
  GPU (12 GB VRAM). Shell is PowerShell.
- Python 3.12 in a virtual environment at `.venv`. Run things with
  `.venv\Scripts\python.exe`.
- RTX 50 series cards need a PyTorch build made for CUDA 12.8 or newer. Older
  builds install fine but cannot run on this GPU. See docs/DECISIONS.md.
- Check the GPU works: `.venv\Scripts\python.exe check_gpu.py`
- Run tests: `.venv\Scripts\python.exe -m pytest`

## Honesty rules for results

- Split by time: the model is fitted on results up to the end of 2023 and
  tested on 2024 and 2025 without refitting. Any tuning uses only data up to
  2023 (for example by holding out 2022 to 2023). Never let 2024 or 2025 leak
  into a choice.
- Only after that test is reported is the model refitted on all data, for the
  ratings the website shows.
- Uncertainty must be honestly wider for older eras and for drivers with few
  races or few teammates.
- Car caused retirements must never count against a driver.
- Report weaknesses. Never overstate results.
- Every number in the README must come from a script in this repo that can be
  re-run.
