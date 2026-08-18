# Deep Hedging vs. Black-Scholes Delta Hedging

A from-scratch Python implementation comparing classical Black-Scholes delta
hedging with a machine-learning-based "deep hedging" policy, under
proportional transaction costs. Built as a personal quant finance project.

Inspired by / benchmarked against:
- Suresh, M. (2026). *Deep Hedging Under Transaction Costs: A Comparison
  with Black-Scholes Delta Hedging.* SSRN Working Paper.
- Buehler, H., Gonon, L., Teichmann, J., and Wood, B. (2019). *Deep Hedging.*
  Quantitative Finance, 19(8):1271-1291.

## What this project does

1. Simulates stock price paths with Geometric Brownian Motion (GBM).
2. Prices a European call option and computes the closed-form Black-Scholes
   delta.
3. Runs a Black-Scholes delta-hedging strategy with proportional transaction
   costs.
4. Trains a small neural network (hand-rolled forward pass, one hidden
   layer) as a "deep hedging" policy, optimized with `scipy.optimize`
   (no PyTorch/TensorFlow required).
5. Compares both strategies on transaction costs and P&L volatility, and
   reproduces the paper's three core figures.
6. **Extension:** sweeps across transaction cost rates (0.1% - 2%) to show
   how the deep hedge's cost advantage and risk penalty scale with market
   friction.

## Project structure

```
deep-hedging-project/
├── deep_hedging_vs_bs.py     # main script - simulation, hedging, training, plots
├── project_report.tex        # LaTeX project report (source)
├── project_report.pdf        # compiled project report
├── README.md                 # this file
├── requirements.txt          # Python dependencies
└── figures/                  # generated plots (created when you run the script)
```

## Requirements

- Python 3.9+
- numpy, scipy, matplotlib

## Setup and running in VS Code

1. **Clone the repo** (or open the folder you downloaded):
   ```bash
   git clone https://github.com/YOUR-USERNAME/deep-hedging-project.git
   cd deep-hedging-project
   ```

2. **Open in VS Code**:
   ```bash
   code .
   ```

3. **Create a virtual environment** (recommended). Open the VS Code
   integrated terminal (`` Ctrl+` `` / `` Cmd+` ``) and run:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If VS Code doesn't automatically detect the venv's interpreter, press
   `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → "Python: Select Interpreter" →
   choose the one inside `venv/`.

5. **Run the script**:
   ```bash
   python deep_hedging_vs_bs.py
   ```
   This prints a results summary to the terminal and saves five PNG
   figures (`fig1_...png` through `fig5_...png`) in the project folder.
   In VS Code you can also just press the "Run" (▶) button in the top
   right while the `.py` file is open.

6. To rebuild the PDF report from the `.tex` source (optional, requires a
   LaTeX distribution such as TeX Live or MacTeX):
   ```bash
   pdflatex project_report.tex
   pdflatex project_report.tex   # run twice for correct references
   ```

## Pushing this project to GitHub

If you don't already have a repository:

1. Create a new, empty repository on GitHub (no README/license, so it
   stays empty) - name it e.g. `deep-hedging-project`.

2. In your project folder, initialize git and make your first commit:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: deep hedging vs Black-Scholes project"
   ```

3. Connect it to the GitHub repo and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/deep-hedging-project.git
   git push -u origin main
   ```

4. Refresh the GitHub page in your browser - your files should now be
   there. Update the link in `project_report.tex` (Section "Code
   Availability") to point to this repo, then recompile the PDF if you
   want the link to match.

### Suggested `.gitignore`
Create a `.gitignore` file so your virtual environment and LaTeX build
artifacts don't get committed:
```
venv/
__pycache__/
*.pyc
*.aux
*.log
*.out
*.toc
```

## Notes

- The neural network is trained with `scipy.optimize` (numerical
  gradients) rather than a deep learning framework, to keep the project
  dependency-light and easy to read end-to-end.
- Random seeds are fixed for reproducibility; exact numbers may differ
  slightly across machines due to floating-point/BLAS differences.

