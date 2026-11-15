# KAIST AX20002 - Homework 2: Classification under Continual Drift
**Student:** Lukas Lindholm (20266508)

## 1. Environment & Setup
To run these scripts, ensure that your environment has `torch`, `torchvision`, `pandas`, `numpy`, `matplotlib`, and `Pillow` installed. 

**File Structure:**
The scripts are configured to run from a "clean folder". Please place the competition data files directly in the same root directory as the Python scripts. The directory should look like this:
- `train.csv`
- `test.csv`
- `images.npz`
- `generate_submission.py`
- `local_validation.py`
- `diagnostics.py`
- `pseudo_analysis.py`

## 2. Running the Code
Run the scripts from your terminal.

1. **Exploratory Data Analysis:**
   `python diagnostics.py`
   *Generates `label_shift.pdf` and `covariate_shift.pdf` in the current directory.*

2. **Local Validation (Train on Stage 1-2, Test on Stage 3):**
   `python local_validation.py`
   *Trains the ResNet-34 model on the first two stages to empirically validate its performance on Stage 3 data.*

3. **Generate Final Kaggle Submission (Train on Stage 1-3, Predict Stage 4-5):**
   `python generate_submission.py`
   *Trains the model on all labeled data and generates `submission.csv` for Kaggle.*

4. **Pseudo-Analysis (Label Shift Verification):**
   `python pseudo_analysis.py`
   *Uses the saved `best_model_all_data.pth` to estimate the label distribution in Stages 4 and 5.*

## 3. Note on Reproducibility and Kaggle Score
The highest Kaggle public score reported in the PDF report (**0.875**) was achieved during an initial unseeded training run using the exact architecture and hyperparameter configuration found in `generate_submission.py` (ResNet-34, Adam, LR=0.0001, Data Augmentation). 

To strictly comply with the assignment requirements for reproducibility, a fixed random seed (`SEED = 42`) and deterministic CuDNN backend settings were retroactively added to all scripts. Running `generate_submission.py` with this fixed seed yields a comparable score of **0.851** on the Kaggle public leaderboard (I also submitted this version to the leaderboard) due to standard hardware and augmentation stochasticity. The underlying logic, model architecture, and training methodology remain identical to the 0.875 run.

## 4. AI Usage Declaration
The overarching strategic direction and algorithmic choices in this project were guided by an AI assistant (Google Gemini). The primary role of the student was that of a critical overseer: rigorously verifying the implementation, questioning the AI's suggestions, ensuring the logical soundness of the selected approaches, and evaluating alternative strategies.