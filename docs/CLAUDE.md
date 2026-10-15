# CLAUDE.md

## Project
- Create an AI imaging tool that can analyze a given scan along with an in-chat feature which allows the user to talk to the model to see what is happening with the scan (for better outputs). 
- The target for the project is for Chest X-rays in identification for is found in scope 
- Still will include an LLM model with workable UI in website form.  

## How to work with me
- I want to use this for a project but also a learning opportunity, for code I want to review and learn it as it continues. 
- So before committing anything I want to understand how the commands work and I'll reexplain it back to Claude to ensure I understand.
- explain the concept before the code, write in small chunks, walk through the code line by line, ask me to explain it back, and don’t treat a piece as done until I can.
- Also add “flag anything you’re unsure about.”

## Scope and decisions
- Scan: X-ray on the Chest
- Findings: The target for the project is for Chest X-rays in identification for; Cardiomegaly, Pneumothorax, Pleural effusion, Atelectasis, Consolidation, Pneumonia, and Lung nodule/mass. Also looking to implement uncertainity if cannot identify. 
- Train data / external test: NIH ChestX-ray14, public, with Kaggle copies. Its labels are NLP-extracted, so some are noisy. 
    - (Open decision) External test: VinDr-CXR first choice, and any replacement must be a different hospital with matching labels, ideally with bounding boxes for the heatmap test. Chat model: tested on 8 GB RAM, with a behavior check that it stays within the summary. 
    - Cleaning: patient-level splits, a label-mapping table, DICOM to PNG conversion for VinDr (check for inverted brightness), resize and normalize, drop corrupted files, and note the AP/PA view mismatch.
- Model and framework: Pytorch
    - DenseNet-121, ImageNet-pretrained, multi-label
    - explainability (Grad-CAM) 
    - calibration (temperature scaling).
- Training happens on: Kaggle, and some testing locally. Intel laptop, 8 GB RAM, no NVIDIA GPU, so local work means CPU, small batches and small subsets.
- Chat: a pretrained small model through Ollama, plus your own chat layer: the structured-summary format, the system prompt, the conversation loop, and a checker that flags claims not in the summary.

## Hard Rules
- Data, images and weights never get committed
- Splits are by patient ID, with zero overlap between train, validation and test
- Seeds are fixed and recorded
- Temperature scaling is fit on validation data only
- The chat may only cite what’s in the classifier’s structured summary, and never gives a diagnosis or treatment advice
- Model shouldn't give answer if doesn't pass threshold

## Layout 
src/daignostic/   library code
configs/          one config file per experiment
scripts/          command-line entry points
notebooks/        thin Kaggle/Colab notebooks
app/              the UI
tests/            tests
docs/             spec, notes
results/          small result files only
data/             local dev subset (git-ignored)

## Commands
- To be added later

## Open Decisions
- external data set
- final chat model
- uncertainty thershold