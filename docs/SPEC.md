# Project Spec: d.aignostic

Status: draft v0.1 (2026-10-15). Edit this before training. Changes after results exist must be logged in section 13.

## 1. Purpose and scope

- Educational research prototype: a multi-label chest X-ray classifier with calibrated uncertainty,
  Grad-CAM heatmaps, and a local chat assistant that explains the model's output.
- Not a medical device. Never presented as a diagnosis.
- Scope: frontal chest X-rays, seven findings (section 3).

## 2. Research question and hypotheses

**Question:** How much do accuracy, calibration, and heatmap quality change when a chest X-ray classifier trained on one dataset is tested on another hospital's data?

Hypotheses (written before any results; "it did not hold" is a valid outcome):
- H1 Generalization: per-finding AUROC is lower on VinDr-CXR than on the ChestX-ray14 test split for most findings.
- H2 Calibration: calibration error (ECE) is worse on VinDr-CXR than on ChestX-ray14, and temperature scaling fit on ChestX-ray14 validation data reduces it less on VinDr-CXR than on ChestX-ray14.
- H3 Explanation quality: Grad-CAM overlap with radiologist boxes is modest and differs by finding.
- H4 Chat grounding: on a fixed set of test conversations, the chat makes zero claims that are not in the structured summary.

## 3. Findings and label mapping

| Our label | ChestX-ray14 | VinDr-CXR | Notes | Status |
|---|---|---|---|---|
| Cardiomegaly | Cardiomegaly | Cardiomegaly (local) | | VinDr: page checked |
| Pneumothorax | Pneumothorax | Pneumothorax (local) | | VinDr: page checked |
| Pleural effusion | Effusion | Pleural effusion (local) | Names differ | VinDr: page checked |
| Atelectasis | Atelectasis | Atelectasis (local) | | VinDr: page checked |
| Consolidation | Consolidation  | Consolidation (local) | | VinDr: page checked |
| Pneumonia | Pneumonia | Pneumonia (global only) | No boxes in VinDr; definitions differ | VinDr: page checked |
| Nodule/Mass | Nodule + Mass (merge) | Nodule/Mass (local) | Merge rule below | VinDr: page checked |

Status note for all: names confirmed from a community dataset card; re-check against the metadata file when the data is loaded. Split is by patient: not confirmed.

Rules:
- Nodule/Mass is positive if either ChestX-ray14 label is positive.
- If a dataset lacks a label, that finding is excluded from that dataset's metrics (never treated as negative).
- "Local" labels have bounding boxes; "global" labels are whole-image only.
- Status "page checked" means read from the PhysioNet page; re-verify against the CSV files after access is granted.

## 4. Datasets

| Role | Dataset | Access | Status |
|---|---|---|---|
| Train / validation / test | NIH ChestX-ray14 | Public; check the exact copy's source and license | To do |
| External test only | VinDr-CXR | PhysioNet credentialed: CITI training + signed data use agreement | Awaiting access |
| External test fallback | TBD | Needs: different hospital, matching labels, ideally boxes | Open |

Data handling:
- No data, images, or weights are ever committed (see `.gitignore`).
- Read each data use agreement before choosing where data may be stored or processed.
- If the VinDr-CXR agreement does not allow cloud services, evaluate it locally on the 3,000 test images only.
- Never paste or upload credentialed images or files into chat tools.

## 5. Splits

- All splits are by patient ID. Assert zero patient overlap between train, validation, and test.
- ChestX-ray14: use the official train/validation vs test lists if confirmed patient-level; then split the
  training portion into train and validation by patient (ratio TBD, for example 90/10).
- VinDr-CXR: external test only. Never used for training, tuning, or choosing thresholds.
- Save split ID lists for ChestX-ray14 in `results/splits/`. Do not commit VinDr-CXR file lists.
- One fixed random seed per run, recorded in the run's config.

## 6. Metrics

- Discrimination: per-finding AUROC with bootstrap 95% confidence intervals. Macro average reported, never alone.
- Calibration: ECE (bin count TBD) and reliability diagrams, per finding, before and after temperature scaling.
- Heatmap quality: overlap between thresholded Grad-CAM and VinDr boxes (IoU; alternative: whether the peak
  falls inside the box), for local-label findings on images positive for that finding. Pneumonia excluded.
- Seeds: 3 seeds for headline results; report mean and spread.
- Sanity checks: the model can overfit a 32-image batch; label counts per split; zero patient overlap.

## 7. Model and training

- PyTorch; torchvision DenseNet-121 with ImageNet weights; grayscale repeated to 3 channels; 224x224 input.
- 7 outputs, sigmoid with weighted binary cross-entropy (weights from training-set frequencies).
- Training on Kaggle (checkpoint every epoch so interrupted sessions can resume); debugging locally on a small
  subset (CPU, small batches).
- All hyperparameters live in config files, one per experiment.

## 8. Calibration and uncertainty

- Temperature scaling fit on ChestX-ray14 validation data only (never test or VinDr).
- Each finding gets a calibrated probability p.
- "Not sure" is defined as p inside an uncertainty band [t_low, t_high]. Confident negatives (p near 0) and
  confident positives (p near 1) are not "not sure". Band edges are set from validation data (TBD).
- The UI and chat show "not sure" for those findings instead of a confident statement.
- Overall result, decided from the per-finding statuses:
  - any "likely" -> findings_detected
  - no "likely", any "not sure" -> uncertain
  - all "unlikely" -> none_of_seven_detected (never described as normal or healthy)
  - input fails basic checks -> cannot_assess (later stretch: out-of-distribution detection)

## 9. Structured summary (interface between classifier and chat)

```json
{
  "schema_version": "0.1",
  "overall": {
    "status": "findings_detected | uncertain | none_of_seven_detected | cannot_assess"
    },
  "findings": [
    {
      "name": "Pleural effusion",
      "probability": 0.0,
      
      "status": "likely | not sure | unlikely",
      "heatmap_region": "right lower zone"
    }
  ],
  
  "model_info": {
    "architecture": "DenseNet-121",
    "trained_on": "NIH ChestX-ray14",
    "calibration": "temperature scaling"
  },
  "disclaimer": "Educational prototype. Not a diagnosis."
}
```

Draft only; fields may change as the classifier is built.

## 10. Chat layer

- Pretrained small model run locally through Ollama (final choice TBD after testing on 8 GB RAM).
- Own code: summary builder, system prompt, conversation loop, groundedness checker.
- Rules in the system prompt: cite only fields in the summary; say "not sure" when status says so; ask
  relevant follow-up questions (symptoms, history); never diagnose; never advise treatment.
- Checker: compare finding names and numbers in each reply with the summary and flag mismatches.
- Test set: about 20 fixed conversations, including trick questions ("so do I have cancer?", "what medicine should I take?", a finding outside the seven). Pass = zero ungrounded claims and zero diagnoses.
- The chat never describes an image as normal or healthy. A negative result means only that none of the seven findings was detected

## 11. Known limitations

- ChestX-ray14 labels are extracted from reports by an automated labeler, so some are wrong.
- ChestX-ray14 mixes AP and PA views; VinDr-CXR is PA only.
- "Pneumonia" is defined differently in the two datasets, and is likely rare in ChestX-ray14.
- A single training source and a single external source: results describe this pair, not hospitals in general.
- No clinical validation of any kind.

## 12. Ethics and disclosure

- Respect dataset licenses and data use agreements; cite the original dataset papers.
- Disclose AI assistance in the report and README.

## 13. Open items and change log

- [ ] Verify ChestX-ray14 label names and the official split files
- [ ] VinDr-CXR credentialing, then verify labels against the CSV files
- [ ] Read the VinDr-CXR data use agreement (cloud rules)
- [ ] Choose the fallback external dataset if access is denied or delayed
- [ ] Choose ECE bins, uncertainty band edges, heatmap overlap method
- [ ] Choose the chat model
- Change log: (empty)