""""The seven findings and how NIH ChestX-ray14 label names maps to them"""

# Fixed order: a finding's position here is its slot in the model's output.
FINDINGS = [
    "Cardiomegaly",
    "Pneumothorax",
    "Pleural effusion",
    "Atelectasis",
    "Consolidation",
    "Pneumonia",
    "Nodule/Mass",
]

# NIH separates multiple labels with this character.
SEPARATOR = "|"

#Use of label-mapping table since uses a cross-dataset model NIH labels are different from ours
NIH_TO_FINDING = {
    "Cardiomegaly": "Cardiomegaly",
    "Pneumothorax": "Pneumothorax",
    "Effusion": "Pleural effusion",
    "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Nodule": "Nodule/Mass",
    "Mass": "Nodule/Mass",
}

# Real NIH labels we deliberately don't use, for images that carry multiple labels
NIH_IGNORED = {
    "No Finding", "Infiltration", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
}


def nih_labels_to_vector(finding_labels: str) -> list[float]:
    """Turn an NIH label string like 'Effusion|Nodule' into a 7-slot 0/1 list."""
    vector = [0.0] * len(FINDINGS)
    for name in finding_labels.split(SEPARATOR):
        if name in NIH_TO_FINDING:
            vector[FINDINGS.index(NIH_TO_FINDING[name])] = 1.0
        elif name not in NIH_IGNORED:
            raise ValueError(f"Unknown NIH label: {name!r}")
    return vector