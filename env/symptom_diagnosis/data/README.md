# Symptom Diagnosis Dataset

## Task Description
Medical symptom-based diagnosis prediction task. Given a patient's symptom description, predict the most likely medical diagnosis.

## Data Format
Each line is a JSON object containing the following fields:
- `question`: Patient symptom description (natural language)
- `answer`: Ground truth diagnosis (disease name)

## Data Statistics
- Training set: 200 samples
- Validation set: 50 samples
- Test set: 212 samples
- Total: 462 samples
- Number of unique diagnoses: 22

## Data Source
Data from `gretelai/symptom_to_diagnosis` on Hugging Face.
- Original dataset: https://huggingface.co/datasets/gretelai/symptom_to_diagnosis
- The test split from the original dataset is preserved as the test split
- The train split from the original dataset (853 samples) was stratified sampled to create:
  - 200 training samples
  - 50 validation samples
- Stratified sampling was performed based on diagnosis labels to maintain class distribution

## Diagnosis Distribution

### Training Set (200 samples)
| Diagnosis | Count |
|-----------|-------|
| diabetes | 10 |
| dengue | 10 |
| chicken pox | 10 |
| allergy | 10 |
| impetigo | 10 |
| arthritis | 10 |
| gastroesophageal reflux disease | 9 |
| typhoid | 9 |
| cervical spondylosis | 9 |
| hypertension | 9 |
| malaria | 9 |
| pneumonia | 9 |
| psoriasis | 9 |
| peptic ulcer disease | 9 |
| drug reaction | 9 |
| bronchial asthma | 9 |
| urinary tract infection | 9 |
| common cold | 9 |
| varicose veins | 9 |
| fungal infection | 9 |
| jaundice | 7 |
| migraine | 7 |

### Validation Set (50 samples)
| Diagnosis | Count |
|-----------|-------|
| drug reaction | 3 |
| bronchial asthma | 3 |
| malaria | 3 |
| hypertension | 3 |
| varicose veins | 3 |
| psoriasis | 3 |
| cervical spondylosis | 2 |
| dengue | 2 |
| chicken pox | 2 |
| typhoid | 2 |
| pneumonia | 2 |
| diabetes | 2 |
| peptic ulcer disease | 2 |
| migraine | 2 |
| common cold | 2 |
| gastroesophageal reflux disease | 2 |
| allergy | 2 |
| arthritis | 2 |
| fungal infection | 2 |
| jaundice | 2 |
| urinary tract infection | 2 |
| impetigo | 2 |

### Test Set (212 samples)
| Diagnosis | Count |
|-----------|-------|
| peptic ulcer disease | 10 |
| pneumonia | 10 |
| cervical spondylosis | 10 |
| gastroesophageal reflux disease | 10 |
| psoriasis | 10 |
| arthritis | 10 |
| malaria | 10 |
| migraine | 10 |
| bronchial asthma | 10 |
| diabetes | 10 |
| chicken pox | 10 |
| impetigo | 10 |
| hypertension | 10 |
| common cold | 10 |
| dengue | 10 |
| varicose veins | 10 |
| allergy | 10 |
| fungal infection | 9 |
| typhoid | 9 |
| urinary tract infection | 9 |
| drug reaction | 8 |
| jaundice | 7 |

## Evaluation Metric
- **Accuracy**: The predicted diagnosis must exactly match the ground truth diagnosis
  - Case-insensitive comparison
  - Whitespace normalized
  - Exact match required (accuracy = 1.0 if correct, 0.0 otherwise)

## Example
```json
{
  "question": "I have been experiencing persistent fatigue, increased thirst, frequent urination, and blurred vision for the past few weeks. I also noticed that I'm losing weight despite eating normally.",
  "answer": "diabetes"
}
```

## Regenerating Data

To regenerate the dataset files, run:
```bash
uv run python env/symptom_diagnosis/prepare_data.py
```

This will:
1. Download the dataset from Hugging Face (`gretelai/symptom_to_diagnosis`)
2. Preserve the original test split (212 samples)
3. Perform stratified sampling on the original train split to create:
   - 200 training samples
   - 50 validation samples
4. Save the splits as `train.jsonl`, `val.jsonl`, and `test.jsonl`

## Dataset Characteristics

This dataset is ideal for:
- **Domain Knowledge Accumulation**: Requires understanding of medical terminology, disease symptoms, and clinical patterns
- **Pattern Recognition**: Models must learn to recognize symptom combinations that indicate specific diseases
- **Multi-class Classification**: 22 different disease categories with varying symptom presentations
- **Real-world Medical Application**: Symptoms are described in natural language, similar to patient complaints

The task requires models to:
1. Parse natural language symptom descriptions
2. Identify key clinical features from unstructured text
3. Apply medical domain knowledge to map symptoms to diseases
4. Distinguish between diseases with overlapping symptoms
5. Handle variations in how patients describe symptoms
