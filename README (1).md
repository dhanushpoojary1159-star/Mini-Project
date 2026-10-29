# Quantum Enhanced Machine Learning for Accelerated Cancer Therapeutics Discovery

Mini project (VTU, B.E. Information Science and Engineering), Canara Engineering College, Bantwal. Academic year 2026-2027.

- **Team:** Adithya Alva (Team Leader), Dhanush, Jnanesh M, Kaushal V
- **Guide:** Ms. Sadhana B, Assistant Professor
- **Status:** Planning / setup (update this line as work progresses)

---

## 1. Purpose of this file

This README is the single source of truth for the project. It records what we are building, the decisions already made, the plan, and what is still open. When resuming work (or asking an AI assistant for help), share this file first so the context is not lost.

---

## 2. Project summary

A software system that predicts whether a molecule is **active or inactive against cancer** from its SMILES string.

Workflow:
1. User enters a SMILES string.
2. User picks a model: **Classical SVM**, **Random Forest**, **QSVM (quantum kernel)**, or **All (compare)**.
3. App shows the predicted class, a confidence value, and the 2D molecular structure.
4. App compares the models using accuracy, precision, recall and F1-score.
5. A **Table-RAG** module reads the stored prediction results and generates a report (tables, visuals, summary), exportable as **PDF or CSV**.

Predictions are computational results only and need laboratory validation. This disclaimer must appear in the app and in every exported report.

---

## 3. Objectives (corrected)

1. Collect cancer compound data from ChEMBL and convert SMILES to Morgan fingerprints using RDKit.
2. Implement and train three classifiers: Classical SVM, Random Forest, and QSVM.
3. Evaluate all three on the same split using accuracy, precision, recall and F1-score, and present results in a Streamlit application.
4. Generate evidence-grounded reports through a Table-RAG module with PDF/CSV export.

### Important: do not copy reported accuracies

The synopsis draft mentions 96% (SVM), 94% (Random Forest) and 60% (QSVM). These numbers are the results of a **different project** (AIT Chikkamagaluru, ref. 49S_BE_2665, "Quantum IntelDrug"). We have not run any experiments yet, so they must **not** be presented as our results.

Action items:
- [ ] Remove those numbers from Objective 3 in the synopsis.
- [ ] Report only the numbers we measure ourselves.
- [ ] Cite the AIT project as related prior work if we build on its idea.

### How we differ from the prior work

- Table-RAG report generation (new component).
- Single well-defined target/assay instead of mixed cancer data.
- Scaffold-based train/test split to avoid inflated accuracy.
- Same-data, fair comparison of classical vs quantum models.
- Reports precision, recall, F1 and confusion matrices, not only accuracy.
- Optional: calibrated confidence scores.

---

## 4. Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10 or 3.11 |
| Cheminformatics | RDKit |
| Classical ML | scikit-learn |
| Quantum ML | Qiskit, qiskit-machine-learning, qiskit-aer (simulator) |
| Data source | ChEMBL (`chembl_webresource_client` or CSV download) |
| Web app | Streamlit |
| Plots | matplotlib |
| PDF export | reportlab |
| Model storage | joblib |
| Prediction log | SQLite or pandas DataFrame |
| RAG | pandas retrieval (or embeddings) + an LLM for summary text |

---

## 5. Suggested repository structure

```
project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # ChEMBL downloads (do not edit)
│   └── processed/           # cleaned data, fingerprints, splits
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_featurization.ipynb
│   ├── 03_classical_models.ipynb
│   └── 04_qsvm.ipynb
├── src/
│   ├── data.py              # download, clean, label
│   ├── features.py          # SMILES -> Morgan fingerprint
│   ├── split.py             # scaffold split
│   ├── train_classical.py
│   ├── train_qsvm.py
│   ├── evaluate.py          # metrics, confusion matrix, charts
│   ├── predict.py           # load model, predict, confidence
│   └── rag_report.py        # Table-RAG + PDF/CSV export
├── models/                  # saved .joblib files
├── app/
│   └── streamlit_app.py
├── reports/                 # generated PDF/CSV outputs
└── docs/                    # synopsis, report, screenshots
```

---

## 6. Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate    |    macOS/Linux: source venv/bin/activate
pip install rdkit scikit-learn pandas numpy qiskit qiskit-machine-learning qiskit-aer \
            streamlit matplotlib reportlab joblib chembl_webresource_client
pip freeze > requirements.txt
```

Use one shared GitHub repository from day one. Commit small and often; never commit large raw data files without checking size.

---

## 7. Technical plan

### 7.1 Data (ChEMBL)
- Choose **one** cancer target or cell-line assay (for example EGFR). Mixed assays give noisy labels.
- Keep IC50 values in nM and convert: `pIC50 = 9 - log10(IC50_nM)`.
- Proposed labelling rule (confirm and document):
  - Active: pIC50 >= 6
  - Inactive: pIC50 < 5
  - Drop the 5 to 6 grey zone, or treat it explicitly and state the choice.
- Remove duplicates, missing values, and invalid SMILES. Handle duplicate molecules with conflicting values (for example take the median).
- Record dataset size and class balance in this README once known.

### 7.2 Features
- Morgan fingerprints with radius 2, 1024 or 2048 bits via RDKit.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def featurize(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    return np.array(fp)
```

### 7.3 Splitting
- Use a **scaffold split** (group by Bemis-Murcko scaffold). A random split inflates scores because of near-duplicate molecules.
- Fix a random seed and save split indices so every model uses the same data.

### 7.4 Classical models
- SVM (RBF kernel) and Random Forest.
- Handle imbalance with `class_weight="balanced"` if needed.
- Report accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

### 7.5 Quantum model (QSVM)
- Fingerprints are too wide for a simulated circuit. Reduce with PCA to 4 to 8 features.
- Subsample to roughly 200 to 500 training molecules (kernel matrix simulation is expensive).
- Use `FidelityQuantumKernel` with `ZZFeatureMap`, then `SVC(kernel="precomputed")` or Qiskit `QSVC`.
- **Fairness rule:** also train SVM and Random Forest on the same reduced and subsampled data, and report both the full-data and reduced-data classical results.
- A QSVM that does not beat classical models is still a valid, honest finding.
- Try a few feature maps and `reps` values; log each experiment.

### 7.6 Streamlit app
- SMILES input, SMILES validation with a clear error message.
- Model selector: SVM / RF / QSVM / All.
- 2D structure via `Draw.MolToImage`.
- Predicted class and confidence value.
- Metrics comparison table and bar chart.
- Load saved models with joblib; do not retrain on each run.
- Disclaimer banner.

### 7.7 Table-RAG report module
- Log every prediction (SMILES, model, class, confidence, timestamp) and the evaluation metrics in one structured table (SQLite or DataFrame).
- On request, retrieve the relevant rows (pandas filters or embeddings), then have an LLM write a summary **grounded only in the retrieved rows**.
- The report includes SMILES, 2D structures, metrics, summary tables and the disclaimer.
- Export as PDF (reportlab) and CSV.
- The summary must never state numbers that are not in the table.

---

## 8. Roadmap

| Week | Goal | Deliverable |
|---|---|---|
| 1 | Environment, repo, ChEMBL target chosen, labelling rule fixed | Cleaned dataset |
| 1-2 | Featurization | Feature matrix saved |
| 2 | Classical baselines with scaffold split | Metrics for SVM and RF |
| 3-4 | QSVM experiments | Quantum kernel results and comparison |
| 4-5 | Streamlit app | Working prototype |
| 5-6 | Table-RAG and PDF/CSV export | Report generator |
| Final | Full evaluation, charts, limitations, documentation | Final report and demo |

Adjust dates to the college's review schedule.

---

## 9. Team split

| Member | Primary responsibility |
|---|---|
| Adithya Alva (Lead) | Streamlit app, integration, repo management |
| Dhanush | Data collection and cleaning (ChEMBL, labelling) |
| Jnanesh M | Featurization and classical models, evaluation |
| Kaushal V | QSVM and Table-RAG/report export |

Everyone should understand the full pipeline for the viva. Rotate review of each other's code.

---

## 10. Progress checklist

- [ ] Repo created and shared
- [ ] Environment working on all four machines
- [ ] ChEMBL target chosen
- [ ] Labelling threshold decided and documented
- [ ] Clean dataset saved
- [ ] Morgan fingerprints generated
- [ ] Scaffold split implemented
- [ ] SVM trained and evaluated
- [ ] Random Forest trained and evaluated
- [ ] PCA reduction and subsampling for QSVM defined
- [ ] QSVM trained and evaluated
- [ ] Models saved with joblib
- [ ] Streamlit app prototype
- [ ] Prediction logging
- [ ] Table-RAG summary working
- [ ] PDF and CSV export working
- [ ] Final comparison charts and tables
- [ ] Synopsis corrected (remove borrowed accuracy numbers)
- [ ] Final report and presentation

---

## 11. Decisions log

Record every decision here with a date so we do not revisit it.

| Date | Decision | Reason |
|---|---|---|
| | Target/assay: TBD | |
| | Active/inactive threshold: TBD | |
| | Fingerprint size: TBD | |
| | QSVM feature count and sample size: TBD | |
| | LLM used for RAG summary: TBD | |

---

## 12. Known limitations (to state in the final report)

- Quantum models run on a **simulator**, not real quantum hardware.
- QSVM needs heavy dimensionality reduction and small samples.
- ChEMBL labels depend on assay conditions and are noisy.
- Binary active/inactive labelling is a simplification of real potency.
- Predictions are not a substitute for laboratory validation.
- Side-effect or cancer-type predictions (if added) would need a separate, validated data source; do not generate them without one.

---

## 13. Future work

- ADMET and toxicity prediction.
- Additional targets and cell lines, multi-task learning.
- Deep learning baselines (graph neural networks) for comparison.
- Running the quantum kernel on real hardware or a cloud backend.
- Calibrated uncertainty estimates.
- Integration with additional databases.

---

## 14. References

1. Bertl, M., Mott, A., Sinno, S., & Bhalgamiya, B. (2025). Quantum machine learning in precision medicine and drug discovery. arXiv:2502.18639. https://arxiv.org/abs/2502.18639
2. Shara, J. (2024). An Overview of Quantum Machine Learning and Drug Discovery. IJAEM, 6(08), 174-183.
3. Arodola, O. A., & Soliman, M. E. (2017). Quantum mechanics implementation in drug-design workflows: does it really help? Drug Design, Development and Therapy, 11, 2551-2564. https://doi.org/10.2147/DDDT.S126344
4. Li, J., et al. (2021). Drug discovery approaches using quantum machine learning. ISQED. https://arxiv.org/abs/2104.00746
5. Quantum Machine Learning for Drug Discovery: Taxonomy, Research Challenges, and the Road Ahead. https://dl.acm.org/doi/full/10.1145/3785660
6. AI-Driven Drug Discovery: Accelerating the Search for New Treatments. https://doi.org/10.5281/zenodo.16276691
7. Rogers, D., & Hahn, M. Extended-connectivity fingerprints. Journal of Chemical Information and Modeling.
8. Gaulton, A., et al. ChEMBL: a large-scale bioactivity database for drug discovery. Nucleic Acids Research.
9. EMBL-EBI, ChEMBL bioactivity database. https://www.ebi.ac.uk/chembl/
10. Related prior work: Quantum IntelDrug, AIT Chikkamagaluru, 49th SPP 2025-26, Ref. 49S_BE_2665.

Reference clean-up: replace the `share.google` shortened links in the synopsis with the real DOI or publisher URLs, and keep the in-text citation numbers consistent with this list.
