# Principal Component Analysis of Breast Cancer Referrals @ Anderson Cancer Center

A reproducible **Apache Spark (PySpark)** pipeline that prepares the center's breast cancer referral data. It then uses **Principal Component Analysis (PCA)** to find the variables that matter most, as a first step toward a model that supports donor funding.

## Files

| File | Description |
| --- | --- |
| `pca_pipeline.ipynb` | The full pipeline: load → clean → assemble & standardise → PCA → bonus logistic regression (saved with outputs) |
| `breast_cancer_referrals.csv` | Raw referral data provided with the assignment (572 rows, 32 columns) |
| `pca_components.png` | Scatter plot of patients on the 2 principal components, coloured by diagnosis |
| `requirements.txt` | Python packages |
| `.gitignore` | Keeps system and Spark scratch files out of the repository |

## Requirements

- **Java 17 or later.** Spark 4 runs on the JVM. Check with `java -version`; on macOS, install with `brew install openjdk@17` if needed.
- **Python 3.10–3.14** with the packages in `requirements.txt`:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## How to run

From this folder:

```bash
jupyter lab
```

Open `pca_pipeline.ipynb` and choose **Run → Run All Cells**. Spark runs locally on your machine (`local[*]`), so no cluster is needed. A full run takes under a minute.

## What the pipeline does

1. **Load with an explicit schema.** A `StructType` declares every column:
   - `patient_id` and `diagnosis` are text.
   - The 30 cell measurements (mean, standard error and worst value of 10 cell-nucleus features) are decimals.

   `enforceSchema=False` checks the file's header against the schema, and `mode="FAILFAST"` stops on any badly formatted value.
2. **Clean:**
   - **Duplicates:** `dropDuplicates()` removes 3 repeated records (572 → 569 rows), and a check confirms every `patient_id` is then unique.
   - **Missing values:** 18 measurements are blank (6 each in `mean_radius`, `mean_texture` and `worst_area`). They're filled with the column median using Spark's `Imputer`, since the median isn't skewed by large size values.
   - **Diagnosis:** rows missing one would be dropped, because the outcome can't be filled in; this file has none. The diagnosis is converted to a numeric `label`: 1 = Malignant, 0 = Benign.
3. **Assemble and standardise:**
   - `VectorAssembler` combines the 30 measurements into one feature vector.
   - `StandardScaler` rescales each to mean 0 and standard deviation 1, so large-valued features like area don't dominate the PCA.
   - These stages and PCA form a Spark ML `Pipeline`, so the same steps can be rerun on future referral exports.
4. **PCA:** `PCA(k=2)` reduces the features to 2 principal components. The notebook reports the variance each explains, the variables that weigh most on each, and a scatter plot of the result.
5. **Bonus: logistic regression.** A second pipeline (assemble → scale → PCA → `LogisticRegression`) is trained on 80% of patients and tested on the other 20%.
   - The split is based on a hash of `patient_id`, so it's the same on every machine.
   - The scaler and PCA are fitted on training data only, so the test accuracy is a fair estimate.

## Results

| Component | Variance explained | Main variables |
| --- | --- | --- |
| PC1 | **44.2%** | concave points, concavity, compactness, perimeter, radius (nucleus size and shape irregularity) |
| PC2 | **19.0%** | fractal dimension (border complexity), plus radius and area |
| **Total** | **63.2%** | |

- **PC1 separates the diagnoses:** malignant and benign referrals fall on opposite sides, with only a narrow overlap (see `pca_components.png`).
- **Essential variables for the donor-funding model:** the nucleus size and shape-irregularity measurements behind PC1, especially **concave points, concavity, compactness and nucleus size**.
- **Bonus:**
  - Logistic regression on just the 2 components predicts the diagnosis with **96.4% accuracy** on the test set (107 of 111 patients), against 95.2% on training data.
  - Of the 4 errors, 2 were benign cases flagged as malignant and 2 were malignant cases missed.

## Notes

- Spark prints a few startup messages, such as log4j settings and a native-hadoop warning. They're normal for a local run and can be ignored.
- The results above come from PySpark 4.2.0. A different Spark version may give slightly different decimals.
