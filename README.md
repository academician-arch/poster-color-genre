# Color-based genre signals in movie posters

Code, data and full results for the study

> **The Syntax of Cinema Posters: Machine Learning Insights into Color-Based Genre Expectations**

The article is under review. The full reference, the authors and the article DOI will be added here after
publication.

The pipeline extracts fifteen interpretable color features from the 25,958 posters of the
[MM-IMDb dataset](https://archive.org/details/mmimdb) (Arevalo et al. 2017), relates them to IMDb genre
labels, and tests how far genre can be recovered from color alone.

## Main results

- **Genre labels in MM-IMDb are listed alphabetically.** Among multi-label films the genre array is strictly
  alphabetical in 89.1% and alphabetical after a leading Short / Animation / Documentary label in a further
  6.7%. The first-listed label is therefore not a "primary genre"; the main analysis uses all labels
  (19 genres with at least 800 films).
- **Posters of different genres differ in color** (Pillai's V = .296, permutation p = .001). The differences
  are concentrated in lightness (HSV value / CIELAB L\*) and colorfulness; these rank first to third under
  five genre definitions and in both color spaces.
- **Color alone predicts every genre above chance, but modestly** (macro ROC AUC = .647). Family, Animation
  and Horror are most recoverable; Music, Biography and War least.
- **Class-specific SHAP values** show that dark posters raise the predicted probability of Horror and
  Thriller, and light, colorful posters that of Comedy and Family.
- **Thirteen of fourteen directional predictions** derived from color-emotion research were met.

## Repository layout

| Path | Content |
|---|---|
| `poster_color_genre_analysis.py` | Complete analysis pipeline (single script) |
| `requirements.txt` | Python dependencies (versions of the reported run) |
| `data/first_listed_sample_features.xlsx` | Feature values of the original first-listed-label sample (7,419 posters), used by stage `RA0` |
| `results/tables/` | All result tables (131 CSV files) |
| `results/figures/` | All figures produced by the pipeline (300 dpi PNG) |
| `results/manuscript_figures/` | The seven figures of the article |
| `results/implementation_details.txt` | Grayscale conversion, entropy, dominant color, SHAP settings, pseudocode |
| `results/run_environment.json` | Software versions and settings of the reported run |

Posters and IMDb metadata are **not** redistributed here; the script downloads MM-IMDb from the Internet
Archive. The SHA-256 checksum of the archive used is in `results/tables/R01_dataset_provenance.csv`.

## Genre definitions and file prefixes

| Prefix | Genre definition | Films | Genres |
|---|---|---|---|
| `RB_` | **Multi-label, all films (main analysis)** | 25,958 | 19 |
| `RC_` | Films with exactly one genre label | 2,988 | 5 |
| `RA0_` | Original first-listed-label sample with its stored feature values | 7,419 | 10 |
| `RA_` | First-listed label, features re-extracted | 7,419 | 10 |
| `RD1_` | Random-label rule (20 draws) | about 11,900 per draw | 19 |
| `RD2_` | Rarest-label rule | 15,017 | 19 |

Other prefixes: `R0x`-`R2x` raw-data audit, feature extraction and robustness checks; `*_LAB_` the same
analysis with CIELAB features; `RB_CROP_` posters cropped by 10% on each edge; `RL` HSV versus CIELAB;
`RX` feature ranking across genre definitions; `RH` directional predictions (hypothesis 3).

Where to find the numbers reported in the article:

| Article | Files |
|---|---|
| Table 1, Figures 1-2 | `R02_genre_order_audit`, `R03_genre_frequency_all_vs_first_position`, `R04_labels_per_film`, `R06_cooccurrence_conditional_probability` |
| Table 2 | `RX1_effect_size_ranks_across_genre_definitions`, `RL2_effect_sizes_and_shap_hsv_vs_cielab`, `*_omnibus_permutation`, `*_manova_permutation` |
| Figure 3 | `RB_cohen_d_matrix`, `RB_genre_feature_effects_cohen_d` |
| Table 3 | `RB_per_genre_metrics_RandomForest`, `RB_classification_summary`, `RB_permutation_test_classifier` |
| Table 4, Figure 4 | `RA0_per_genre_precision_recall_f1`, `RA0_confusion_matrix_row_normalised`, `RA0_imbalance_sensitivity_per_genre`, `RA0_strict_vs_any_label_accuracy` |
| Figures 5-6 | `RB_shap_by_genre_signed`, `RB_shap_signed_matrix`, `RB_shap_feature_family` |
| Table 5, Figure 7 | `RH1_H3_a_priori_predictions_vs_observed`, `RH2_H3_agreement_summary`, `RH3_valdez_mehrabian_PAD_indices_by_genre` |
| Robustness checks | `R12_sensitivity_resize_grayscale_k`, `R13`-`R15_border_crop_*`, `RL1_hsv_vs_cielab_feature_correlations` |

## Running the pipeline

```bash
pip install -r requirements.txt
python poster_color_genre_analysis.py
```

- The working directory defaults to `./work`; set the environment variable `POSTER_GENRE_BASE_DIR` to use
  another location.
- MM-IMDb (`mmimdb.tar.gz`, 8.7 GB) is downloaded on first use; the download resumes if interrupted. An
  already extracted copy can be given through `MMIMDB_DIR_OVERRIDE` in the settings section of the script.
- Set `DATA_SOURCE = "demo"` to run every stage on a small synthetic dataset in a few minutes (smoke test).
- A full run takes one to three hours on a desktop computer. Extracted features are cached.
- Random seeds are fixed (`RNG_SEED = 42`). Results were obtained with Python 3.10.6 on Windows 10 and the
  package versions in `requirements.txt`; other versions of scikit-learn can change Random Forest results
  in the third decimal.

## Provenance of the files in `results/`

The tables and figures were produced by one run of this pipeline on 16 November 2026. Afterwards the script's
console messages and comments were translated into English, its default paths were made generic, and its CSV
output was switched to comma-separated values with a decimal point. No computation was changed: the
translated script reproduces the tables of a test run exactly. The CSV files here were re-exported in the new
format, and the wording of two text columns (`source_note` in `RH1_...` and `CAUTION` in `RH2_...`) was
edited. Labels such as "A0 submitted analysis" and "SVM_RBF_unscaled_as_submitted" refer to the original
first-listed-label analysis of an earlier version of the study.

## Use of generative AI

The analysis code was written with the assistance of a generative AI model (Claude, Anthropic) and was run
and checked by the authors.

## License

Code: MIT License (see `LICENSE`). Result tables and figures: Creative Commons Attribution 4.0 International
(CC BY 4.0). MM-IMDb and IMDb content remain subject to their own terms.

## References

Arevalo, J., T. Solorio, M. Montes-y-Gómez, and F. A. González. 2017. "Gated Multimodal Units for Information
Fusion." arXiv:1702.01992.

Valdez, P., and A. Mehrabian. 1994. "Effects of Color on Emotions." *Journal of Experimental Psychology:
General* 123(4): 394-409.
