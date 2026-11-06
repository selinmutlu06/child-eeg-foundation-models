# What do EEG foundation models learn about children's brains?

Selin Mutlu, Duke University

**Project page:** https://selinmutlu06.github.io/child-eeg-foundation-models/

This project asks whether EEG foundation models capture information about brain development and autism beyond standard spectral features. The first step compares embeddings from pretrained [LaBraM](https://arxiv.org/abs/2405.18765v1), the same architecture with random weights, and 229 standard spectral features for predicting age from eyes-closed resting EEG in 373 children from the [Healthy Brain Network](https://neuromechanist.github.io/data/hbn) (releases 1 to 3).

| Feature set | Nested CV R², age (SD) | MAE, years | Robustness R |
|---|---|---|---|
| Standard spectral features | 0.49 (0.11) | 1.85 | 0.41 |
| LaBraM, pretrained | 0.59 (0.09) | 1.69 | 0.48 |
| LaBraM, random weights | 0.51 (0.07) | 1.91 | 0.38 |
| Standard + LaBraM pretrained | 0.61 (0.09) | 1.67 | |

Pretrained LaBraM beat standard features in 24 of 25 test folds (corrected resampled t-test p = .013, .047 after Holm correction). Adding standard features to LaBraM did not significantly improve it (p = .48), suggesting LaBraM already captures most of the standard-feature information. After removing what the standard features explain, pretrained and random-weights embeddings kept similar small amounts of age information (R² 0.09 vs. 0.12, p = .45), so the pretraining advantage appears to lie mainly in representing spectral information more usefully rather than in new information. A ridge model on the full power spectrum (1,691 values: every 0.5 Hz from 1 to 45 Hz on 19 channels) matched pretrained LaBraM (R² 0.60 vs. 0.59, p = .90), and LaBraM kept no significant age information after the full spectrum was removed (R² 0.04, p = .14). None of the feature sets predicted the CBCL attention factor or the p-factor (R² near zero, n = 362), and movie-watching EEG (The Present, Despicable Me), including inter-subject synchrony, did not predict any of the four CBCL scores either. Details, figures, and limitations are on the project page.

## Repository

```
index.html                         project page (served by GitHub Pages)
pipeline/hbn_pipeline.py           download, preprocessing, standard features, LaBraM embeddings
pipeline/run_release.py            processes one HBN release, one .npz per participant
pipeline/analyze.py                nested CV, prediction robustness, paired fold comparison
pipeline/analyze_v3.py             MAE, combined model, corrected resampled t-tests (Colab cell)
pipeline/analyze_residual.py       residualization test: what is left after removing standard-feature information
pipeline/run_all_releases.py       reprocesses releases 1-3 with standard features, full spectrum, and LaBraM in one pass
pipeline/analyze_full_spectrum.py  full-spectrum comparison and two-way residualization
pipeline/run_movies.py             processes The Present and Despicable Me recordings, including synchrony signals
pipeline/analyze_movies.py         movie vs. rest comparison and inter-subject correlation
pipeline/hbn_labram_colab.ipynb    runs everything in Google Colab
results/                           summary tables for the 373-participant analysis
```

## Running it

Open `pipeline/hbn_labram_colab.ipynb` in Google Colab and run all cells. It downloads HBN resting-state EEG from the public S3 bucket (no data agreement needed for age and CBCL factors), saves per-participant features to Google Drive, and writes the results tables and figures. Processing one release takes about an hour, mostly downloading.

Main dependencies: `mne`, `braindecode`, `specparam`, `scikit-learn`, `boto3`, `torch`.

## Method summary

1. Interpolate noisy channels (robust z-score of log SD), average reference, 0.5 to 75 Hz band-pass, 60 Hz notch.
2. Map HBN's 128-channel EGI net to the 19 standard 10-20 positions, resample to 200 Hz, cut eyes-closed blocks into 15 s windows, reject windows above 500 µV peak-to-peak.
3. Features per participant: absolute and relative band power, aperiodic offset and exponent (specparam), alpha peak frequency; mean token embeddings from each of LaBraM's 12 layers, pretrained and randomly initialized.
4. Ridge regression with robust scaling, nested 5-fold CV repeated 5 times, layer selected inside each training set, predictions capped to the training range.
5. Robustness adapted from Carson et al. (2024), using agreement between predictions of models trained on disjoint subsets.

## Data

Healthy Brain Network EEG, Child Mind Institute, CC-BY-SA 4.0. Raw data is not included in this repository.
