#### Data preparation
We are experimenting on a new training scheme inspired by the FILIP model, which tries to align word to image patches instead of raw images. We want to use it to increase the interpretability of the ECG diagnosis and potentially improve the model performance.

For this project, we want to avoid changing the current codebase, so if certain parts of the codebase doesn't work, please write a new script variation but do not change the original code. This is important.
Any new code should be placed in /filip/ directory while maintaining the directory structures.

For the training data used for the FILIP model experiment, we want to use the MIMIC-IV-ECG dataset.
The dataset root directory is located at "~/8TB/blmcg/datasets/physionet.org.5/files/mimic-iv-ecg/1.0/" as a read only directory, with machine_measurements.csv and record_list.csv in the root, the other files in /files/ folder.

The dataset is saved in .dat and .hea files, instead of the WFDB format, so we may need to create an alternative data loader that can read the .dat and .hea files, or convert the dataset to the WFDB format.

The dataset is currently saved with each patient recording having its own folder, and inside each folder there are multiple .dat and .hea files, corresponding to multiple ECG recordings from the same patient. A mapping of subject_id/study_id to waveform_path can be found in data/mimic-iv-ecg/waveform_note_links.csv, although this file may not be necessary for data processing.

For the FILIP model experiment, we want to extract the following items from the dataset:

1. ECG images: 12-lead ECG images in .png format, with each image containing 4 leads per row, 3 rows in total. Check how can we read the data and whether we can reuse our existing codebase for image generation, and if not, write a new script variation but do not change the original code.
    

2. feature labels: A json file containing the intermediate feature labels for each sample.
    The ECG challenge event has provided labeler scripts which can be found in "~/8TB/blmcg/project/ecg-ksteer/ecg-fm/labeler" as a readonly directory. Please check the labeler scripts first to understand their function and use it as a reference or directly apply it to extract the feature labels.

3. Diagnosis labels: A json file containing the diagnosis labels for each sample. Use the "machine_measurements.csv" to retrieve the main diagnosis labels, and other measurements, which may be useful for the FILIP model experiment. Please check the file first and understand the structure.
You can find the data/mimic-iv-ecg/machine_measurements_data_dictionary.csv, which explains what does each column in the machine_measurements.csv file mean.