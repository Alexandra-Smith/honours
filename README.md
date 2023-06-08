Honours project 2021
==============================
Title: The automated detection of brain tumours in MRI scans
==============================
by Alexandra Smith

- **preprocessing.py** takes the original images downloaded from Kaggle as well as from the MICCAI Multimodal Brain Tumour Segmentation (BraTS) challenge and applies various preprocessing techniques to obtain the final images used for model training
- **indv_modality.py** contains the functions for working with only one of the imaging modalities
- **all_modalities.py** contains functions for working with all 4 image modalities (image depth 4) which was used for the final project results
- **main.py** executes the training and testing of the model
