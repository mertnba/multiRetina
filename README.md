# retina-fusion

retina-fusion is a modular pipeline for multimodal retinal image analysis. It integrates feature extraction using pretrained foundation models (e.g. RetFound) and vessel morphology analysis (e.g. AutoMorph), with support for dataset-agnostic preprocessing, fine-tuning, and downstream classification or regression tasks.

## Components

- Data preprocessing with support for patient-level stratified splits  
- Latent feature extraction using vision transformers (RetFound)  
- Morphological feature extraction using vessel segmentation models (AutoMorph)  
- Fusion experiments with shallow MLPs and classical models  
- Statistical tests and model evaluation tools  


## Dependencies

- Python ≥ 3.9  
- torch, timm, numpy, pandas, scikit-learn, seaborn, pingouin, statsmodels, etc.  
- External: [RetFound MAE](https://github.com/rmaphoh/RETFound_MAE), [AutoMorph](https://github.com/rmaphoh/AutoMorph)
