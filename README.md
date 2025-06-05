# Mamba-ITS: Next-Generation Time Series-Vision Model with SSM

## Project Overview

**Mamba-ITS** inherits the visualization strategy of ViTST and reconstructs the structural advantages of Swin Transformer (local window, hierarchical patch merging, inter-window interaction, etc.) with SSM (Mamba)-based modules. This next-generation time series-vision model targets the PAM (Patient Activity Monitoring) dataset, aiming to overcome the limitations of existing Transformer-based models by combining the intuitiveness of time series → image conversion with the theoretical strengths of SSM.

## Dataset

- **PAM (Patient Activity Monitoring)**
  - Converts time series data into line graph images as input
  - Provides data preprocessing, visualization pipeline, and augmentation

## Installation & Usage

### 1. Environment Setup

```bash
git clone https://github.com/Seungjun1127/Mamba-ITS.git
cd Mamba-ITS
# (Virtual environment recommended)
pip install -r requirements.txt
```

### 2. Data Preparation

- Run mamba_its/dataset/RF_information_gain.py
- Download preprocessed PAM dataset: https://figshare.com/articles/dataset/PAM_dataset_for_Raindrop/19514347/1?file=34683103
- Place /processed_data, /splits from downloaded dataset into mamba_its/dataset/PAMdata. 
- Directories mamba_its/dataset/PAMdata/process_scripts, mamba_its/dataset/PAMdata/processed_data, and mamba_its/dataset/PAMdata/splits must be now present.
- Run /mamba_its/dataset/PAMdata/process_scripts/ConstructDataset.py

### 3. Training/Evaluation

For training + evaluation on base dataset,
```bash
cd Mamba-ITS
source venv/bin/activate

cd mamba_its
sh ./code/Vision/imgcls_script.sh
```

For evaluation on leave-sensors-out dataset after training,

```bash
sh ./code/Vision/imgcls_script_leave_sensors_out.sh
```

Note: there are bugs in the MambaVision code that need manual fixing. 
The initial execution of imgcls_script.sh will return an error. Afterwards, go to 
root/.cache/huggingface/modules/transformers_modules/nvidia/MambaVision-B-21K/bfe552a588f9f250ea0583951ea4dd5a10a198f8/modelling_mambavision.py
or similar cache folder with the cached MambaVision model file. 

Change 
torch.nn.cross_entropy to torch.nn.functional.cross_entropy
And change all instances of num_head=1000 to num_head=8.

Run imgcls_script.sh again as per normal to start training. 

## References

- [Time Series as Images: Vision Transformer for Irregularly Sampled Time Series](https://arxiv.org/abs/2303.12799)
- [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)

## Contribution & Contact

- Pull requests and issues are welcome
- For research collaboration and inquiries: tmdwns1127@kaist.ac.kr

## License

- MIT License

### ✨ The starting point for next-generation time series-vision model research that captures both mathematical interpretability and practical performance!
