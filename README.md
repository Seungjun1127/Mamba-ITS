# 🌟 Mamba-ITS: Next-Generation Time Series-Vision Model with SSM

**Mamba-ITS** visualization strategy of ViTST and efficiency of MambaVision. This next-generation time series-vision model targets the PAM (Patient Activity Monitoring) dataset, aiming to overcome the limitations of existing Transformer-based models by combining the intuitiveness of time series → image conversion with the theoretical strengths of SSM.

---

## 📋 Project Overview

Mamba-ITS leverages the power of SSM to deliver a robust and interpretable solution for time series analysis. By converting time series data into visual representations, it enables seamless integration with vision-based models, achieving both mathematical rigor and practical performance.

---

## 📊 Dataset

### PAM (Patient Activity Monitoring)
- **Input**: Time series data transformed into line graph images.
- **Features**:
  - Comprehensive data preprocessing pipeline.
  - Visualization and augmentation tools for enhanced model training.

---

## 🛠️ Installation & Usage

### 1. Environment Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/Seungjun1127/Mamba-ITS.git
cd Mamba-ITS
# Recommended: Use a virtual environment
pip install -r requirements.txt
```

### 2. Data Preparation
Follow these steps to prepare the PAM dataset:

1. Run the initial preprocessing script:
   ```bash
   python mamba_its/dataset/RF_information_gain.py
   ```
2. Download the preprocessed PAM dataset from [Figshare](https://figshare.com/articles/dataset/PAM_dataset_for_Raindrop/19514347/1?file=34683103).
3. Extract and place `/processed_data` and `/splits` into `mamba_its/dataset/PAMdata/`.
4. Ensure the following directories exist:
   - `mamba_its/dataset/PAMdata/process_scripts/`
   - `mamba_its/dataset/PAMdata/processed_data/`
   - `mamba_its/dataset/PAMdata/splits/`
5. Construct the dataset:
   ```bash
   python mamba_its/dataset/PAMdata/process_scripts/ConstructDataset.py
   ```

### 3. Training & Evaluation

#### Base Dataset
To train and evaluate on the base dataset:

```bash
cd Mamba-ITS
source venv/bin/activate
cd mamba_its
bash ./code/Vision/imgcls_script.sh
```

#### Leave-Sensors-Out Dataset
For evaluation on the leave-sensors-out dataset after training:

```bash
bash ./code/Vision/imgcls_script_leave_sensors_out.sh
```

#### 🐞 Bug Fix for MambaVision
The initial run of `imgcls_script.sh` may fail due to bugs in the MambaVision code. To resolve:

1. Locate the cached MambaVision model file, typically at:
   ```
   root/.cache/huggingface/modules/transformers_modules/nvidia/MambaVision-B-21K/bfe552a588f9f250ea0583951ea4dd5a10a198f8/modelling_mambavision.py
   ```
2. Make the following changes:
   - Replace `torch.nn.cross_entropy` with `torch.nn.functional.cross_entropy`.
   - Update all instances of `num_classes` to 8 instead of 1000.
3. Re-run `imgcls_script.sh`.

---

## 📚 References

- [Time Series as Images: Vision Transformer for Irregularly Sampled Time Series](https://arxiv.org/abs/2303.12799)
- [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)

---

## 🤝 Contribution & Contact

- **Contributions**: Pull requests and issues are warmly welcomed!
- **Contact**: For research collaboration or inquiries, reach out at [tmdwns1127@kaist.ac.kr](mailto:tmdwns1127@kaist.ac.kr).

---

## 📜 License

This project is licensed under the **MIT License**.

---

### ✨ The starting point for next-generation time series-vision model research that captures both mathematical interpretability and practical performance!