# Hybrid Quantum Neural Network (HQNN)

A hybrid classical–quantum neural network (HQNN) designed to explore quantum advantages in machine learning for regression tasks on Option pricing on NIFTY  Options datasets.

> 🚧 This project is a work in progress. Contributions, feedback, and collaboration inquiries are welcome.

---

## ✨ Overview

This project combines classical neural networks with quantum circuits using Qiskit and PyTorch to build a hybrid model for predictive tasks. The goal is to evaluate whether quantum-enhanced layers can contribute to learning efficiency or expressivity in practical settings.

---
## 📌 Latest update: Submitted and under consideration at npj Quantum Information.

---

## 🚀 Features

### 🧠 Model Architecture
- Hybrid **Quantum-Classical Neural Network** using PennyLane and PyTorch.
- **Quantum Neural Network (QNN)** layer implemented via `qml.qnode` and `qml.qnn.TorchLayer`.
- **Autoencoder Structure**:
  - `PreAutoencoder`: Encodes classical input data.
  - `PostAutoencoder`: Decodes quantum circuit outputs.

### ⚙️ Training Setup
- Full **GPU support** using CUDA with fallback checks.
- **Mixed Precision Training** using `torch.autocast` and `torch.cuda.amp.GradScaler`.
- Adaptive **Learning Rate Scheduler** (`ReduceLROnPlateau`).
- **Early Stopping** to avoid overfitting and reduce training time.

### 📉 Monitoring and Logging
- **Gradient Norm Tracking** (total and per-layer) for interpretability and debugging.
- **Learning Rate Logging** across epochs.
- **Comprehensive Training Logs** saved in JSON and CSV formats.
- **Best Model Checkpointing** based on validation loss.

### 📊 Post-Training Visualizations
- **Training & Validation Loss Curves**.
- **Gradient Norm Visualization** across training batches.
- **Learning Rate Schedule Plot** across epochs.
- **Predictions & Ground Truth Exported** to CSV for analysis.

### 🧾 Logging and Output Management
- Automatic creation of `logs/` and `Plots/` directories.
- **Final Run Summary** exported to CSV for experiment tracking.

### ✅ Robustness and Error Handling
- **CUDA availability check** with clear runtime errors.
- **Safe initialization** of quantum device (`lightning.gpu`) with error fallback.


---

## 🛠 Tech Stack

| **Component**             | **Technology / Library**                                |
|---------------------------|----------------------------------------------------------|
| **Deep Learning**         | PyTorch (`TorchConnector` from Qiskit ML)               |
| **Quantum Integration**   | Qiskit, Qiskit Machine Learning, Qiskit Aer, cuQuantum, pennylane   |
| **Quantum Backend**       | AerSimulator with GPU support (`cuStateVec`)            |
| **Automatic Differentiation** | PyTorch Autograd with Qiskit EstimatorQNN        |
| **Mixed Precision**       | `torch.cuda.amp` (Autocast & GradScaler)                |
| **Optimization**          | AdamW, ReduceLROnPlateau                                 |
| **Data Handling**         | pandas, NumPy, StandardScaler                            |
| **Progress Tracking**     | tqdm                                                     |
| **GPU Acceleration**      | NVIDIA CUDA (`torch.device("cuda")`)                    |
| **Model Checkpointing**   | `torch.save`, `torch.load`                              |
| **Evaluation & Splits**   | scikit-learn (`train_test_split`)                       |


---

## 🔧 Current Status

- ✅ Model definition (autoencoders + quantum circuit + integration)
- ✅ Qiskit backend configuration for GPU-accelerated simulation (cuQuantum)
- ✅ Mixed precision training using `torch.cuda.amp`
- ✅ Dataset preprocessing and scaling on GPU. Custom dataloaders for GPU based data preprocessing and loading.
- ✅ Transitioned to PennyLane.
- ✅ Optimizing HQNN pipeline to have global parameters and making it more functional to automate it.
- ✅ Trained 36 models in an epirical, systematic manner recording the logs(per epoch), metrics and parameters for different models on 1000 sample out of 22000 (for NIFTY over 4 years) of total dataset for 50 epochs. (1 model is trained with 3 different random seeds). 
- ✅ Performance benchmarking against classical baselines
- ✅ Found out that quant circuits negatively contribute to the learning by a factor of (~ -2.3, qc = 1 - (q_avg / c_avg) where q_avg = sum(q_losses) / len(q_losses) and c_avg = sum(c_losses) / len(c_losses)) .
- ✅ Submitted to npj Quantum Information, under consideration.
- ✅ Patent pending.
 
Long term goals (next step)
- 🚧 See you at AAAI 2025 and ICLR 2025 !
- 🚧 Performing Entaglemnet Entropy analysis for data to check for volume law for data in HEA (hardware efficient ansatz). 
- 🚧 Explore Sparse HEA, and compare to current HEA to check for mitigation techniques for barren plateaus. 
- 🚧 Include noise simulation to simulate real quantum hardware
- 🚧 Compute on both data sets (Two major stock indices from 2 different markets NSE (NIFTY) and Chicago stock exchange (SNP500))
- 🚧 Implement on real quantum hardware, IBM runtime, Google, look for options.
- 🚧 Reach out to institutes and companies for more compute power and funding to expand to entire dataset spanning over 3 years to analyse performacnce large set of data .
---

## 💬 Collaboration

I’m open to collaboration with:

- **Quantum ML researchers** interested in hybrid systems
- **Optimization experts** with experience in GPU acceleration or circuit transpilation or HPC.
- **Labs or academic mentors** working on interpretable quantum ML
- **Open-source contributors** who want to help shape the future of quantum-classical ML

Feel free to open a GitHub Issue or connect directly if you'd like to collaborate or follow along.

---
## 📁 Current Project Structure (WIP) 

```bash
HQNN/
├── Qiskit + cuQuantum Pipeline #  (abstracted)
├── Qiskit + Pennylane Pipeline #  (abstracted)
├── Autoencoder                 # Jupyter notebooks for deciding architecture of classical pre and post encoders
├── environment.yml             # Conda environment file
├── requirements.txt            # PIP dependencies (if not using conda)
├── README.md                   # Project overview
├── .gitignore
```
## 📁 Final Project Structure (WIP) (Expected)

Empty folder are deliberately kept empty to maintain confedentiality of the research

```bash
HQNN/
├── notebooks/               # Jupyter notebooks for experiments, EDA, prototyping
│   └── pipeline_experiment.ipynb
│   └── qiskit_pipeline_v1.ipynb
│
├── src/                    # Source code (cleaned-up Python files)
│   ├── __init__.py
│   ├── model/              # Quantum + classical model components
│   │   ├── autoencoders.py
│   │   ├── hybrid_model.py
│   │   └── quantum_circuit.py
│   ├── training/           # Training and evaluation logic
│   │   ├── trainer.py
│   │   └── eval.py
│   ├── utils/              # Helpers for preprocessing, logging, plotting
│   │   ├── data_loader.py
│   │   ├── metrics.py
│   │   └── visualizer.py
│   └── config.py           # Configs, hyperparameters, paths
│
├── data/                   # Processed input data (not tracked in git)
│   └── raw/            
│   └── processed/
│
├── checkpoints/            # Model checkpoints (not tracked in git)
│
├── outputs/                # Logs, plots, evaluation results
│   └── predictions/
│
├── tests/                  # Unit tests for components
│   └── test_model.py
│
├── environment.yml         # Conda environment file
├── requirements.txt        # PIP dependencies (if not using conda)
├── README.md               # Project overview
├── .gitignore
└── run_pipeline.py         # Not active as project in progress


