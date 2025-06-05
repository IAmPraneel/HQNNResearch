# Hybrid Quantum Neural Network (HQNN)

A hybrid classical–quantum neural network (HQNN) designed to explore quantum advantages in machine learning for regression tasks on Option pricing on NIFTY  Options datasets (cleaned and preprocessed).

> 🚧 This project is a work in progress. Contributions, feedback, and collaboration inquiries are welcome.

---

## ✨ Overview

This project combines classical neural networks with quantum circuits using Qiskit and PyTorch to build a hybrid model for predictive tasks. The goal is to evaluate whether quantum-enhanced layers can contribute to learning efficiency or expressivity in practical settings.

---
## 📌 Latest update: Working on 1st draft for preprint to showcase proof of concept and attract collaborators and Institutes for computational resources. Found out that quant circuits negatively contribute to the learning by a factor of (~2.3, qc = 1 - (q_avg / c_avg) where q_avg = sum(q_losses) / len(q_losses) and c_avg = sum(c_losses) / len(c_losses)) . Observed barren plateaus for even small shallow quantum circuits.

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
- ✅ Trained 36 models in an epirical, systematic manner recording the logs(per epoch), metrics and parameters for different models on 3 subsets of 100, 500, 1000 samples out of 22000 (for NIFTY over 4 years) of total dataset for 20 epochs. (1 model is trained thrice with 3 different random seeds). The smaller dataset analysis is to check for low data environment situations and 1000->500->100 samples to monitor change in model behaviour with respect to data scale.
- ✅ Benchmark HQNN models with different params and create comparitive visualizations and interpret them.
- ✅ Performance benchmarking against classical baselines
- ✅ Found out that quant circuits negatively contribute to the learning by a factor of (~ -2.3, qc = 1 - (q_avg / c_avg) where q_avg = sum(q_losses) / len(q_losses) and c_avg = sum(c_losses) / len(c_losses)) .
- ✅ Barren plateaus appear early (n ≥ 3), even in shallow circuits. With gradient efficiency = 1 for all models & gradient efficiency scaling linearly with qubits (2*n) instead of expected 4^(n) - 1. Which contradicts theoretical assumptions in current literature. (npj QI)
- 
- 🚧 Performing Entaglemnet Entropy analysis for data to check for volume law for data in HEA (hardware efficient ansatz). (PRX Quantum)
- 🚧 Explore Sparse HEA, and compare to current HEA to check for mitigation techniques for barren plateaus. (Science Advancements, Nature Communication)
- 🚧 Create 1st draft of the results, providing solid proof of work and potential of the idea to approach institutes for computational resources and funding.

Long term goals (next step)
- 🚧 Include noise simulation to simulate real quantum hardware
- 🚧 Compute on both data sets (Two major stock indices from 2 different markets NSE and Chicago stock exchange)
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


