# Hybrid Quantum Neural Network (HQNN)

A hybrid classical–quantum neural network (HQNN) designed to explore quantum advantages in machine learning for regression tasks on structured datasets.

> 🚧 This project is a work in progress. Contributions, feedback, and collaboration inquiries are welcome.

---

## ✨ Overview

This project combines classical neural networks with quantum circuits using Qiskit and PyTorch to build a hybrid model for predictive tasks. The goal is to evaluate whether quantum-enhanced layers can contribute to learning efficiency or expressivity in practical settings.

---
## 📌 Latest update: Switched to pennylane, implemented 27 HQNN models on subset of dataset(1K samples) for 20 epochs. Recording the logs, parameters and metrics. (9 models, each 3 times with different seed for rigorous results). Based on summary performance 5 models were selected as 'best' and out of those based on epoch wise analysis 2 were chosen for noisy simulation/real quantum hardware tests (IBM Runtime). 

---

## 🧠 Approach

- **Pre-Autoencoder**: Reduces input dimensionality using classical neural layers.
- **Quantum Circuit**: Parametrized unitary operations and entanglement for core processing.
- **Post-Autoencoder**: Classical decoding layer for final regression output.
- **Full Pipeline**: Fully differentiable, GPU-accelerated, and designed for mixed precision.

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
- ✅ Trained 30+ models in an epirical, systematic manner recording the logs(per epoch), metrics and parameters for different models on a subset of 1000 samples out of 22000 of total dataset for 20 epochs. (1 model is trained thrice with 3 different random seeds)
- ✅ Benchmark HQNN models with different params and create comparitive visualizations and interpret them.
- 🚧 Performance benchmarking against classical baselines
- 🚧 Create 1st draft of the results, providing solid proof of work and potential of the idea to approach institutes for computational resources and funding.

- 🧠 Imporvement opportunity identified:
-    - rn the autoencoders are optimized for 5 qubit circuits, in next iteration optimize them for 2,3,4 qubit systems also. RN complete this task with noisy simulations, and preprint then go for refinement.
     - rn standard scaler was used to scale the data, which might not be appropriate for option pricing.
  
Long term goals (next step)
- 🚧 Include noise simulation to simulate real quantum hardware
- 🚧 Compute on both data sets (Two major stock indices from 2 different markets NSE and Chicago stock exchange)
- 🚧 Implement on real quantum hardware, IBM runtime, Google, look for options
-  
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
├── ETEPipeline             # Latest progress on the HQNN pipeline (abstracted)
├── Autoencoder             # Jupyter notebooks for deciding architecture of classical pre and post encoders
├── environment.yml         # Conda environment file
├── requirements.txt        # PIP dependencies (if not using conda)
├── README.md               # Project overview
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


