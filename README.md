# Hybrid Quantum Neural Network (HQNN)

A hybrid classical–quantum neural network (HQNN) designed to explore quantum advantages in machine learning for regression tasks on structured datasets.

> 🚧 This project is a work in progress. Contributions, feedback, and collaboration inquiries are welcome.

---

## ✨ Overview

This project combines classical neural networks with quantum circuits using Qiskit and PyTorch to build a hybrid model for predictive tasks. The goal is to evaluate whether quantum-enhanced layers can contribute to learning efficiency or expressivity in practical settings.

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
| **Quantum Integration**   | Qiskit, Qiskit Machine Learning, Qiskit Aer, cuQuantum   |
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
- ✅ Qiskit backend configuration for GPU-accelerated simulation
- ✅ Mixed precision training using `torch.cuda.amp`
- ✅ Dataset preprocessing and scaling
- 🚧 Performance benchmarking against classical baselines
- 🚧 Preparing transition to PennyLane for framework comparison

---

## 💬 Collaboration

I’m open to collaboration with:

- **Quantum ML researchers** interested in hybrid systems
- **Optimization experts** with experience in GPU acceleration or circuit transpilation
- **Labs or academic mentors** working on interpretable quantum ML
- **Open-source contributors** who want to help shape the future of quantum-classical ML

Feel free to open a GitHub Issue or connect directly if you'd like to collaborate or follow along.

---

## 📁 Project Structure (WIP)

```bash
├── models/              # HQNN components (autoencoders, hybrid model)
├── training/            # Training scripts and evaluation logic
├── utils/               # Dataset handling and preprocessing
├── checkpoints/         # Model checkpoints (not tracked in repo)
├── README.md
└── requirements.txt     # To be added

