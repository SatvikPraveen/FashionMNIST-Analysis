# # 🎨 **Fashion MNIST Analysis** 🧥👕👗👟

![MIT License](https://img.shields.io/github/license/SatvikPraveen/FashionMNIST-Analysis)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![Repo Size](https://img.shields.io/github/repo-size/SatvikPraveen/FashionMNIST-Analysis)
![Issues](https://img.shields.io/github/issues/SatvikPraveen/FashionMNIST-Analysis)
![Stars](https://img.shields.io/github/stars/SatvikPraveen/FashionMNIST-Analysis?style=social)

> An in-depth exploration of fashion item classification using the Fashion MNIST dataset.

Welcome to the **Fashion MNIST Analysis** project, where we dive deep into data exploration, model training, and fine-tuning to classify fashion items with **state-of-the-art techniques**.

---

## **Overview**

This project focuses on analyzing the **Fashion MNIST** dataset using various Convolutional Neural Networks (CNNs), including **MiniCNN**, **TinyVGG**, and **ResNet**. The models were trained with a production-ready pipeline featuring correct Fashion-MNIST normalization, val_acc-based early stopping, and automatic best-model selection. **TinyVGG** achieved the highest performance across all three architectures.

---

## **Key Features**

- **Traditional Machine Learning Models**: Classification using **Random Forest**, **k-Nearest Neighbors**, and **XGBoost**, combined with dimensionality reduction techniques like PCA, t-SNE, and UMAP.
- **Custom Baseline Models**: Implementation of lightweight **MiniCNN**, **TinyVGG**, and **ResNet** architectures.
- **Fine-Tuning Pipeline**: A robust pipeline to tune hyperparameters such as learning rates, batch sizes, and early stopping patience values.
- **Visualization Tools**:
  - Confusion matrices for baseline and fine-tuned models.
  - Sample predictions for visual validation.
- **Metrics Comparison**: Detailed comparison of baseline vs. fine-tuned models using accuracy, precision, recall, and F1-score.
- **Reproducibility**: Scripts are modular and adaptable for other datasets.

---

## **Project Structure**

```bash
FashionMNIST-Analysis/
├── data/
│   ├── processed/          # Train/val/test CSVs (fashion_mnist_{train,val,test}.csv)
│   └── FashionMNIST/raw/   # Raw binary files downloaded by torchvision
├── eda/
│   └── EDA.ipynb           # Exploratory data analysis notebook
├── figures/
│   ├── EDA_plots/          # EDA visualizations
│   ├── evaluation_plots/   # Confusion matrix and prediction visualizations
│   ├── modeling_plots/     # Plots generated during model training
│   └── Traditional_ML_Algo_plots/  # Traditional ML confusion matrices
├── models/
│   ├── all_models/
│   │   ├── minicnn/        # MiniCNN checkpoint + training history
│   │   ├── tinyvgg/        # TinyVGG checkpoint + training history
│   │   └── resnet/         # ResNet checkpoint + training history
│   └── best_model_weights/ # Best overall model weights + best_model_info.json
├── notebooks/              # Jupyter notebooks for the workflow
├── results/
│   ├── evaluation_results/ # predictions_vector.csv, evaluation_metrics.csv
│   └── Traditional_ML_Algo_results/
├── src/
│   ├── cli/                # train.py, evaluate.py, finetune.py, prepare_data.py
│   ├── data/               # dataset.py, augmentation.py, preparation.py
│   ├── models/             # architectures.py, ensemble.py, transfer.py
│   ├── training/           # trainer.py, tuner.py, utils.py
│   ├── evaluation/         # evaluate.py, metrics.py, explainability.py
│   └── serving/            # inference.py, api.py
├── tests/                  # Unit tests (test_models.py, test_inference.py, test_utils.py)
├── apps/                   # Streamlit dashboard and Gradio app
├── docker/                 # Dockerfile and docker-compose.yml
├── docs/                   # FEATURES.md, DEPLOYMENT.md, USAGE_GUIDE.md, etc.
├── config.yaml             # All training/augmentation hyperparameters
├── requirements.txt
└── setup_project.py
```

---

### **Description of Key Components**

- **`data/`**: Directory for storing raw data.
- **`data_preparation/`**: Processed CSV files for training, validation, and testing datasets.
- **`eda/`**: Designated folder for EDA
  - **`EDA.ipynb`**: For exploratory data analysis.
- **`figures/`**:
  - **`EDA_plots/`**: Visualizations from Exploratory Data Analysis.
  - **`evaluation_plots/`**: Visualizations and metrics for model evaluation.
  - **`modeling_plots/`**: Figures generated during model training.
  - **`fine_tuning_plots/`**: Figures generated during fine-tuning of models.
- **`models/`**:
  - **`all_models/`**: Contains saved weights for all trained models.
  - **`best_model/`**: Contains the final best model after evaluation.
  - **`best_model_weights/`**: Weights of the best-performing model.
- **`notebooks/`**: Jupyter notebooks for the entire workflow:
  - **`modeling.ipynb`**: For training baseline models.
  - **`finetuning.ipynb`**: For fine-tuning models with hyperparameter optimization.
  - **`evaluation.ipynb`**: For evaluation and comparison of models.
- **`src/`**:
  - **`model_definitions.py`**: Contains all model architecture definitions.
  - **`utils.py`**: Utility functions for training, testing, and evaluation.
  - **`evaluation.py`**: Handles model evaluation, including metrics calculation and prediction visualization.
- **`README.md`**: Project documentation and execution details.
- **`requirements.txt`**: Required libraries and dependencies for the project.
- **`setup_project.py`**: Script for creating the project directory structure.
- **`main.py`**: Evaluates the best-trained model, generating predictions, metrics, and visualizations.

---

## Dataset

- **Fashion MNIST** is a dataset of Zalando’s article images, consisting of **60,000 training** and **10,000 testing** grayscale images in **10 classes**.
- Each image is **28x28 pixels**.

![EDA Visualization](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/EDA_plots/sample_images_grid.png)

## Class Labels

Below are the 10 class labels for the Fashion MNIST dataset:

| **Class Label** | **Examples** |
| --------------- | ------------ |
| T-shirt/top     | 👕           |
| Trouser         | 👖           |
| Pullover        | 🧥           |
| Dress           | 👗           |
| Coat            | 🧥           |
| Sandal          | 🩴           |
| Shirt           | 👔           |
| Sneaker         | 👟           |
| Bag             | 👜           |
| Ankle Boot      | 🥾           |

---

## **Steps in the Workflow**

### **1. Exploratory Data Analysis**

- Distribution of labels.
- Sample image visualizations.
- Data normalization and preprocessing.

### **2. Traditional Machine Learning**

- Models Used:
  - **Random Forest**
  - **k-Nearest Neighbors**
  - **XGBoost**
- Dimensionality Reduction Techniques:
  - **PCA**: Principal Component Analysis for feature reduction.
  - **t-SNE**: Non-linear dimensionality reduction for visualization.
  - **UMAP**: Uniform Manifold Approximation and Projection for clustering and analysis.
- Hyperparameter Tuning:
  - Grid search optimization for all models.
- **Evaluation Metrics**:
  - Confusion matrices, accuracy, and classification reports for each model.

### **3. CNN Baseline Modeling**

- Architectures implemented:
  - **MiniCNN**: A lightweight custom CNN.
  - **TinyVGG**: Inspired by VGG architecture, with fewer layers.
  - **ResNet**: Residual Network with skip connections for better gradient flow.
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score for all models.

### **4. Fine-Tuning**

- **Hyperparameter Grid**:
  - Learning Rates: `[1e-5, 5e-6]`
  - Batch Sizes: `[32, 64]`
  - Early Stopping Patience: `[2, 3]`
- **Best Model Selection**:
  - **TinyVGG** achieved the highest test accuracy (**93.21%**) and was automatically saved to `models/best_model_weights/`.

### **5. Evaluation**

- Confusion matrix and prediction visualization for the best model (TinyVGG).
- **Evaluation Metrics**:
  - MiniCNN test accuracy: **89.93%**
  - ResNet test accuracy: **91.46%**
  - **TinyVGG test accuracy: 93.21%** ✅ Best
- **Visualization**:
  - Sample predictions from the best TinyVGG model.

---

## **Key Visualizations**

### Confusion Matrix - Best Model (TinyVGG, 93.21% test accuracy)

![Confusion Matrix](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/evaluation_plots/confusion_matrix.png)

### Prediction Visualization

Sample predictions from the best TinyVGG model:

![Prediction Visualization](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/evaluation_plots/prediction_visualization.png)

---

## **Results**

| Metric        | MiniCNN | ResNet | **TinyVGG (Best)** |
| ------------- | ------- | ------ | ------------------ |
| **Accuracy**  | 0.8993  | 0.9146 | **0.9321**         |
| **Precision** | 0.8995  | 0.9149 | **0.9323**         |
| **Recall**    | 0.8993  | 0.9146 | **0.9321**         |
| **F1-Score**  | 0.8992  | 0.9146 | **0.9321**         |

---

## **🆕 New Features (2026)**

### **Production-Ready Training Pipeline**

- ✅ **End-to-End CLI Scripts**: `train.py`, `finetune.py`, `prepare_data.py`
- ✅ **Data Augmentation**: Mixup, CutMix, RandomErasing, torchvision transforms - fully integrated!
- ✅ **Multi-Device Support**: Auto-detects CUDA, MPS (Apple Silicon M1/M2/M3), or CPU
- ✅ **Config-Driven**: All parameters in `config.yaml` for reproducibility
- ✅ **Model Checkpointing**: Saves best models automatically with early stopping
- ✅ **Comprehensive Logging**: Training history, metrics tracking, JSON outputs

### **Quick Start - New Pipeline**

```bash
# 1. Prepare data
python src/cli/prepare_data.py

# 2. Train all models (MiniCNN, TinyVGG, ResNet) — best model auto-saved
python src/cli/train.py --model all \
  --use-csv \
  --train-csv data/processed/fashion_mnist_train.csv \
  --val-csv   data/processed/fashion_mnist_val.csv \
  --test-csv  data/processed/fashion_mnist_test.csv

# 3. Evaluate best model (architecture auto-detected from best_model_info.json)
python src/cli/evaluate.py \
  --model_path models/best_model_weights/best_model_weights.pth \
  --test_csv   data/processed/fashion_mnist_test.csv

# 4. Fine-tune best model
python src/cli/finetune.py \
  --model tinyvgg \
  --pretrained models/best_model_weights/best_model_weights.pth
```

**📖 See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete instructions**

---

## **How to Run**

### Environment Setup

Follow these steps to set up the project on your local machine:

1. **Clone this repository**:  
   Clone the FashionMNIST-Analysis repository to your local machine.
   ```bash
   git clone https://github.com/SatvikPraveen/FashionMNIST-Analysis.git
   cd FashionMNIST-Analysis
   ```
2. **Create an environment**:  
   Set up a virtual Python environment to manage dependencies.

   - For Linux/MacOS:

     ```bash
     python -m venv envf
     source envf/bin/activate
     ```

   - For Windows:
     ```bash
     python -m venv envf
     envf\Scripts\activate
     ```

3. **Install dependencies:**

   Install all required Python libraries listed in `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the setup script:**

   Initializes the project by creating necessary directories

   ```bash
   python setup_project.py
   ```

## **Execution**

- **Exploratory Data Analysis**: `eda.ipynb`
- **Traditional ML Algorithms**:`Traditional_ML_Algo.ipynb`
- **Model Training**: `modeling.ipynb`
- **Fine-Tuning**: `finetuning.ipynb`
- **Evaluation**: `evaluation.ipynb`

---

## **Evaluating the Model**

### **Using the `evaluate.py` CLI**

Evaluates the best model. Architecture is auto-detected from `models/best_model_weights/best_model_info.json`.

- **Plots** are saved to `figures/evaluation_plots/`:
  - `confusion_matrix.png`
  - `prediction_visualization.png`
- **CSVs** are saved to `results/evaluation_results/`:
  - `predictions_vector.csv`
  - `evaluation_metrics.csv`

#### **Command to Run**

```bash
python src/cli/evaluate.py \
  --model_path models/best_model_weights/best_model_weights.pth \
  --test_csv   data/processed/fashion_mnist_test.csv
```

Optional overrides:
- **`--model_name`**: Force architecture (`ResNet`, `TinyVGG`, `MiniCNN`). Auto-detected if omitted.
- **`--figures_dir`**: Override plot output directory (default: `figures/evaluation_plots`).
- **`--results_dir`**: Override CSV output directory (default: `results/evaluation_results`).

---

## **Technologies Used**

- **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn, SciPy

  - Essential libraries for data manipulation, statistical analysis, and visualization.

- **Machine Learning and Data Mining**: Scikit-learn

  - Provides tools for traditional ML models and techniques like PCA and t-SNE.

- **Deep Learning (PyTorch)**: PyTorch, TorchVision, Pillow

  - PyTorch and TorchVision are used for designing, training, and evaluating neural networks. Pillow is used for image preprocessing.

- **Jupyter Notebooks for Analysis**: Jupyter, IPython, IPyKernel, Notebook

  - Enables interactive analysis and visualization in Jupyter Notebook environments.

- **Progress Bar**: TQDM

  - Adds progress bars to loops and processes for better tracking.

- **Generating Model Summaries**: TorchInfo

  - Generates detailed summaries of PyTorch models, including layer-wise parameters and memory usage.

- **Dimensionality Reduction and XGBoost**: UMAP-learn, XGBoost
  - UMAP for advanced dimensionality reduction and XGBoost for traditional gradient-boosted models.

---

## **Acknowledgments**

This project is inspired by the Fashion MNIST dataset provided by Zalando Research. Special thanks to open-source contributors of **PyTorch** and **Scikit-learn** for enabling this work.

For a detailed explanation of this project, refer to the accompanying [blog post](https://medium.com/@meetdheerajreddy/fashion-mnist-analysis-classifying-fashion-with-deep-learning-0ba793ba5234).

---

## Implementation Notebooks

To explore the different stages of the project workflow, you can access the following Jupyter notebooks:

- **[Data Preparation](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/DataPreparation.ipynb)**: Prepares the dataset for performing tasks
- **[Exploratory Data Analysis (EDA)](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/eda/EDA.ipynb)**: Visualizations and preprocessing steps for Fashion MNIST.
- **[Traditional Machine Learning](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/Traditional_ML_Algo.ipynb)**: Implementation of Random Forest, k-NN, and XGBoost models with dimensionality reduction techniques.
- **[Model Training](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/modeling.ipynb)**: Training baseline CNN models like MiniCNN, TinyVGG, and ResNet.
- **[Fine-Tuning](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/finetuning.ipynb)**: Hyperparameter tuning for the CNN models.
- **[Evaluation](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/evaluate_best_model.ipynb)**: Model evaluation, confusion matrices, and metrics comparison.

You can access the full repository [here](https://github.com/SatvikPraveen/FashionMNIST-Analysis).

## **Documentation**

For comprehensive guides and documentation, please refer to the `docs/` folder:

- **[FEATURES.md](docs/FEATURES.md)** - Complete feature documentation (800+ lines) covering all new modules and capabilities.
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment guide for Docker, Kubernetes, and cloud platforms.
- **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)** - Detailed implementation report of the modernization project.
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Guidelines for contributing to the project.
- **[CODE_OF_CONDUCT.md](docs/CODE_OF_CONDUCT.md)** - Community code of conduct.

---

## **Future Work**

- Explore transfer learning with pretrained models like **ResNet50**, **EfficientNet**, or **Vision Transformers (ViTs)**.
- Implement model ensembling for improved predictions.
- Extend dimensionality reduction techniques like **t-SNE** and **UMAP** to more components and integrate them into end-to-end pipelines.
- Test the best model on unseen real-world data.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### **Let’s Classify Fashion Together!** 👕👗🧥👠👟
