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

This project focuses on analyzing the **Fashion MNIST** dataset using various Convolutional Neural Networks (CNNs), including **MiniCNN**, **TinyVGG**, and **ResNet**. The models were fine-tuned to optimize performance, with evaluation metrics and visualizations providing insights into their efficacy. The fine-tuning process resulted in a **fine-tuned ResNet model** that outperformed the baseline in all key metrics.

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
├── data/  # Contains raw data files and datasets for the project.
├── data_preparation/  # Stores preprocessed datasets (e.g., train, validation, and test splits in CSV format).
├── eda/
│   ├── EDA.ipynb  # Notebook for exploratory data analysis, including visualizations and insights into the dataset.
├── figures/  # Contains subfolders for plots generated during various stages of the project:
│   ├── EDA_plots/  # EDA visualizations.
│   ├── evaluation_plots/  # Model evaluation plots such as confusion matrices and accuracy plots.
│   ├── modeling_plots/  # Figures generated during model training.
│   ├── Traditional_ML_Algo_plots/  # Plots related to Traditional Machine Learning algorithms.
├── models/  # Stores model weights and architecture:
│   ├── all_models/  # Saved weights for all trained models.
│   ├── best_model_weights/  # Weights for the best-performing model.
├── notebooks/  # Jupyter notebooks for different parts of the workflow:
│   ├── modeling.ipynb  # For training baseline models.
│   ├── finetuning.ipynb  # For fine-tuning models with hyperparameter optimization.
│   ├── evaluation.ipynb  # For evaluation and comparison of models.
│   ├── Traditional_ML_Algo.ipynb  # For implementing and evaluating traditional machine learning algorithms.
├── results/  # Stores results from training and evaluation processes:
│   ├── fine_tuning_results/  # Results from hyperparameter tuning of CNN models.
│   ├── Traditional_ML_Algo_results/  # Results from traditional ML models.
├── src/  # Source files containing reusable scripts:
│   ├── model_definitions.py  # Model architectures for CNNs and others.
│   ├── utils.py  # Helper functions for training, evaluation, and visualization.
│   ├── evaluation.py  # Module for evaluation tasks: metrics (precision, recall, F1-score, accuracy), confusion matrices, and visualizations.
├── tests/  # Contains test outputs, including predictions, confusion matrices, and evaluation metrics for verification.
├── README.md  # Comprehensive project documentation, including setup, structure, and instructions.
├── requirements.txt  # Lists all dependencies required to run the project.
├── setup_project.py  # Script for setting up the directory structure and initializing the project environment.
├── main.py  # Main script to evaluate the best-trained model, generate predictions, and produce evaluation metrics.
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
  - Fine-tuned ResNet achieved the highest validation accuracy (**93.23%**).

### **5. Evaluation**

- Confusion matrices for **baseline ResNet** and **fine-tuned ResNet** models.
- **Evaluation Metrics**:
  - Baseline ResNet F1-score: **90.95%**
  - Fine-tuned ResNet F1-score: **93.21%**
- **Visualization**:
  - Sample predictions from the fine-tuned ResNet model.

---

## **Key Visualizations**

### Confusion Matrix - Baseline ResNet

![Baseline ResNet](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/evaluation_plots/Baseline_ResNet_confusion_matrix.png)

### Confusion Matrix - Fine-Tuned ResNet

![Fine-Tuned ResNet](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/evaluation_plots/Best_ResNet_confusion_matrix.png)

### Prediction Visualization

Below are sample predictions made by the fine-tuned ResNet model, showcasing the model's accuracy in identifying fashion items:

![Prediction Visualization](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/evaluation_plots/prediction_visualization.png)

---

## **Results**

| Metric        | Baseline ResNet | Fine-Tuned ResNet |
| ------------- | --------------- | ----------------- |
| **Accuracy**  | 0.9094          | 0.9323            |
| **Precision** | 0.9116          | 0.9323            |
| **Recall**    | 0.9094          | 0.9323            |
| **F1-Score**  | 0.9095          | 0.9321            |

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

### **Using the `main.py` Script**

The `main.py` script enables the evaluation of the best-trained model with the following capabilities:

- **Load Pre-trained Weights**: The script loads pre-trained weights for the best-performing model.
- **Output Predictions and Metrics**: The following outputs are generated during model evaluation and are saved in the `tests/` folder:

  - **Predictions Vector**:

    - Saved as a CSV file (`predictions_vector.csv`) containing the true and predicted labels.

  - **Evaluation Metrics**:

    - Metrics such as accuracy, precision, recall, and F1-score are saved as a CSV file (`evaluation_metrics.csv`).

  - **Visualizations**:
    - Includes:
      - Confusion matrix (`confusion_matrix.png`).
      - Sample prediction visualizations (`prediction_visualization.png`).

  These outputs are helpful for analyzing the model's performance and understanding its predictions visually.

#### **Command to Run**

```bash
python main.py --model_path models/best_model_weights/best_model_weights.pth --test_csv data_preparation/test_data.csv --test_dir tests
```

- **`--model_path`**: Path to the saved model weights.
- **`--test_csv`**: Path to the test dataset in CSV format.
- **`--test_dir`**: Directory to save all outputs.

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
