# # 🎨 **Fashion MNIST Analysis** 🧥👕👗👟

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
├── data/
├── data_preparation/
├── eda/
│   ├── EDA.ipynb
├── figures/
│   ├── EDA_plots/
│   ├── evaluation_plots/
│   ├── modeling_plots/
│   ├── Traditional_ML_Algo_plots/ 
├── models/
│   ├── all_models/
│   ├── best_model_weights/
├── notebooks/
│   ├── modeling.ipynb
│   ├── finetuning.ipynb
│   ├── evaluation.ipynb
│   ├── Traditional_ML_Algo.ipynb
├── results/
│   ├── fine_tuning_results/
│   ├── Traditional_ML_Algo_results/
├── src/
│   ├── model_definitions.py
│   ├── utils.py
├── README.md
├── requirements.txt
├── setup_project.py
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
- **`README.md`**: Project documentation and execution details.
- **`requirements.txt`**: Required libraries and dependencies for the project.
- **`setup_project.py`**: Script for creating the project directory structure.



---

## Dataset

- **Fashion MNIST** is a dataset of Zalando’s article images, consisting of **60,000 training** and **10,000 testing** grayscale images in **10 classes**.
- Each image is **28x28 pixels**.


![EDA Visualization](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/EDA_plots/sample_images_grid.png) 


## Class Labels
Below are the 10 class labels for the Fashion MNIST dataset:

|Col 1|Col 2|Col 3|
|---------------|---------------|---------------|
| T-shirt/top   | Trouser       | Pullover      |
| Dress         | Coat          | Sandal        |
| Shirt         | Sneaker       | Bag           |
| Ankle Boot    |               |               |

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

| Metric         | Baseline ResNet | Fine-Tuned ResNet |
|----------------|-----------------|-------------------|
| **Accuracy**   | 0.9094          | 0.9323           |
| **Precision**  | 0.9116          | 0.9323           |
| **Recall**     | 0.9094          | 0.9323           |
| **F1-Score**   | 0.9095          | 0.9321           |

---

## **How to Run**

### **Environment Setup**
1. Clone this repository:
   ```bash
   git clone https://github.com/SatvikPraveen/FashionMNIST-Analysis.git
   cd FashionMNIST-Analysis
2. Create an environment
   ```bash
   python -m venv envf
   source envf/bin/activate 
4. Install dependencies
   ```bash
   pip install -r requirements.txt
5. Run the setup script
   ```bash
   python setup_project.py

## **Execution**

- **Exploratory Data Analysis**: `eda.ipynb`
- **Traditional ML Algorithms**:`Traditional_ML_Algo.ipynb`
- **Model Training**: `modeling.ipynb`
- **Fine-Tuning**: `finetuning.ipynb`
- **Evaluation**: `evaluation.ipynb`

---

## **Technologies Used**

- **Frameworks**: PyTorch, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Tools**: Jupyter Notebook, NumPy, Pandas
- **Dataset**: Fashion MNIST from Zalando

---

## **Acknowledgments**

This project is inspired by the Fashion MNIST dataset provided by Zalando Research. Special thanks to open-source contributors of **PyTorch** and **Scikit-learn** for enabling this work.

---

## **Future Work**

- Explore transfer learning with pretrained models like **ResNet50** or **EfficientNet**.
- Implement model ensembling for improved predictions.
- Extend dimensionality reduction techniques like **t-SNE** and **UMAP** to more components and integrate them into end-to-end pipelines.
- Test the best model on unseen real-world data.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### **Let’s Classify Fashion Together!** 👕👗🧥👠👟
