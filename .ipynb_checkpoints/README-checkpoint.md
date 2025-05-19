## PyTorch Skill Path - Capstone Project Example

### Capstone Overview

#### 🩺 Analyzing Health Factors - Predicting Diabetes and Age
- **Goal:** Predict diabetes diagnosis (binary classification) and patient age (regression) using health-related factors.
- **Dataset:** Subset of a [CDC dataset](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)
    - Smaller, cleaned version available within the [UCI Machine Learning Repo](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
    - License: CC0 Public Domain
- **Models:**
    - Regression Models:
        - Linear regression (baseline)
        - Feedforward Neural Network
    - Classification Models:
        - Feedforward Neural Network
- **Results:**
    - Regression: The FNN outperformed the linear regression baseline when predicting age, achieving lower MSE scores on the testing set.
    - Classification: The FNN achieved an accuracy of 76% and decently balanced performance across both classes (F1-score of 0.77 for class 1 and an F1-score of 0.74 for class 0).

#### 📚 Classifying Medical Text
- **Goal:** Classify medical question-answer pairs into focus areas (multiclass classification) from trusted medical sources.
- **Dataset:** [MedQuAD dataset](https://github.com/abachaa/MedQuAD/tree/master) from the research paper [A Question-Entailment Approach to Question Answering](https://arxiv.org/pdf/1901.08079)
    - License: CC BY 4.0
- **Models:**
    - Feedforward Neural Network
    - Specialized BERT Transformer ([BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)) 
- **Results:**
    - The FNN achieved ~83% accuracy on the testing set
    - The specialized BERT achieved ~98% accuracy on the testing set, a more balanced performance across all five classes (higher precision, recall, and F1-scores)
    - The specialized BERT demonstrated better understanding of medical text

#### 👁 Classifying Retinal Images for Diabetes Retinopathy 
- **Goal:** Classify high-quality retinal fundus images for diabetes retinopathy (binary classification). 
- **Dataset:** [IDRiD dataset](https://idrid.grand-challenge.org/Data/)
    - License: CC BY 4.0 
- **Models:**
    - Convolutional Neural Network
    - [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224)
- **Results:**
    - The CNN achieved ~58% accuracy on the testing set
    - The ViT achieved ~82% accuracy on the testing set
    - The ViT showed better performance across both classes (higher precision, recall, and F1-scores)

### Getting Started

To explore and run the code for this capstone project, we suggest creating an environment containing the required list of libraries and their versions, which is included in the `requirements.txt` file. 

The notebooks can run locally or on the cloud using Jupyter-compatible environments (recommended) like:
- Google Colab
- Kaggle Notebooks
- Paperspace Gradient
- Deepnote

💡 **Tip:** Enabling the GPU is highly recommended, especially for the image classification task, which involves classifying retinal fundus images.

### Install Dependencies

All required libraries and their versions can be installed using:

```bash
pip install -r requirements.txt
```