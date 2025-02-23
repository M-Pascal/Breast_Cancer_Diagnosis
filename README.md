# **Project:** Breast_Cancer_Diagnosis
## Project Description 
This project focuses on developing an AI-driven platform for early breast cancer detection in Rwanda. By leveraging machine learning algorithms, the model aims to analyze patient data including socio-economic factors and lifestyle habits—to improve screening accuracy and accessibility. The ultimate goal is to integrate this cost-effective solution into Rwanda’s healthcare system, empowering clinicians with reliable diagnostic support and reducing late stage diagnoses and mortality rates.

## Problem Statement
Early breast cancer detection is hindered by inaccurate screenings, especially for dense breast tissue, and limited healthcare access in Rwanda. A smarter, reliable tool is needed to improve accuracy and make screening accessible for all.

This dataset contains breast cancer diagnostic data, with features derived from tumor cell characteristics. The **diagnosis** column indicates whether a tumor is **malignant (M)** or **benign (B)**. The dataset is used to train machine learning models for **early breast cancer detection**, addressing issues of accuracy and accessibility in Rwanda.

## Project Structure
```
|-- Summative_Intro_to_ml_[Pascal_Mugisha]_assignment.ipynb
|-- saved_models/          # Directory containing trained models
|   |-- logistic_model.pkl # Logistic regression model (non-neural network)
|   |-- model_1.h5         # Model from First neural network
|   |-- model_2.h5         # model from Second neural network
|   |-- model_3.h5         # Model from Third neural network 
|   |-- model_4.h5         # model Fourth neural network
|   |-- model_5.h5         # Model from Fifth neural network
|-- README.md              # Project documentation
|-- requirements.txt       # List of required dependencies
```

## Dataset Description
The dataset consists of multiple features derived from tumor cell characteristics, including:
- **Mean radius, texture, perimeter, area, smoothness**
- **Worst and standard error of each feature**
- **Diagnosis (Target variable: M = Malignant, B = Benign)**

## Model Implementations
This project implements **Classical model(Logistic regression)** plus**five models**, including four deep learning models and one traditional machine learning model:
1. **Logistic Regression** - A simple but effective baseline model.
2. **Neural Network Model 1** - Basic architecture with 3 hidden layers.
3. **Neural Network Model 2** - Optimized with dropout and different activation functions.
4. **Neural Network Model 3** - Uses a different optimizer and learning rate.
5. **Neural Network Model 4** - Further refined with batch normalization and different regularization.
6. **Neural Network Model 5** - used different optimization techniques

## Performance Metrics
Each model is evaluated using the following metrics:
- **Accuracy**
- **Loss**
- **F1-score**
- **Precision**
- **Recall**

## Neural Network Model Summary

| Training Instance | Optimizer Used | Regularizer Used | Epochs | Early Stopping | Number Of Layers | Learning Rate | Accuracy | Recall | Precision | F1 Score |
| ----------------- | -------------- | ---------------- | ------ | -------------- | ---------------- | ------------- | -------- | ------ | -------- | --------- |
| Instance 1        | None           | None             | None     | False             | 3                | Default       | 96.49%      | 0.9375   | 0.9375     | 0.9375     |
| Instance 2        | Adam           | L2               | 50     | True            | 3                | 0.0001        | 98.25%      | 1.000   |0.9412     | 0.9697     |
| Instance 3        | SGD        | L2               | 70     | True            | 5                | 0.005        | 98.25%      | 1.000   | 0.9412     | 0.9697      |
| Instance 4        | RMSprop          | L1               | 100     | True            | 4                | 0.00001        | 98.25%      | 1.000   | 0.9412     | 0.9697      |
| Instance 5        | Adagrad        | L1\_L2           | 150     | True            | 4                | 0.0001         | 100%      | 1.000   | 1.000     | 1.000      |

## Classical Model (Logistic Regression) Vs Neural Network Model (All models)
**Analysis:**
- The classical Logistic Regression model achieves an F1-score of 0.98, which is on par with the neural networks using Adam, SGD, and RMSprop.
- Instance 5 (Neural Network with Adagrad + L1_L2 Regularization) outperforms the classical model with perfect scores (100%). Although `perfect is Imperfection in ML` as Facilitator said.
- Instance 1 (no regularization) performs the worst among the neural networks, with an F1-score of 0.9375.
- Neural Networks with L2 or L1 regularization (Instances 2, 3, and 4) match the classical model’s 98.25% accuracy and have slightly higher recall but slightly lower precision.


## How to Run the Notebook
### 1. Install Dependencies
Make sure you have all the required dependencies installed.

### 2. Open and Run the Jupyter Notebook
Launch Jupyter Notebook and open `Summative_Intro_to_ml_[Pascal_Mugisha]_assignment.ipynb`. Run all the cells sequentially to train and evaluate the models.
```bash
jupyter notebook
```

### 3. Load the Best Model for Prediction
To use the best-performing model to make predictions:
```python
from tensorflow.keras.models import load_model
import numpy as np

def make_predictions(model_path, X):
    model = load_model(model_path)
    probabilities = model.predict(X)
    predictions = (probabilities > 0.5).astype("int32")
    return predictions
```
Load the best model and make predictions:
```python
best_model_path = "saved_models/best_model.h5"
y_pred_best = make_predictions(best_model_path, X_test)
```

## Future Improvements
- Incorporate **transfer learning** using pre-trained deep learning models.
- Develop a **mobile or web-based tool** for real-world deployment.
- Improve model **generalization** by using data augmentation techniques.

## Conclusion
This project provides a machine learning-based approach to improve **early breast cancer detection** in Rwanda. By leveraging deep learning and traditional models, it aims to enhance screening accuracy and accessibility for better patient outcomes.

Here is link to the [YouTube_Demo video.mp4](https://youtu.be/cmCICOqp16g)

Github Link to [Notebook](https://github.com/M-Pascal/Breast_Cancer_Diagnosis/blob/main/Summative_Intro_to_ml_%5BPascal_Mugisha%5D_assignment.ipynb)