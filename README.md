# 📈 Stock Movement Predictor using ANN

This project uses an Artificial Neural Network (ANN) to classify whether the next day's stock closing price will be higher or lower, based on historical trading data from major tech companies.

---

## 🗂️ Dataset Overview

- **Source**: [Big Tech Giants Stock Price Data on Kaggle](https://www.kaggle.com/datasets/umerhaddii/big-tech-giants-stock-price-data)
- **Period**: January 2010 – January 2023
- **Companies**: Apple, Amazon, Alphabet, Meta, Tesla, NVIDIA, and more
- **Features**:
  - `open`: Price at market open
  - `high`: Highest price of the day
  - `low`: Lowest price of the day
  - `close`: Price at market close
  - `adj_close`: Adjusted close (for splits/dividends)
  - `volume`: Number of shares traded
- **Target**: Binary label (`1` if next day's closing price is higher, else `0`)

---

## 🧠 Model Architecture

A deep ANN built with Keras and TensorFlow:

- Input: 6 trading features
- Hidden Layers:
  - Dense(64) → BatchNormalization → Dropout(0.4)
  - Dense(32) → BatchNormalization → Dropout(0.3)
  - Dense(16)
- Output: Dense(1) with sigmoid activation
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Regularization: Dropout + EarlyStopping

---

## 🔄 Workflow

1. **Data Cleaning**: Remove duplicates and missing values
2. **Target Creation**: Use `shift(-1)` to compare next day's closing price
3. **Feature Selection**: Use core trading metrics
4. **Train-Test Split**: 80/20 with stratification
5. **Scaling**: StandardScaler for normalization
6. **Model Training**: 150 epochs with validation split and early stopping
7. **Evaluation**: Accuracy score and confusion matrix
8. **Visualization**: Training history plotted for performance tracking

---

## 📊 Results

- **Test Accuracy**: ~`[Insert your result here]`
- **Confusion Matrix**:
- **Training History**: Accuracy vs. Epochs plotted using Matplotlib

---

## 🚀 How to Run

### 🧰 Requirements
pip install numpy pandas matplotlib scikit-learn tensorflow keras

## 📜 License
This project is licensed under the MIT License.
