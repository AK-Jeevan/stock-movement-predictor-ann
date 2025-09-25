# This Model is used to predict whether the stock price of a company will be higher or lower using Artificial Neural Networks (ANN)
# Classification Problem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

# Step 1: Load and inspect the data
data = pd.read_csv(r"C:\Users\akjee\Documents\AI\DL\ANN\big_tech_stock_prices.csv")
print(data.head())
print(data.describe())

# Step 2: Clean the data
data = data.drop_duplicates()
data = data.dropna().reset_index(drop=True)

# Create target variable: 1 if next day's closing price is higher, else 0
data['price_movement'] = (data['close'].shift(-1) > data['close']).astype(int)
data.dropna(inplace=True)  # Remove last row with NaN

''' shift(-1)
What it does: Shifts the data in the 'close' column up by one row. This means that each row in the close column has their next row value in them.
i.e. 0    100  after they become 0    105  ← next day's price
     1    105                    1    NaN
'''

# Step 3: Select features (X) and target (y)
x = data.iloc[:, 2:8]
y = data["price_movement"]

# Step 4: Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 6: Build the ANN model (deeper, with batch normalization and dropout)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Step 7: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1, mode='min', min_delta=0.0005
)

# Step 9: Train the model
history = model.fit(
    x_train_scaled, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.25,
    callbacks=[early_stopping],
    verbose=1
)

# Step 10: Evaluate the model
y_pred_prob = model.predict(x_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {acc:.3f}")
print("Confusion Matrix:")
print(cm)

# Step 11: Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Training History')
plt.show()