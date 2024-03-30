# update 7.0
# update 5.1


import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load the data
bitcoin_prices = pd.read_csv("/content/BTC-USD new.csv", index_col=["Date"], parse_dates=True)
ethereum_prices = pd.read_csv("/content/ETH-USD (1).csv", index_col=["Date"], parse_dates=True)

# Merge the data
data = pd.merge(bitcoin_prices[['Close']], ethereum_prices[['Close']], on='Date')

# Impute the missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)
data_imputed = pd.DataFrame(data_imputed, columns=['Bitcoin_Close', 'Ethereum_Close'])

# Calculate the spread and spread statistics
data_imputed['Spread'] = data_imputed['Bitcoin_Close'] - data_imputed['Ethereum_Close']
data_imputed['Spread_mean'] = data_imputed['Spread'].rolling(window=30).mean()
data_imputed['Spread_std'] = data_imputed['Spread'].rolling(window=30).std()
data_imputed['Spread_Narrowing'] = np.where(data_imputed['Spread'] < (data_imputed['Spread_mean'] - data_imputed['Spread_std']), 1, 0)

# Order the features by importance
data_imputed = data_imputed[['Spread_mean', 'Spread', 'Ethereum_Close', 'Bitcoin_Close', 'Spread_std', 'Spread_Narrowing']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_imputed[['Spread_mean', 'Spread', 'Ethereum_Close', 'Bitcoin_Close', 'Spread_std']],
                                                        data_imputed['Spread_Narrowing'],
                                                        test_size=0.3,
                                                        random_state=35)

# Define a pipeline for preprocessing the data and training the model
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
])

# Preprocess the training and testing data using the pipeline
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create a neural network model using Keras
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add learning rate reduction on plateau and early stopping callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=19, restore_best_weights=True)

# Train the model with specified epochs and callbacks
history = model.fit(X_train_processed, y_train, epochs=150, validation_data=(X_test_processed, y_test), callbacks=[reduce_lr, early_stopping])

# Trading signals based on predictions
predictions = model.predict(X_test_processed)

# Calculate the Sharpe ratio and Sortino ratio
closing_prices = data_imputed[['Bitcoin_Close', 'Ethereum_Close']]
returns = closing_prices.pct_change()

# Training Progress Visualization
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(10, 6))
plt.plot(range(len(train_accuracy)), train_accuracy, label='Training Accuracy')
plt.plot(range(len(val_accuracy)), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Progress')
plt.show()# Calculate daily returns
daily_returns = closing_prices.pct_change()

# Calculate the Sharpe ratio
sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)

# Calculate the Sortino ratio
sortino_ratio = np.mean(daily_returns[daily_returns > 0]) / np.std(daily_returns[daily_returns < 0])

# Print the Sharpe ratio and Sortino ratio
print("Sharpe ratio:", sharpe_ratio)
print("Sortino ratio:", sortino_ratio) # Define transaction fee percentage and market impact percentage
transaction_fee_percent = 0.01
market_impact_percent = 0.02

# Define a function to execute trades
def execute_trade(trade_type, trade_size):
    # ... Perform trade execution logic ...
    print(f"Executing trade: {trade_type}, size: {trade_size}")

# Define trade size
trade_size = 100

# Calculate transaction costs and slippage
def calculate_transaction_costs(trade_size, transaction_fee_percent):
    transaction_fee = trade_size * transaction_fee_percent
    return transaction_fee

def calculate_slippage(trade_size, market_impact_percent):
    slippage = trade_size * market_impact_percent
    return slippage

# Apply transaction costs and slippage to trading signals
for i in range(len(predictions)):
    if predictions[i] == 1:  # Buy signal
        transaction_fee = calculate_transaction_costs(trade_size, transaction_fee_percent)
        slippage = calculate_slippage(trade_size, market_impact_percent)
        adjusted_trade_size = trade_size - transaction_fee - slippage
        execute_trade('BUY', adjusted_trade_size)
    elif predictions[i] == 0:  # Sell signal
        transaction_fee = calculate_transaction_costs(trade_size, transaction_fee_percent)
        slippage = calculate_slippage(trade_size, market_impact_percent)
        adjusted_trade_size = trade_size - transaction_fee - slippage
        execute_trade('SELL', adjusted_trade_size)
