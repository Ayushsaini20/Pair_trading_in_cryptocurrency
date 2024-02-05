#Pair Trading

The provided Python code orchestrates a comprehensive analysis and implementation of a cryptocurrency trading strategy. 
Leveraging historical price data for Bitcoin and Ethereum, the code commences by merging datasets and imputing missing values to ensure a robust foundation for subsequent analysis.
Subsequently, the script engages in feature engineering, calculating the spread between Bitcoin and Ethereum prices and computing statistical measures such as mean and standard deviation. 
It also introduces a dynamic indicator, signaling spread narrowing based on predefined conditions.
Moving forward, the code delves into machine learning, employing a neural network model implemented with TensorFlow and Keras.
The model is trained on a split dataset, undergoes preprocessing via a pipeline, and incorporates callback mechanisms for learning rate reduction and early stopping during training.
The resultant model is then applied to predict trading signals on a testing set.
An integral aspect of the trading strategy lies in the translation of predicted probabilities into actionable buy or sell decisions. 
A threshold is established to determine the strength of these signals, with the code printing executed trades, their sizes, and corresponding predicted probabilities. 
Additionally, the code assesses the strategy's performance by computing transaction costs and slippage, ultimately calculating daily returns and evaluating key financial metrics, such as the Sharpe and Sortino ratios.
The script doesn't merely conclude with model training and evaluation; it further extends into visualization, plotting the training progress in terms of accuracy over epochs. 
This visual representation aids in understanding the model's learning trajectory. Importantly, the code emphasizes adaptability by including a note about a missing function (calculate_transaction_costs), which is expected to be provided separately. 
This ensures that users can integrate custom transaction cost logic tailored to their specific requirements.
In essence, the code encapsulates a sophisticated yet adaptable approach to cryptocurrency trading, seamlessly integrating machine learning, financial metrics, and visualization for a comprehensive analysis of historical data and the implementation of a robust trading strategy.

<img width="574" alt="Screenshot 2024-01-14 at 5 23 08 PM" src="https://github.com/Ayushsaini20/Pair_trading_in_cryptocurrency/assets/73630171/457b9825-3d7b-4c32-91b0-4dd547158274">
<img width="511" alt="Screenshot 2024-01-14 at 5 23 15 PM" src="https://github.com/Ayushsaini20/Pair_trading_in_cryptocurrency/assets/73630171/6fa0ee2f-799f-4ae4-8a27-715daa1fc376">
<img width="804" alt="Screenshot 2024-01-14 at 5 23 01 PM" src="https://github.com/Ayushsaini20/Pair_trading_in_cryptocurrency/assets/73630171/e4a39532-0c1a-4340-b141-98111a7344c3">
