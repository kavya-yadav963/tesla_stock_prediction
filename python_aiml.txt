import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Feature Engineering combined into one function
def prepare_tesla_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    # Convert date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    # Technical indicators
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()

    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()

    # Price ratios and ranges
    df['Close_to_Open'] = df['Close'] / df['Open']
    df['High_to_Low'] = df['High'] / df['Low']
    df['High_Low_Range_Pct'] = (df['High'] - df['Low']) / df['Open'] * 100

    # Lagged features (previous days' values)
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)

    # Create target variable - next day's closing price
    df['Next_Close'] = df['Close'].shift(-1)

    # Drop NaN values
    df = df.dropna()
    return df

# 2. Prepare features and target
def prepare_model_data(df):
    X = df.drop('Next_Close', axis=1)
    y = df['Next_Close']
    return X, y

# 3. Build model with simplified parameters
def build_tesla_model(X, y):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

# 4. Evaluate model with key metrics
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Calculate accuracy within various thresholds
    accuracy_5pct = np.mean(np.abs((y - y_pred) / y) <= 0.05) * 100

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Accuracy (within 5% of actual): {accuracy_5pct:.2f}%")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Plot top 10 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance for Tesla Stock Prediction')
    plt.tight_layout()
    plt.show()

    return rmse, mae, r2, accuracy_5pct, feature_importance

# 5. Train and test split, visualization
def train_test_predict(model, df, X, y, test_size=0.2):
    # Split data
    split_idx = int(len(df) * (1 - test_size))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train the model on training data
    model.fit(X_train, y_train)

    # Predict
    y_test_pred = model.predict(X_test)

    # Calculate metrics on test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_accuracy_5pct = np.mean(np.abs((y_test - y_test_pred) / y_test) <= 0.05) * 100

    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test Accuracy (within 5% of actual): {test_accuracy_5pct:.2f}%")

    # Plot results
    plt.figure(figsize=(16, 8))
    plt.plot(df_test.index, y_test, label='Actual', linewidth=2)
    plt.plot(df_test.index, y_test_pred, label='Predicted', linewidth=2)
    plt.title('Tesla Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return test_rmse, test_r2, test_accuracy_5pct

# 6. Completely revised prediction function for a specific period
def predict_month(model, df, year, month):
    # Create dates for the month
    if month == 12:
        next_month = pd.Timestamp(year=year+1, month=1, day=1)
    else:
        next_month = pd.Timestamp(year=year, month=month+1, day=1)

    date_range = pd.date_range(
        start=pd.Timestamp(year=year, month=month, day=1),
        end=next_month - pd.Timedelta(days=1),
        freq='B'  # Business days
    )

    # Check if this period is in our historical data
    month_data = df[(df.index.year == year) & (df.index.month == month)]

    if not month_data.empty:
        print(f"Found actual data for {year}-{month:02d}. Comparing actual vs predicted.")
        # Make a copy of the features to avoid the 'Next_Close' KeyError
        X_period = month_data.copy()
        if 'Next_Close' in X_period.columns:
            X_period = X_period.drop('Next_Close', axis=1)

        y_actual = month_data['Close']
        y_pred = model.predict(X_period)

        # Create comparison DataFrame
        result_df = pd.DataFrame({
            'Date': month_data.index,
            'Actual_Close': y_actual,
            'Predicted_Close': y_pred,
            'Error_Percentage': ((y_actual - y_pred) / y_actual) * 100
        })

        # Plot
        plt.figure(figsize=(14, 7))
        plt.plot(result_df['Date'], result_df['Actual_Close'], label='Actual')
        plt.plot(result_df['Date'], result_df['Predicted_Close'], label='Predicted', linestyle='--')
        plt.title(f'Tesla Stock Price - Actual vs Predicted for {year}-{month:02d}')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Average prediction error: {result_df['Error_Percentage'].abs().mean():.2f}%")
        return result_df

    # For future prediction, we need to create the features for each day
    print(f"Predicting Tesla stock for {year}-{month:02d} (future period)...")

    # Get the most recent data point to start our predictions
    latest_data = df.iloc[-1:].copy()
    # Make sure we're working with a copy that doesn't have 'Next_Close'
    if 'Next_Close' in latest_data.columns:
        feature_data = latest_data.drop('Next_Close', axis=1)
    else:
        feature_data = latest_data.copy()

    predictions = []

    # For each day in the prediction period:
    for i, date in enumerate(date_range):
        # For the first prediction, use the last known data
        if i == 0:
            predicted_price = model.predict(feature_data)[0]
            predictions.append((date, predicted_price))
            # Update close price for next prediction
            feature_data.loc[:, 'Close'] = predicted_price
            continue

        # Update lagged features
        for lag in range(5, 0, -1):
            if lag > 1:
                feature_data.loc[:, f'Close_Lag_{lag}'] = feature_data[f'Close_Lag_{lag-1}'].values[0]
            else:
                feature_data.loc[:, 'Close_Lag_1'] = feature_data['Close'].values[0]

        # Make prediction for current day
        predicted_price = model.predict(feature_data)[0]
        predictions.append((date, predicted_price))

        # Update Close for next prediction
        feature_data.loc[:, 'Close'] = predicted_price

    # Create prediction DataFrame
    future_df = pd.DataFrame(predictions, columns=['Date', 'Predicted_Close'])
    future_df.set_index('Date', inplace=True)

    # Plot predictions
    plt.figure(figsize=(14, 7))
    plt.plot(future_df.index, future_df['Predicted_Close'], label='Predicted', color='blue')
    plt.title(f'Tesla Stock Price Prediction for {year}-{month:02d}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Predicted prices for {year}-{month:02d}:")
    print(future_df)

    return future_df

# 7. Main function
def tesla_stock_prediction(file_path, prediction_year, prediction_month):
    # Load and prepare data
    print("Loading and preparing Tesla stock data...")
    df = prepare_tesla_data(file_path)

    # Prepare model data
    X, y = prepare_model_data(df)

    # Build model
    print("Building Random Forest model...")
    model = build_tesla_model(X, y)

    # Evaluate model
    print("\nEvaluating model performance on training data...")
    metrics = evaluate_model(model, X, y)

    # Train-test split and evaluation
    print("\nTesting model on holdout data...")
    test_metrics = train_test_predict(model, df, X, y)

    # Top features
    print("\nTop 5 most important features:")
    for i, (feature, importance) in enumerate(zip(metrics[4]['Feature'].head(5),
                                             metrics[4]['Importance'].head(5)), 1):
        print(f"{i}. {feature}: {importance:.4f}")

    # Make predictions for specified month
    print(f"\nGenerating predictions for {prediction_year}-{prediction_month:02d}...")
    predictions = predict_month(model, df, prediction_year, prediction_month)

    print("\n=== TESLA STOCK PREDICTION SUMMARY ===")
    print(f"Predictions for {prediction_year}-{prediction_month:02d} completed.")

    return model, predictions

# Entry point
if __name__ == "__main__":
    # File path
    file_path = 'TSLA.csv'

    # Get prediction period (can be replaced with command-line arguments)
    prediction_year = int(input("Enter year for prediction: "))
    prediction_month = int(input("Enter month for prediction (1-12): "))

    # Run prediction
    model, predictions = tesla_stock_prediction(file_path, prediction_year, prediction_month)
