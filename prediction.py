import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go

class StockPredictor:
    """
    Class for stock price prediction using Prophet and LSTM models
    """
    
    def __init__(self):
        self.prophet_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data_for_prophet(self, df):
        """Prepare data for Prophet model"""
        prophet_df = df.reset_index()[['Date', 'Close']]
        prophet_df.columns = ['ds', 'y']
        return prophet_df
    
    def fit_prophet_model(self, df, period=30):
        """Fit Prophet model and make prediction"""
        prophet_df = self.prepare_data_for_prophet(df)
        
        # Train the Prophet model
        self.prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        self.prophet_model.fit(prophet_df)
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=period)
        
        # Make predictions
        forecast = self.prophet_model.predict(future)
        
        return forecast
    
    def prepare_data_for_lstm(self, df, look_back=60):
        """Prepare data for LSTM model"""
        # Scale the data
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        # Create datasets with lookback period
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
            
        # Convert to numpy arrays
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaled_data
    
    def fit_lstm_model(self, df, look_back=60, epochs=50, batch_size=32):
        """Fit LSTM model and make prediction"""
        # Prepare data
        X, y, scaled_data = self.prepare_data_for_lstm(df, look_back)
        
        # Split into train and test sets (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build LSTM model
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        self.lstm_model.add(Dropout(0.2))
        self.lstm_model.add(LSTM(units=50, return_sequences=False))
        self.lstm_model.add(Dropout(0.2))
        self.lstm_model.add(Dense(units=1))
        
        # Compile the model
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Make predictions
        train_predict = self.lstm_model.predict(X_train)
        test_predict = self.lstm_model.predict(X_test)
        
        # Inverse transforms
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)
        actual_y_train = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        actual_y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate errors
        train_mae = mean_absolute_error(actual_y_train, train_predict)
        test_mae = mean_absolute_error(actual_y_test, test_predict)
        train_rmse = np.sqrt(mean_squared_error(actual_y_train, train_predict))
        test_rmse = np.sqrt(mean_squared_error(actual_y_test, test_predict))
        
        # Prepare results
        results = {
            'model': self.lstm_model,
            'history': history,
            'train_predict': train_predict,
            'test_predict': test_predict,
            'metrics': {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            },
            'scaler': self.scaler,
            'data': {
                'X_train': X_train,
                'X_test': X_test, 
                'y_train': y_train,
                'y_test': y_test
            }
        }
        
        return results
    
    def predict_future_with_lstm(self, df, days_to_predict=30, look_back=60):
        """Make future predictions with LSTM model"""
        # Prepare the last "look_back" days for prediction
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        # Use the last "look_back" days as input
        X_future = scaled_data[-look_back:].reshape(1, look_back, 1)
        
        # Make predictions day by day
        predictions = []
        current_batch = X_future[0]
        
        for i in range(days_to_predict):
            # Make prediction for the next day
            current_pred = self.lstm_model.predict(current_batch.reshape(1, look_back, 1))
            predictions.append(current_pred[0, 0])
            
            # Update the batch to include the new prediction
            current_batch = np.append(current_batch[1:], [[current_pred[0, 0]]], axis=0)
        
        # Convert predictions to price values
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Create date range for future predictions
        last_date = df.index[-1]
        prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict)
        
        # Create dataframe of predictions
        future_predictions = pd.DataFrame(
            data=predictions,
            index=prediction_dates,
            columns=['Prediction']
        )
        
        return future_predictions
    
    def create_prediction_chart(self, historical_data, prophet_forecast, lstm_predictions, ticker_symbol):
        """Create chart combining historical data and predictions"""
        # Create a new figure
        fig = go.Figure()
        
        # Currency symbol based on ticker (assuming .TW for Taiwan stocks)
        currency_symbol = "NT$" if ticker_symbol.upper().endswith('.TW') else "$"
        
        # Add historical prices
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                name='Historical Price',
                line=dict(color='blue')
            )
        )
        
        # Add Prophet predictions
        prophet_dates = prophet_forecast['ds'].iloc[-len(prophet_forecast) + len(historical_data):]
        prophet_values = prophet_forecast['yhat'].iloc[-len(prophet_forecast) + len(historical_data):]
        
        fig.add_trace(
            go.Scatter(
                x=prophet_dates,
                y=prophet_values,
                name='Prophet Prediction',
                line=dict(color='green')
            )
        )
        
        # Add Prophet confidence intervals
        fig.add_trace(
            go.Scatter(
                x=prophet_dates.tolist() + prophet_dates.tolist()[::-1],
                y=prophet_forecast['yhat_upper'].iloc[-len(prophet_forecast) + len(historical_data):].tolist() + 
                  prophet_forecast['yhat_lower'].iloc[-len(prophet_forecast) + len(historical_data):].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(0,100,80,0.2)'),
                name='Prophet Confidence Interval'
            )
        )
        
        # Add LSTM predictions
        fig.add_trace(
            go.Scatter(
                x=lstm_predictions.index,
                y=lstm_predictions['Prediction'],
                name='LSTM Prediction',
                line=dict(color='red')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"{ticker_symbol} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title=f"Stock Price ({currency_symbol})",
            legend_title="Data Source",
            height=600,
            hovermode="x unified"
        )
        
        return fig
    
    def predict_stock(self, ticker_symbol, period="1y", prophet_days=30, lstm_days=30):
        """
        Main method to predict stock price
        
        Parameters:
            ticker_symbol (str): Stock symbol
            period (str): Historical data period to use
            prophet_days (int): Days to predict with Prophet
            lstm_days (int): Days to predict with LSTM
            
        Returns:
            dict: Dictionary with prediction results
        """
        # Fetch historical data
        data = yf.download(ticker_symbol, period=period)
        
        # Make sure we have enough data
        if len(data) < 100:
            return {
                'success': False,
                'error': 'Not enough historical data for prediction'
            }
        
        try:
            # Prophet prediction
            prophet_forecast = self.fit_prophet_model(data, period=prophet_days)
            
            # LSTM prediction
            lstm_results = self.fit_lstm_model(data)
            lstm_predictions = self.predict_future_with_lstm(data, days_to_predict=lstm_days)
            
            # Create prediction chart
            prediction_chart = self.create_prediction_chart(
                data, prophet_forecast, lstm_predictions, ticker_symbol
            )
            
            # Create a prediction summary
            current_price = data['Close'].iloc[-1]
            prophet_last = prophet_forecast['yhat'].iloc[-1]
            lstm_last = lstm_predictions['Prediction'].iloc[-1]
            
            # Calculate predicted change
            prophet_change = (prophet_last - current_price) / current_price * 100
            lstm_change = (lstm_last - current_price) / current_price * 100
            
            # Average prediction
            avg_prediction = (prophet_last + lstm_last) / 2
            avg_change = (avg_prediction - current_price) / current_price * 100
            
            prediction_summary = {
                'current_price': current_price,
                'prophet_prediction': {
                    'price': prophet_last,
                    'change_percent': prophet_change
                },
                'lstm_prediction': {
                    'price': lstm_last,
                    'change_percent': lstm_change
                },
                'average_prediction': {
                    'price': avg_prediction,
                    'change_percent': avg_change
                },
                'predicted_trend': 'Up' if avg_change > 0 else 'Down',
                'confidence_score': min(100, max(0, 100 - abs(prophet_change - lstm_change)))
            }
            
            return {
                'success': True,
                'historical_data': data,
                'prophet_forecast': prophet_forecast,
                'lstm_predictions': lstm_predictions,
                'prediction_summary': prediction_summary,
                'prediction_chart': prediction_chart
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }