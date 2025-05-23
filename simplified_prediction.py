"""
A simplified stock price prediction module that does not rely on TensorFlow
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

class SimpleStockPredictor:
    """
    Class for stock price prediction using Prophet and statistical methods
    """
    
    def __init__(self):
        self.prophet_model = None
        self.linear_model = None
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
        self.prophet_model = Prophet(changepoint_prior_scale=0.05)
        self.prophet_model.fit(prophet_df)
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=period)
        
        # Make predictions
        forecast = self.prophet_model.predict(future)
        
        return forecast
    
    def prepare_data_for_linear(self, df, look_back=60):
        """Prepare data for Linear Regression model"""
        # Get close prices and scale them
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        # Create datasets with lookback period
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
            
        # Convert to numpy arrays
        X, y = np.array(X), np.array(y)
        
        return X, y, scaled_data
    
    def fit_linear_model(self, df, look_back=60):
        """Fit linear regression model"""
        # Prepare data
        X, y, scaled_data = self.prepare_data_for_linear(df, look_back)
        
        # Reshape X for Linear Regression (samples, features)
        X_reshaped = X.reshape(X.shape[0], X.shape[1])
        
        # Split into train and test sets (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X_reshaped[:train_size], X_reshaped[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train linear regression model
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)
        
        # Make predictions
        train_predict = self.linear_model.predict(X_train).reshape(-1, 1)
        test_predict = self.linear_model.predict(X_test).reshape(-1, 1)
        
        # Inverse transforms
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict.reshape(-1, 1))
        actual_y_train = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        actual_y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Return results
        return {
            'model': self.linear_model,
            'train_predict': train_predict,
            'test_predict': test_predict,
            'data': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        }
    
    def predict_future_with_linear(self, df, days_to_predict=30, look_back=60):
        """Make future predictions with linear regression model"""
        # Prepare the last "look_back" days for prediction
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        # Get the last look_back days
        last_data = scaled_data[-look_back:].reshape(1, -1)
        
        # Make predictions day by day
        predictions = []
        current_batch = last_data.copy()
        
        for i in range(days_to_predict):
            # Make prediction for the next day
            current_pred = self.linear_model.predict(current_batch).reshape(-1, 1)[0, 0]
            predictions.append(current_pred)
            
            # Update the batch to include the new prediction and remove the oldest day
            current_batch = np.append(current_batch[:, 1:], [[current_pred]], axis=1)
        
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
    
    def create_prediction_chart(self, historical_data, prophet_forecast, linear_predictions, ticker_symbol):
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
        
        # Add Linear Regression predictions
        fig.add_trace(
            go.Scatter(
                x=linear_predictions.index,
                y=linear_predictions['Prediction'],
                name='Linear Regression Prediction',
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
    
    def predict_stock(self, ticker_symbol, period="1y", prophet_days=30, linear_days=30):
        """
        Main method to predict stock price
        
        Parameters:
            ticker_symbol (str): Stock symbol
            period (str): Historical data period to use
            prophet_days (int): Days to predict with Prophet
            linear_days (int): Days to predict with linear regression
            
        Returns:
            dict: Dictionary with prediction results
        """
        # Fetch historical data
        try:
            data = yf.download(ticker_symbol, period=period)
            
            # Make sure we have enough data
            if len(data) < 100:
                return {
                    'success': False,
                    'error': 'Not enough historical data for prediction'
                }
            
            # Prophet prediction
            prophet_forecast = self.fit_prophet_model(data, period=prophet_days)
            
            # Linear Regression prediction
            linear_results = self.fit_linear_model(data)
            linear_predictions = self.predict_future_with_linear(data, days_to_predict=linear_days)
            
            # Create prediction chart
            prediction_chart = self.create_prediction_chart(
                data, prophet_forecast, linear_predictions, ticker_symbol
            )
            
            # Create a prediction summary (ensuring scalar values)
            current_price = float(data['Close'].iloc[-1])
            prophet_last = float(prophet_forecast['yhat'].iloc[-1])
            linear_last = float(linear_predictions['Prediction'].iloc[-1])
            
            # Calculate predicted change
            prophet_change = (prophet_last - current_price) / current_price * 100
            linear_change = (linear_last - current_price) / current_price * 100
            
            # Average prediction
            avg_prediction = (prophet_last + linear_last) / 2
            avg_change = (avg_prediction - current_price) / current_price * 100
            
            # Determine trend (making sure to use scalar values not pandas Series)
            avg_change_value = float(avg_change)
            if avg_change_value > 2:
                trend = "up"
            elif avg_change_value < -2:
                trend = "down"
            else:
                trend = "sideways"
                
            # Confidence score based on agreement between models (ensuring scalar values)
            prophet_change_value = float(prophet_change)
            linear_change_value = float(linear_change)
            confidence = 100 - min(abs(prophet_change_value - linear_change_value), 50)
            
            prediction_summary = {
                'current_price': current_price,
                'prophet_prediction': {
                    'price': prophet_last,
                    'change_percent': prophet_change
                },
                'linear_prediction': {
                    'price': linear_last,
                    'change_percent': linear_change
                },
                'average_prediction': {
                    'price': avg_prediction,
                    'change_percent': avg_change
                },
                'predicted_trend': trend,
                'confidence_score': confidence
            }
            
            return {
                'success': True,
                'historical_data': data,
                'prophet_forecast': prophet_forecast,
                'linear_predictions': linear_predictions,
                'prediction_summary': prediction_summary,
                'prediction_chart': prediction_chart
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }