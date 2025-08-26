# Overview

This is a comprehensive stock analysis platform built with Streamlit that provides users with real-time stock data visualization, technical analysis, portfolio management, and AI-powered insights. The application supports multiple languages and offers features like favorite stocks tracking, portfolio risk analysis, and stock price predictions using machine learning models.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application framework for rapid prototyping and deployment
- **Visualization**: Plotly for interactive charts and financial data visualization
- **UI Components**: Streamlit's built-in components for forms, sidebars, and data display
- **Multi-language Support**: Custom translation system with support for English and Traditional Chinese

## Backend Architecture
- **Data Layer**: Hybrid approach using both in-memory storage and optional database integration
- **Session Management**: UUID-based session tracking for user identification
- **Caching Strategy**: Local caching system for stock data to reduce API calls
- **Modular Design**: Separated concerns with dedicated modules for different functionalities

## Data Processing Pipeline
- **Stock Data Source**: Yahoo Finance API via yfinance library for real-time market data
- **Technical Analysis**: Custom implementation of technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- **Prediction Models**: 
  - Prophet for time series forecasting
  - Linear regression as fallback model
  - Simplified approach avoiding TensorFlow dependencies
- **Risk Assessment**: Portfolio risk analysis with volatility calculations and diversification scoring

## Database Design
- **Flexible Storage**: Dual-mode operation supporting both in-memory storage and SQL database
- **Data Models**: 
  - User preferences and settings
  - Favorite stocks tracking
  - Portfolio management with holdings
  - Stock data caching
- **Fallback Strategy**: Graceful degradation to in-memory storage when database is unavailable

## Security and Performance
- **API Key Management**: Environment variable-based configuration for external services
- **Error Handling**: Comprehensive error handling with graceful fallbacks
- **Performance Optimization**: Data caching and efficient DataFrame operations

# External Dependencies

## Financial Data APIs
- **Yahoo Finance**: Primary data source for stock prices, market data, and company information via yfinance library
- **OpenAI API**: AI-powered portfolio analysis and investment recommendations (requires API key)

## Machine Learning Libraries
- **Prophet**: Facebook's time series forecasting library for stock price predictions
- **scikit-learn**: Machine learning utilities for data preprocessing and linear regression models
- **pandas & numpy**: Core data manipulation and numerical computing libraries

## Visualization and UI
- **Plotly**: Interactive charting library for financial data visualization
- **Streamlit**: Web application framework for the user interface

## Database Support
- **SQLAlchemy**: Optional ORM for database operations with PostgreSQL support
- **In-memory Fallback**: Built-in dictionary-based storage when database is unavailable

## Development Tools
- **UUID**: Session management and user identification
- **datetime**: Time series data handling and date calculations
- **json**: Data serialization for preferences and configuration storage