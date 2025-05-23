"""
A simplified version of the database module for user preferences.
"""

import os
import json
import datetime
import pandas as pd

# Dictionary to store data in memory
user_preferences = {}
favorite_stocks = {}
stock_data_cache = {}
portfolios = {}
portfolio_items = {}

def get_user_preference(user_id, preference_name, default_value=None):
    """Get a user preference value, with simple local storage"""
    if user_id not in user_preferences:
        return default_value
    
    return user_preferences.get(user_id, {}).get(preference_name, default_value)

def save_user_preference(user_id, preference_name, preference_value):
    """Save a user preference value, with simple local storage"""
    if user_id not in user_preferences:
        user_preferences[user_id] = {}
    
    user_preferences[user_id][preference_name] = preference_value
    return True

def get_favorite_stocks(user_id):
    """Get favorite stocks for a user, with simple local storage"""
    return favorite_stocks.get(user_id, [])

def save_favorite_stock(user_id, symbol, notes=None):
    """Save a favorite stock, with simple local storage"""
    if user_id not in favorite_stocks:
        favorite_stocks[user_id] = []
    
    # Check if already exists
    for idx, (existing_symbol, _) in enumerate(favorite_stocks[user_id]):
        if existing_symbol == symbol:
            # Update notes
            favorite_stocks[user_id][idx] = (symbol, notes)
            return True
    
    # Add new favorite
    favorite_stocks[user_id].append((symbol, notes))
    return True

def remove_favorite_stock(user_id, symbol):
    """Remove a favorite stock, with simple local storage"""
    if user_id not in favorite_stocks:
        return True
    
    favorite_stocks[user_id] = [
        (s, n) for s, n in favorite_stocks[user_id] if s != symbol
    ]
    return True

def save_stock_data(symbol, dataframe):
    """Save stock data, with simple local storage"""
    # We'll just keep the latest data in memory
    stock_data_cache[symbol] = dataframe.copy()
    return True

def get_cached_stock_data(symbol, start_date, end_date=None):
    """Get cached stock data, with simple local storage"""
    if symbol not in stock_data_cache:
        return None
    
    df = stock_data_cache[symbol]
    
    # Filter by date
    df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    return df

def create_portfolio(user_id, name, description=None):
    """Create a new portfolio for a user, with simple local storage"""
    if user_id not in portfolios:
        portfolios[user_id] = []
    
    # Generate an ID
    portfolio_id = f"{user_id}_{len(portfolios[user_id]) + 1}_{datetime.datetime.now().timestamp()}"
    
    # Create portfolio
    portfolio = {
        'id': portfolio_id,
        'name': name,
        'description': description,
        'created_at': datetime.datetime.now()
    }
    
    portfolios[user_id].append(portfolio)
    return portfolio_id

def get_portfolios(user_id):
    """Get all portfolios for a user, with simple local storage"""
    if user_id not in portfolios:
        return []
    
    return [(p['id'], p['name'], p['description']) for p in portfolios[user_id]]

def delete_portfolio(portfolio_id):
    """Delete a portfolio and all its items, with simple local storage"""
    # Find portfolio and remove it
    for user_id, user_portfolios in portfolios.items():
        for idx, portfolio in enumerate(user_portfolios):
            if portfolio['id'] == portfolio_id:
                portfolios[user_id].pop(idx)
                
                # Also remove all portfolio items
                if portfolio_id in portfolio_items:
                    del portfolio_items[portfolio_id]
                
                return True
    
    return False

def add_portfolio_item(portfolio_id, symbol, shares, purchase_price=None, purchase_date=None, notes=None):
    """Add a stock item to a portfolio, with simple local storage"""
    if portfolio_id not in portfolio_items:
        portfolio_items[portfolio_id] = []
    
    # Check if item already exists
    for idx, item in enumerate(portfolio_items[portfolio_id]):
        if item['symbol'] == symbol:
            # Update existing item
            portfolio_items[portfolio_id][idx]['shares'] = shares
            if purchase_price is not None:
                portfolio_items[portfolio_id][idx]['purchase_price'] = purchase_price
            if purchase_date is not None:
                portfolio_items[portfolio_id][idx]['purchase_date'] = purchase_date
            if notes is not None:
                portfolio_items[portfolio_id][idx]['notes'] = notes
            return True
    
    # Create new item
    new_item = {
        'symbol': symbol,
        'shares': shares,
        'purchase_price': purchase_price,
        'purchase_date': purchase_date if purchase_date else datetime.datetime.now(),
        'notes': notes,
        'added_at': datetime.datetime.now()
    }
    
    portfolio_items[portfolio_id].append(new_item)
    return True

def remove_portfolio_item(portfolio_id, symbol):
    """Remove a stock item from a portfolio, with simple local storage"""
    if portfolio_id not in portfolio_items:
        return False
    
    # Remove item
    portfolio_items[portfolio_id] = [
        item for item in portfolio_items[portfolio_id] if item['symbol'] != symbol
    ]
    return True

def get_portfolio_items(portfolio_id):
    """Get all items in a portfolio, with simple local storage"""
    if portfolio_id not in portfolio_items:
        return []
    
    return [(item['symbol'], item['shares'], item['purchase_price'], 
             item['purchase_date'], item['notes']) for item in portfolio_items[portfolio_id]]

def get_portfolio_by_id(portfolio_id):
    """Get portfolio by ID, with simple local storage"""
    # Find portfolio
    for user_id, user_portfolios in portfolios.items():
        for portfolio in user_portfolios:
            if portfolio['id'] == portfolio_id:
                return portfolio
    
    return None