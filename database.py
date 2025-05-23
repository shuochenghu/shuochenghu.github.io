import os
import pandas as pd
import datetime
import json

# Flag to determine if database features are available
database_available = False

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Table, ForeignKey, MetaData, select, insert, update, delete, desc, func
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship
    
    # Get the database URL from environment variables
    DATABASE_URL = os.getenv("DATABASE_URL")
    if DATABASE_URL is not None:
        database_available = True
except ImportError:
    pass  # SQLAlchemy not available

# Only create SQLAlchemy engine if database is available
if database_available:
    try:
        engine = create_engine(DATABASE_URL)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        database_available = False

# Create a base class for declarative models
Base = declarative_base()

# Define the StockData model for caching stock price data
class StockData(Base):
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True, nullable=False)
    date = Column(DateTime, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<StockData(symbol='{self.symbol}', date='{self.date}')>"

# Define the UserPreference model for saving user settings
class UserPreference(Base):
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False) # This could be a session ID or user ID
    preference_name = Column(String, nullable=False)
    preference_value = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<UserPreference(user_id='{self.user_id}', preference_name='{self.preference_name}')>"

# Define the FavoriteStock model for tracking favorite stocks
class FavoriteStock(Base):
    __tablename__ = 'favorite_stocks'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    symbol = Column(String, nullable=False)
    added_at = Column(DateTime, default=datetime.datetime.utcnow)
    notes = Column(String)
    
    def __repr__(self):
        return f"<FavoriteStock(user_id='{self.user_id}', symbol='{self.symbol}')>"

# Define the Portfolio model for tracking user's stock portfolio
class Portfolio(Base):
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False, default="Default Portfolio")
    description = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<Portfolio(user_id='{self.user_id}', name='{self.name}')>"

# Define the PortfolioItem model for individual holdings in a portfolio
class PortfolioItem(Base):
    __tablename__ = 'portfolio_items'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolio.id', ondelete='CASCADE'), nullable=False)
    symbol = Column(String, nullable=False)
    shares = Column(Float, default=0)
    purchase_price = Column(Float)
    purchase_date = Column(DateTime)
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<PortfolioItem(portfolio_id='{self.portfolio_id}', symbol='{self.symbol}', shares={self.shares})>"

# Create all tables in the database
Base.metadata.create_all(engine)

# Create a session factory
Session = sessionmaker(bind=engine)

def get_session():
    """Return a new database session"""
    return Session()

def save_stock_data(symbol, dataframe):
    """Save stock price data to the database"""
    session = get_session()
    
    try:
        # Convert DataFrame to list of dictionaries
        records = []
        for index, row in dataframe.iterrows():
            record = {
                'symbol': symbol,
                'date': index,
                'open': row.get('Open'),
                'high': row.get('High'),
                'low': row.get('Low'),
                'close': row.get('Close'),
                'volume': row.get('Volume')
            }
            records.append(record)
        
        # Check for existing records and update them or insert new ones
        for record in records:
            # Check if record exists
            existing = session.query(StockData).filter_by(
                symbol=record['symbol'], 
                date=record['date']
            ).first()
            
            if existing:
                # Update existing record
                for key, value in record.items():
                    setattr(existing, key, value)
            else:
                # Insert new record
                session.add(StockData(**record))
        
        session.commit()
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error saving stock data: {e}")
        return False
    
    finally:
        session.close()

def get_cached_stock_data(symbol, start_date, end_date=None):
    """Retrieve stock data from the database cache"""
    session = get_session()
    
    try:
        query = session.query(StockData).filter(
            StockData.symbol == symbol,
            StockData.date >= start_date
        )
        
        if end_date:
            query = query.filter(StockData.date <= end_date)
        
        query = query.order_by(StockData.date)
        results = query.all()
        
        if not results:
            return None
        
        # Convert to DataFrame
        data = {
            'Open': [r.open for r in results],
            'High': [r.high for r in results],
            'Low': [r.low for r in results],
            'Close': [r.close for r in results],
            'Volume': [r.volume for r in results]
        }
        
        df = pd.DataFrame(data, index=[r.date for r in results])
        return df
    
    except Exception as e:
        print(f"Error retrieving cached stock data: {e}")
        return None
    
    finally:
        session.close()

def save_user_preference(user_id, preference_name, preference_value):
    """Save or update a user preference"""
    session = get_session()
    
    try:
        # Check if preference exists
        pref = session.query(UserPreference).filter_by(
            user_id=user_id,
            preference_name=preference_name
        ).first()
        
        if pref:
            # Update existing preference
            pref.preference_value = preference_value
            pref.updated_at = datetime.datetime.utcnow()
        else:
            # Create new preference
            pref = UserPreference(
                user_id=user_id,
                preference_name=preference_name,
                preference_value=preference_value
            )
            session.add(pref)
        
        session.commit()
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error saving user preference: {e}")
        return False
    
    finally:
        session.close()

def get_user_preference(user_id, preference_name, default_value=None):
    """Get a user preference value"""
    session = get_session()
    
    try:
        pref = session.query(UserPreference).filter_by(
            user_id=user_id,
            preference_name=preference_name
        ).first()
        
        if pref:
            return pref.preference_value
        else:
            return default_value
    
    except Exception as e:
        print(f"Error getting user preference: {e}")
        return default_value
    
    finally:
        session.close()

def save_favorite_stock(user_id, symbol, notes=None):
    """Save a stock as favorite"""
    session = get_session()
    
    try:
        # Check if already a favorite
        fav = session.query(FavoriteStock).filter_by(
            user_id=user_id,
            symbol=symbol
        ).first()
        
        if not fav:
            fav = FavoriteStock(
                user_id=user_id,
                symbol=symbol,
                notes=notes
            )
            session.add(fav)
            session.commit()
        
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error saving favorite stock: {e}")
        return False
    
    finally:
        session.close()

def remove_favorite_stock(user_id, symbol):
    """Remove a stock from favorites"""
    session = get_session()
    
    try:
        fav = session.query(FavoriteStock).filter_by(
            user_id=user_id,
            symbol=symbol
        ).first()
        
        if fav:
            session.delete(fav)
            session.commit()
        
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error removing favorite stock: {e}")
        return False
    
    finally:
        session.close()

def get_favorite_stocks(user_id):
    """Get all favorite stocks for a user"""
    session = get_session()
    
    try:
        favs = session.query(FavoriteStock).filter_by(
            user_id=user_id
        ).all()
        
        return [(fav.symbol, fav.notes) for fav in favs]
    
    except Exception as e:
        print(f"Error getting favorite stocks: {e}")
        return []
    
    finally:
        session.close()
        
def create_portfolio(user_id, name, description=None):
    """Create a new portfolio for a user"""
    session = get_session()
    
    try:
        portfolio = Portfolio(
            user_id=user_id,
            name=name,
            description=description
        )
        session.add(portfolio)
        session.commit()
        return portfolio.id
    
    except Exception as e:
        session.rollback()
        print(f"Error creating portfolio: {e}")
        return None
    
    finally:
        session.close()

def get_portfolios(user_id):
    """Get all portfolios for a user"""
    session = get_session()
    
    try:
        portfolios = session.query(Portfolio).filter_by(
            user_id=user_id
        ).all()
        
        return [(p.id, p.name, p.description) for p in portfolios]
    
    except Exception as e:
        print(f"Error getting portfolios: {e}")
        return []
    
    finally:
        session.close()
        
def delete_portfolio(portfolio_id):
    """Delete a portfolio and all its items"""
    session = get_session()
    
    try:
        portfolio = session.query(Portfolio).filter_by(
            id=portfolio_id
        ).first()
        
        if portfolio:
            session.delete(portfolio)
            session.commit()
            return True
        return False
    
    except Exception as e:
        session.rollback()
        print(f"Error deleting portfolio: {e}")
        return False
    
    finally:
        session.close()
        
def add_portfolio_item(portfolio_id, symbol, shares, purchase_price=None, purchase_date=None, notes=None):
    """Add a stock item to a portfolio"""
    session = get_session()
    
    try:
        # Check if item already exists
        item = session.query(PortfolioItem).filter_by(
            portfolio_id=portfolio_id,
            symbol=symbol
        ).first()
        
        if item:
            # Update existing item
            item.shares = shares
            if purchase_price is not None:
                item.purchase_price = purchase_price
            if purchase_date is not None:
                item.purchase_date = purchase_date
            if notes is not None:
                item.notes = notes
        else:
            # Create new item
            item = PortfolioItem(
                portfolio_id=portfolio_id,
                symbol=symbol,
                shares=shares,
                purchase_price=purchase_price,
                purchase_date=purchase_date,
                notes=notes
            )
            session.add(item)
            
        session.commit()
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error adding portfolio item: {e}")
        return False
    
    finally:
        session.close()
        
def remove_portfolio_item(portfolio_id, symbol):
    """Remove a stock item from a portfolio"""
    session = get_session()
    
    try:
        item = session.query(PortfolioItem).filter_by(
            portfolio_id=portfolio_id,
            symbol=symbol
        ).first()
        
        if item:
            session.delete(item)
            session.commit()
            return True
        return False
    
    except Exception as e:
        session.rollback()
        print(f"Error removing portfolio item: {e}")
        return False
    
    finally:
        session.close()
        
def get_portfolio_items(portfolio_id):
    """Get all items in a portfolio"""
    session = get_session()
    
    try:
        items = session.query(PortfolioItem).filter_by(
            portfolio_id=portfolio_id
        ).all()
        
        return [(item.symbol, item.shares, item.purchase_price, item.purchase_date, item.notes) 
                for item in items]
    
    except Exception as e:
        print(f"Error getting portfolio items: {e}")
        return []
    
    finally:
        session.close()
        
def get_portfolio_by_id(portfolio_id):
    """Get portfolio by ID"""
    session = get_session()
    
    try:
        portfolio = session.query(Portfolio).filter_by(
            id=portfolio_id
        ).first()
        
        if portfolio:
            return {
                'id': portfolio.id,
                'user_id': portfolio.user_id,
                'name': portfolio.name,
                'description': portfolio.description,
                'created_at': portfolio.created_at
            }
        return None
    
    except Exception as e:
        print(f"Error getting portfolio: {e}")
        return None
    
    finally:
        session.close()