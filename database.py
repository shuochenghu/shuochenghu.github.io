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