# database.py
"""
Handles database connection and session management using SQLAlchemy.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

# The engine is the entry point to the database.
engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)

# Each instance of the SessionLocal class will be a database session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This Base class will be inherited by the ORM models.
Base = declarative_base()

def get_db():
    """
    FastAPI dependency to create and yield a new database session for each request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()