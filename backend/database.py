# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

# Create the SQLAlchemy engine.
# connect_args is needed only for SQLite to allow it to be used by multiple threads,
# which is what FastAPI does. This is a critical setting for this stack.
engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)

# Each instance of the SessionLocal class will be a database session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This Base will be used by our ORM models to inherit from.
Base = declarative_base()

# Dependency for FastAPI: This function will be called for each request
# that needs a database connection. It ensures the session is always
# closed after the request is finished, even if there was an error.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()