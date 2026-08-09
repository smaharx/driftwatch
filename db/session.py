# db/session.py

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # auto-reconnect if connection drops
    pool_size=5,  # max 5 persistent connections
    max_overflow=10,  # allow 10 extra connections under load
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session, always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
