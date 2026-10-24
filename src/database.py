
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DB_PASSWORD = os.getenv("DB_PASSWORD")

if not DB_PASSWORD:
    raise RuntimeError(
        "DB_PASSWORD is not set. Set it in your terminal before starting PredictX."
    )

DATABASE_URL = (
    f"postgresql+psycopg://postgres:{DB_PASSWORD}"
    "@localhost:5432/predictx"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass
