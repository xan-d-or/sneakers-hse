from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://sneakers:sneakers@localhost:5432/sneakers_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)