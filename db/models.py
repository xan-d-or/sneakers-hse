from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base
import datetime

Base = declarative_base()

class Sneaker(Base):
    __tablename__ = "sneakers"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    brand = Column(String)
    image_path = Column(String)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    image_name = Column(String)
    top1 = Column(String)
    top2 = Column(String)
    top3 = Column(String)
    latency = Column(Float)