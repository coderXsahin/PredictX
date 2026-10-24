
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    machine_type: Mapped[str] = mapped_column(String(1), nullable=False)
    air_temperature: Mapped[float] = mapped_column(Float, nullable=False)
    process_temperature: Mapped[float] = mapped_column(Float, nullable=False)
    rotational_speed: Mapped[int] = mapped_column(Integer, nullable=False)
    torque: Mapped[float] = mapped_column(Float, nullable=False)
    tool_wear: Mapped[int] = mapped_column(Integer, nullable=False)

    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
