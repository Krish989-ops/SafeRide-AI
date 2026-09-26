from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base


# ============================================================
# DRIVER TABLE
# ============================================================

class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, index=True)

    driver_id = Column(String, unique=True, index=True, nullable=False)

    name = Column(String, nullable=False)

    phone = Column(String, default="")

    vehicle_number = Column(String, default="")

    total_trips = Column(Integer, default=0)

    average_score = Column(Float, default=0.0)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    # Relationship with trips
    trips = relationship(
        "Trip",
        back_populates="driver",
        cascade="all, delete-orphan"
    )


# ============================================================
# TRIP TABLE
# ============================================================

class Trip(Base):
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)

    trip_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    driver_id = Column(
        String,
        ForeignKey("drivers.driver_id"),
        nullable=False
    )

    status = Column(
        String,
        default="active"
    )

    start_time = Column(
        DateTime,
        default=datetime.utcnow
    )

    end_time = Column(
        DateTime,
        nullable=True
    )

    eye_closure_events = Column(
        Integer,
        default=0
    )

    total_yawns = Column(
        Integer,
        default=0
    )

    distraction_events = Column(
        Integer,
        default=0
    )

    warnings = Column(
        Integer,
        default=0
    )

    max_eye_closure = Column(
        Float,
        default=0.0
    )

    max_distraction_duration = Column(
        Float,
        default=0.0
    )

    safety_score = Column(
        Float,
        default=100.0
    )

    final_score = Column(
        Float,
        nullable=True
    )

    rating = Column(
        Float,
        nullable=True
    )

    performance = Column(
        String,
        nullable=True
    )

    # Relationship with driver
    driver = relationship(
        "Driver",
        back_populates="trips"
    )

    # Relationship with safety events
    safety_events = relationship(
        "SafetyEvent",
        back_populates="trip",
        cascade="all, delete-orphan"
    )

    # Relationship with alerts
    alerts = relationship(
        "Alert",
        back_populates="trip",
        cascade="all, delete-orphan"
    )


# ============================================================
# SAFETY EVENT TABLE
# ============================================================

class SafetyEvent(Base):
    __tablename__ = "safety_events"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    trip_id = Column(
        String,
        ForeignKey("trips.trip_id"),
        nullable=False
    )

    eye_closure_duration = Column(
        Float,
        default=0.0
    )

    yawn_count = Column(
        Integer,
        default=0
    )

    distraction_duration = Column(
        Float,
        default=0.0
    )

    safety_score = Column(
        Float,
        default=100.0
    )

    alertness_score = Column(
        Float,
        default=100.0
    )

    timestamp = Column(
        DateTime,
        default=datetime.utcnow
    )

    trip = relationship(
        "Trip",
        back_populates="safety_events"
    )


# ============================================================
# ALERT TABLE
# ============================================================

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    alert_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    trip_id = Column(
        String,
        ForeignKey("trips.trip_id"),
        nullable=False
    )

    alert_type = Column(
        String,
        nullable=False
    )

    message = Column(
        String,
        nullable=False
    )

    severity = Column(
        String,
        default="warning"
    )

    timestamp = Column(
        DateTime,
        default=datetime.utcnow
    )

    trip = relationship(
        "Trip",
        back_populates="alerts"
    )