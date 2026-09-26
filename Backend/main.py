from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4
from sqlalchemy.orm import Session

from database import engine, Base, get_db
from models import Driver, Trip, SafetyEvent, Alert


# ============================================================
# SAFE RIDE AI - FASTAPI BACKEND
# DATABASE CONNECTED VERSION
# ============================================================

app = FastAPI(
    title="SafeRide AI API",
    description="Backend API for the SafeRide AI driver safety platform.",
    version="2.0.0"
)
# ============================================================
# CORS - ALLOW FRONTEND TO CONNECT TO BACKEND
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create database tables if they don't exist
Base.metadata.create_all(bind=engine)


# ============================================================
# DATA MODELS
# ============================================================

class DriverCreate(BaseModel):
    driver_id: str
    name: str
    phone: str = ""
    vehicle_number: str = ""


class TripStart(BaseModel):
    driver_id: str


class SafetyData(BaseModel):
    eye_closure_duration: float = Field(default=0.0, ge=0)
    yawn_count: int = Field(default=0, ge=0)
    distraction_duration: float = Field(default=0.0, ge=0)
    safety_score: float = Field(default=100.0, ge=0, le=100)
    alertness_score: float = Field(default=100.0, ge=0, le=100)


class AlertCreate(BaseModel):
    trip_id: str
    alert_type: str
    message: str
    severity: str = "warning"


# ============================================================
# HOME / HEALTH
# ============================================================

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "SafeRide AI Backend is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected"
    }


# ============================================================
# DRIVER APIs
# ============================================================

@app.post("/drivers")
def create_driver(
    driver: DriverCreate,
    db: Session = Depends(get_db)
):

    existing_driver = (
        db.query(Driver)
        .filter(Driver.driver_id == driver.driver_id)
        .first()
    )

    if existing_driver:
        raise HTTPException(
            status_code=400,
            detail="Driver already exists."
        )

    new_driver = Driver(
        driver_id=driver.driver_id,
        name=driver.name,
        phone=driver.phone,
        vehicle_number=driver.vehicle_number,
        total_trips=0,
        average_score=0.0
    )

    db.add(new_driver)
    db.commit()
    db.refresh(new_driver)

    return {
        "message": "Driver created successfully",
        "driver": {
            "driver_id": new_driver.driver_id,
            "name": new_driver.name,
            "phone": new_driver.phone,
            "vehicle_number": new_driver.vehicle_number,
            "total_trips": new_driver.total_trips,
            "average_score": new_driver.average_score,
            "created_at": new_driver.created_at.isoformat()
            if new_driver.created_at else None
        }
    }


@app.get("/drivers/{driver_id}")
def get_driver(
    driver_id: str,
    db: Session = Depends(get_db)
):

    driver = (
        db.query(Driver)
        .filter(Driver.driver_id == driver_id)
        .first()
    )

    if not driver:
        raise HTTPException(
            status_code=404,
            detail="Driver not found."
        )

    return {
        "driver_id": driver.driver_id,
        "name": driver.name,
        "phone": driver.phone,
        "vehicle_number": driver.vehicle_number,
        "total_trips": driver.total_trips,
        "average_score": driver.average_score,
        "created_at": driver.created_at.isoformat()
        if driver.created_at else None
    }


@app.get("/drivers/{driver_id}/history")
def get_driver_history(
    driver_id: str,
    db: Session = Depends(get_db)
):

    driver = (
        db.query(Driver)
        .filter(Driver.driver_id == driver_id)
        .first()
    )

    if not driver:
        raise HTTPException(
            status_code=404,
            detail="Driver not found."
        )

    driver_trips = (
        db.query(Trip)
        .filter(Trip.driver_id == driver_id)
        .order_by(Trip.start_time.desc())
        .all()
    )

    return {
        "driver_id": driver_id,
        "total_trips": len(driver_trips),
        "trips": [
            {
                "trip_id": trip.trip_id,
                "status": trip.status,
                "start_time": trip.start_time.isoformat()
                if trip.start_time else None,
                "end_time": trip.end_time.isoformat()
                if trip.end_time else None,
                "final_score": trip.final_score,
                "rating": trip.rating,
                "performance": trip.performance
            }
            for trip in driver_trips
        ]
    }


# ============================================================
# TRIP APIs
# ============================================================

@app.post("/trips/start")
def start_trip(
    trip_request: TripStart,
    db: Session = Depends(get_db)
):

    driver = (
        db.query(Driver)
        .filter(Driver.driver_id == trip_request.driver_id)
        .first()
    )

    if not driver:
        raise HTTPException(
            status_code=404,
            detail="Driver not found."
        )

    # Prevent multiple active trips for the same driver
    active_trip = (
        db.query(Trip)
        .filter(
            Trip.driver_id == trip_request.driver_id,
            Trip.status == "active"
        )
        .first()
    )

    if active_trip:
        raise HTTPException(
            status_code=400,
            detail="Driver already has an active trip."
        )

    trip_id = "TRIP-" + uuid4().hex[:8].upper()

    new_trip = Trip(
        trip_id=trip_id,
        driver_id=trip_request.driver_id,
        status="active",
        start_time=datetime.now(),
        eye_closure_events=0,
        total_yawns=0,
        distraction_events=0,
        warnings=0,
        max_eye_closure=0.0,
        max_distraction_duration=0.0,
        safety_score=100.0
    )

    db.add(new_trip)

    driver.total_trips += 1

    db.commit()
    db.refresh(new_trip)

    return {
        "message": "Trip started successfully",
        "trip": {
            "trip_id": new_trip.trip_id,
            "driver_id": new_trip.driver_id,
            "status": new_trip.status,
            "start_time": new_trip.start_time.isoformat()
        }
    }


@app.post("/trips/{trip_id}/safety-data")
def update_safety_data(
    trip_id: str,
    data: SafetyData,
    db: Session = Depends(get_db)
):

    trip = (
        db.query(Trip)
        .filter(Trip.trip_id == trip_id)
        .first()
    )

    if not trip:
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    if trip.status != "active":
        raise HTTPException(
            status_code=400,
            detail="Trip is not active."
        )

    # Update maximum eye closure
    trip.max_eye_closure = max(
        trip.max_eye_closure or 0,
        data.eye_closure_duration
    )

    # Update maximum distraction
    trip.max_distraction_duration = max(
        trip.max_distraction_duration or 0,
        data.distraction_duration
    )

    # Keep the highest yawn count observed
    trip.total_yawns = max(
        trip.total_yawns or 0,
        data.yawn_count
    )

    # Count distraction events
    if data.distraction_duration > 0:
        trip.distraction_events = (
            trip.distraction_events or 0
        ) + 1

    # Count serious eye-closure events
    if data.eye_closure_duration >= 8:
        trip.eye_closure_events = (
            trip.eye_closure_events or 0
        ) + 1

    trip.safety_score = data.safety_score

    # Store individual safety event
    safety_event = SafetyEvent(
        trip_id=trip.trip_id,
        eye_closure_duration=data.eye_closure_duration,
        yawn_count=data.yawn_count,
        distraction_duration=data.distraction_duration,
        safety_score=data.safety_score,
        alertness_score=data.alertness_score,
        timestamp=datetime.now()
    )

    db.add(safety_event)
    db.commit()

    return {
        "message": "Safety data recorded",
        "trip_id": trip_id,
        "current_safety_score": trip.safety_score
    }


# ============================================================
# END TRIP
# ============================================================

@app.post("/trips/{trip_id}/end")
def end_trip(
    trip_id: str,
    db: Session = Depends(get_db)
):

    trip = (
        db.query(Trip)
        .filter(Trip.trip_id == trip_id)
        .first()
    )

    if not trip:
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    if trip.status != "active":
        raise HTTPException(
            status_code=400,
            detail="Trip has already ended."
        )

    trip.status = "completed"
    trip.end_time = datetime.now()

    # ========================================================
    # CALCULATE FINAL SCORE
    # ========================================================

    score = 10.0

    # Yawning penalty
    if trip.total_yawns >= 5:
        score -= 2.0
    elif trip.total_yawns >= 2:
        score -= 1.0

    # Eye closure penalty
    if trip.eye_closure_events > 0:
        score -= min(
            trip.eye_closure_events * 0.5,
            2.0
        )

    # Distraction penalty
    if trip.distraction_events > 0:
        score -= min(
            trip.distraction_events * 0.5,
            2.0
        )

    score = max(
        0,
        round(score, 1)
    )

    trip.final_score = score

    # Convert score out of 10 to rating out of 5
    trip.rating = round(
        score / 2,
        1
    )

    # Performance category
    if score >= 9:
        trip.performance = "Excellent"
    elif score >= 7:
        trip.performance = "Good"
    elif score >= 5:
        trip.performance = "Needs Improvement"
    else:
        trip.performance = "High Risk"

    # ========================================================
    # UPDATE DRIVER AVERAGE
    # ========================================================

    driver = (
        db.query(Driver)
        .filter(
            Driver.driver_id == trip.driver_id
        )
        .first()
    )

    completed_trips = (
        db.query(Trip)
        .filter(
            Trip.driver_id == trip.driver_id,
            Trip.status == "completed",
            Trip.final_score.isnot(None)
        )
        .all()
    )

    if completed_trips and driver:
        average = sum(
            t.final_score
            for t in completed_trips
        ) / len(completed_trips)

        driver.average_score = round(
            average,
            1
        )

    db.commit()
    db.refresh(trip)

    return {
        "message": "Trip completed successfully",
        "trip_report": {
            "trip_id": trip.trip_id,
            "driver_id": trip.driver_id,
            "status": trip.status,
            "start_time": trip.start_time.isoformat()
            if trip.start_time else None,
            "end_time": trip.end_time.isoformat()
            if trip.end_time else None,
            "final_score": trip.final_score,
            "rating": trip.rating,
            "performance": trip.performance,
            "total_yawns": trip.total_yawns,
            "eye_closure_events": trip.eye_closure_events,
            "distraction_events": trip.distraction_events,
            "warnings": trip.warnings,
            "max_eye_closure": trip.max_eye_closure,
            "max_distraction_duration": trip.max_distraction_duration
        }
    }


# ============================================================
# TRIP REPORT
# ============================================================

@app.get("/trips/{trip_id}/report")
def get_trip_report(
    trip_id: str,
    db: Session = Depends(get_db)
):

    trip = (
        db.query(Trip)
        .filter(Trip.trip_id == trip_id)
        .first()
    )

    if not trip:
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    if trip.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Trip has not been completed yet."
        )

    return {
        "trip_id": trip.trip_id,
        "driver_id": trip.driver_id,
        "start_time": trip.start_time.isoformat()
        if trip.start_time else None,
        "end_time": trip.end_time.isoformat()
        if trip.end_time else None,

        "final_score": trip.final_score,
        "rating": trip.rating,
        "performance": trip.performance,

        "total_yawns": trip.total_yawns,
        "eye_closure_events": trip.eye_closure_events,
        "distraction_events": trip.distraction_events,
        "warnings": trip.warnings,

        "max_eye_closure": trip.max_eye_closure,
        "max_distraction_duration":
            trip.max_distraction_duration
    }


# ============================================================
# ALERT APIs
# ============================================================

@app.post("/alerts")
def create_alert(
    alert: AlertCreate,
    db: Session = Depends(get_db)
):

    trip = (
        db.query(Trip)
        .filter(Trip.trip_id == alert.trip_id)
        .first()
    )

    if not trip:
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    alert_id = "ALERT-" + uuid4().hex[:8].upper()

    new_alert = Alert(
        alert_id=alert_id,
        trip_id=alert.trip_id,
        alert_type=alert.alert_type,
        message=alert.message,
        severity=alert.severity,
        timestamp=datetime.now()
    )

    db.add(new_alert)

    trip.warnings = (trip.warnings or 0) + 1

    db.commit()
    db.refresh(new_alert)

    return {
        "message": "Alert recorded successfully",
        "alert": {
            "alert_id": new_alert.alert_id,
            "trip_id": new_alert.trip_id,
            "alert_type": new_alert.alert_type,
            "message": new_alert.message,
            "severity": new_alert.severity,
            "timestamp": new_alert.timestamp.isoformat()
        }
    }


@app.get("/alerts/{trip_id}")
def get_trip_alerts(
    trip_id: str,
    db: Session = Depends(get_db)
):

    trip = (
        db.query(Trip)
        .filter(Trip.trip_id == trip_id)
        .first()
    )

    if not trip:
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    trip_alerts = (
        db.query(Alert)
        .filter(Alert.trip_id == trip_id)
        .order_by(Alert.timestamp.asc())
        .all()
    )

    return {
        "trip_id": trip_id,
        "total_alerts": len(trip_alerts),
        "alerts": [
            {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "message": alert.message,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in trip_alerts
        ]
    }