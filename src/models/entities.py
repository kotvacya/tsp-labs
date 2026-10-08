from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import ForeignKey, String, Text, Float, Integer, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.database import Base

def utcnow():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(150))
    role: Mapped[str] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    datasets: Mapped[List["Dataset"]] = relationship(back_populates="user")
    experiments: Mapped[List["BenchmarkExperiment"]] = relationship(back_populates="user")

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    storage_dir: Mapped[str] = mapped_column(String(255))
    total_samples: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship(back_populates="datasets")
    samples: Mapped[List["DatasetSample"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    experiments: Mapped[List["BenchmarkExperiment"]] = relationship(back_populates="dataset")

class DatasetSample(Base):
    __tablename__ = "dataset_samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    sample_index: Mapped[int] = mapped_column(Integer)
    rgb_image_rel_path: Mapped[str] = mapped_column(String(255))
    gt_depth_rel_path: Mapped[str] = mapped_column(String(255))

    dataset: Mapped["Dataset"] = relationship(back_populates="samples")
    metric_records: Mapped[List["MetricRecord"]] = relationship(back_populates="sample", cascade="all, delete-orphan")

class NeuralModel(Base):
    __tablename__ = "neural_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    architecture: Mapped[str] = mapped_column(String(50))
    backbone: Mapped[str] = mapped_column(String(50))
    weights_rel_path: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    runs: Mapped[List["ExperimentModelRun"]] = relationship(back_populates="model", cascade="all, delete-orphan")

class BenchmarkExperiment(Base):
    __tablename__ = "benchmark_experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    title: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(30), default="PENDING")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship(back_populates="experiments")
    dataset: Mapped["Dataset"] = relationship(back_populates="experiments")
    model_runs: Mapped[List["ExperimentModelRun"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")

class ExperimentModelRun(Base):
    __tablename__ = "experiment_model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("benchmark_experiments.id", ondelete="CASCADE"))
    model_id: Mapped[int] = mapped_column(ForeignKey("neural_models.id"))
    mean_silog: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_absrel: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_delta_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    run_status: Mapped[str] = mapped_column(String(30), default="PENDING")

    experiment: Mapped["BenchmarkExperiment"] = relationship(back_populates="model_runs")
    model: Mapped["NeuralModel"] = relationship(back_populates="runs")
    metric_records: Mapped[List["MetricRecord"]] = relationship(back_populates="run", cascade="all, delete-orphan")

class MetricRecord(Base):
    __tablename__ = "metric_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("experiment_model_runs.id", ondelete="CASCADE"))
    sample_id: Mapped[int] = mapped_column(ForeignKey("dataset_samples.id", ondelete="CASCADE"))
    silog: Mapped[float] = mapped_column(Float)
    abs_rel: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    delta_1: Mapped[float] = mapped_column(Float)
    inference_time_ms: Mapped[float] = mapped_column(Float)

    run: Mapped["ExperimentModelRun"] = relationship(back_populates="metric_records")
    sample: Mapped["DatasetSample"] = relationship(back_populates="metric_records")
