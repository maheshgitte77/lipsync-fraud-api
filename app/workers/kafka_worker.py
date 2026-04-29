"""
Kafka producer/consumer loops used to offload proctor-signals jobs.

Flow
----
Submit:    POST /analyze/proctor-signals/submit → Kafka request topic
Consume:   worker loop → run orchestrator → publish result topic
Listen:    API instances subscribe to result topic and update job_store
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from app.models.proctor import ProctorSignalsRequest
from app.services.orchestration.proctor_orchestrator import proctor_orchestrator
from app.workers.job_store import job_store

logger = get_logger("workers.kafka")

_stop_event = threading.Event()
_worker_threads: list[threading.Thread] = []


class KafkaUnavailableError(RuntimeError):
    """Raised when kafka-python is not installed or brokers cannot be reached."""


# ---- producer / consumer factories -----------------------------------------


def _brokers() -> list[str]:
    return [b.strip() for b in settings.kafka.brokers.split(",") if b.strip()]


def make_producer():
    try:
        from kafka import KafkaProducer
    except ImportError as exc:
        raise KafkaUnavailableError(
            "kafka-python is required for Kafka mode. Install it via requirements.txt"
        ) from exc
    return KafkaProducer(
        bootstrap_servers=_brokers(),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )


def make_consumer(topic: str, group_id: str):
    try:
        from kafka import KafkaConsumer
    except ImportError as exc:
        raise KafkaUnavailableError(
            "kafka-python is required for Kafka mode. Install it via requirements.txt"
        ) from exc
    return KafkaConsumer(
        topic,
        bootstrap_servers=_brokers(),
        group_id=group_id,
        enable_auto_commit=True,
        auto_offset_reset="latest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        consumer_timeout_ms=1000,
    )


# ---- public enqueue helper --------------------------------------------------


def enqueue_proctor_job(job_id: str, payload: ProctorSignalsRequest) -> None:
    """Publish a proctor-signals request onto the request topic."""
    producer = make_producer()
    try:
        producer.send(
            settings.kafka.request_topic,
            {
                "jobId": job_id,
                "payload": payload.model_dump() if hasattr(payload, "model_dump") else payload.dict(),
                "ts": int(time.time()),
            },
        )
        producer.flush(timeout=10)
    finally:
        producer.close()


# ---- background loops -------------------------------------------------------


def _request_loop() -> None:
    req_consumer = make_consumer(settings.kafka.request_topic, settings.kafka.group)
    producer = make_producer()
    try:
        while not _stop_event.is_set():
            for msg in req_consumer:
                if _stop_event.is_set():
                    break
                data = msg.value or {}
                job_id = str(data.get("jobId") or "")
                raw_payload = data.get("payload") or {}
                if not job_id:
                    continue
                try:
                    result = _process(job_id, raw_payload)
                    producer.send(
                        settings.kafka.result_topic,
                        {"jobId": job_id, "status": "DONE", "result": result, "ts": int(time.time())},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("kafka worker failed job %s", job_id)
                    producer.send(
                        settings.kafka.result_topic,
                        {
                            "jobId": job_id,
                            "status": "FAILED",
                            "error": str(exc)[:4000],
                            "ts": int(time.time()),
                        },
                    )
    finally:
        req_consumer.close()
        producer.close()


def _result_loop() -> None:
    group = f"{settings.kafka.group}-results-{uuid.uuid4().hex[:6]}"
    consumer = make_consumer(settings.kafka.result_topic, group)
    try:
        while not _stop_event.is_set():
            for msg in consumer:
                if _stop_event.is_set():
                    break
                data = msg.value or {}
                job_id = str(data.get("jobId") or "")
                if not job_id:
                    continue
                status = str(data.get("status") or "UNKNOWN")
                if status == "DONE":
                    job_store.set(
                        job_id,
                        status="DONE",
                        updatedAt=int(time.time()),
                        completedAt=int(time.time()),
                        result=data.get("result"),
                        error=None,
                    )
                elif status == "FAILED":
                    job_store.set(
                        job_id,
                        status="FAILED",
                        updatedAt=int(time.time()),
                        completedAt=int(time.time()),
                        error=str(data.get("error") or "unknown error")[:4000],
                    )
    finally:
        consumer.close()


def _process(job_id: str, payload_dict: dict) -> dict[str, Any]:
    payload = ProctorSignalsRequest(**payload_dict)
    job_store.set(job_id, status="PROCESSING", updatedAt=int(time.time()))
    try:
        result = proctor_orchestrator.execute(payload, job_id=job_id)
        job_store.set(
            job_id,
            status="DONE",
            updatedAt=int(time.time()),
            completedAt=int(time.time()),
            result=result,
            error=None,
        )
        return result
    except Exception as exc:
        job_store.set(
            job_id,
            status="FAILED",
            updatedAt=int(time.time()),
            completedAt=int(time.time()),
            error=str(exc)[:4000],
        )
        raise


# ---- lifecycle hooks called from main --------------------------------------


def start_background_threads() -> None:
    if not settings.kafka.enabled:
        return
    t_result = threading.Thread(target=_result_loop, name="kafka-result-loop", daemon=True)
    t_result.start()
    _worker_threads.append(t_result)
    logger.info("kafka result listener started (topic=%s)", settings.kafka.result_topic)
    if settings.kafka.start_worker:
        t_req = threading.Thread(target=_request_loop, name="kafka-request-loop", daemon=True)
        t_req.start()
        _worker_threads.append(t_req)
        logger.info("kafka request worker started (topic=%s)", settings.kafka.request_topic)


def stop_background_threads() -> None:
    _stop_event.set()
