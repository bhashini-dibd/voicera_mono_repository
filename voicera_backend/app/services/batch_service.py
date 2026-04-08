"""
Batch CSV service for immutable batch uploads and parsed contacts storage.
"""
from __future__ import annotations

import csv
import io
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from gridfs import GridFS
from pymongo import ReturnDocument

from app.database import get_database

import logging

logger = logging.getLogger(__name__)

BATCH_COLLECTION = "Batches"
BATCH_CONTACT_COLLECTION = "BatchContacts"
GRIDFS_COLLECTION = "batch_csv_files"
AGENT_COLLECTION = "AgentConfig"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
CONTACT_COLUMN = "contact_number"
DEFAULT_BATCH_CONCURRENCY = 5
MIN_BATCH_CONCURRENCY = 1
MAX_BATCH_CONCURRENCY = 20


class BatchNotFoundError(Exception):
    """No batch found for org and batch_id."""


class BatchRunStateError(Exception):
    """Batch is in a state that cannot transition to requested action."""


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _normalize_contact_number(value: str) -> str:
    stripped = (value or "").strip()
    if not stripped:
        return ""
    normalized = re.sub(r"[ \-\(\)\.]", "", stripped)
    if normalized.startswith("++"):
        normalized = normalized.lstrip("+")
    return normalized


def _is_valid_contact_number(value: str) -> bool:
    return bool(re.fullmatch(r"\+?\d{8,15}", value))


def _normalize_batch_name(value: str) -> str:
    return (value or "").strip()


def _is_valid_concurrency(value: Any) -> bool:
    return type(value) is int and MIN_BATCH_CONCURRENCY <= value <= MAX_BATCH_CONCURRENCY


def validate_agent_for_org(org_id: str, agent_type: str) -> bool:
    db = get_database()
    agent_table = db[AGENT_COLLECTION]
    return (
        agent_table.find_one({"org_id": org_id, "agent_type": agent_type}) is not None
    )


def _insert_contacts_in_chunks(contacts: List[Dict[str, Any]]) -> None:
    if not contacts:
        return
    db = get_database()
    table = db[BATCH_CONTACT_COLLECTION]
    chunk_size = 1000
    for start in range(0, len(contacts), chunk_size):
        table.insert_many(contacts[start : start + chunk_size], ordered=False)


def _parse_csv_contacts(
    *, csv_content: str, batch_id: str, org_id: str, agent_type: str
) -> tuple[int, int, int]:
    stream = io.StringIO(csv_content)
    reader = csv.DictReader(stream)

    fieldnames = [name.strip() for name in (reader.fieldnames or []) if name]
    if CONTACT_COLUMN not in fieldnames:
        raise ValueError(f"CSV must include '{CONTACT_COLUMN}' column")

    now = _now_iso()
    contacts_to_insert: List[Dict[str, Any]] = []
    total_contacts = 0
    valid_contacts = 0
    invalid_contacts = 0

    for row_number, row in enumerate(reader, start=2):
        total_contacts += 1
        raw_contact = ""
        if row:
            for key, value in row.items():
                if (key or "").strip() == CONTACT_COLUMN:
                    raw_contact = value or ""
                    break
        normalized_contact = _normalize_contact_number(str(raw_contact))
        is_valid = _is_valid_contact_number(normalized_contact)
        if is_valid:
            valid_contacts += 1
        else:
            invalid_contacts += 1

        dynamic_fields = {
            (key or "").strip(): (value.strip() if isinstance(value, str) else value)
            for key, value in (row or {}).items()
            if key and (key or "").strip() != CONTACT_COLUMN
        }

        contacts_to_insert.append(
            {
                "batch_id": batch_id,
                "org_id": org_id,
                "agent_type": agent_type,
                "row_number": row_number,
                "contact_number": normalized_contact,
                "is_valid": is_valid,
                "status": "queued" if is_valid else "invalid",
                "dynamic_fields": dynamic_fields,
                "created_at": now,
                "updated_at": now,
            }
        )

    if total_contacts == 0:
        raise ValueError("CSV file has no contact rows")

    _insert_contacts_in_chunks(contacts_to_insert)
    return total_contacts, valid_contacts, invalid_contacts


def create_batch_from_csv(
    *,
    org_id: str,
    batch_name: str,
    agent_type: str,
    original_filename: str,
    csv_bytes: bytes,
) -> Dict[str, Any]:
    if len(csv_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"CSV too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"
        )

    try:
        csv_content = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("CSV must be UTF-8 encoded") from e

    db = get_database()
    fs = GridFS(db, collection=GRIDFS_COLLECTION)
    batches = db[BATCH_COLLECTION]
    contacts = db[BATCH_CONTACT_COLLECTION]

    normalized_batch_name = _normalize_batch_name(batch_name)
    if not normalized_batch_name:
        raise ValueError("Batch name is required")

    existing_batch = batches.find_one(
        {"org_id": org_id, "batch_name": normalized_batch_name},
        {"_id": 1},
    )
    if existing_batch:
        raise ValueError("Batch name already exists in this organization")

    batch_id = str(uuid.uuid4())
    now = _now_iso()

    file_id = fs.put(
        csv_bytes,
        filename=original_filename,
        content_type="text/csv",
        metadata={
            "batch_id": batch_id,
            "org_id": org_id,
            "batch_name": normalized_batch_name,
            "agent_type": agent_type,
            "uploaded_at": now,
        },
    )

    try:
        total_contacts, valid_contacts, invalid_contacts = _parse_csv_contacts(
            csv_content=csv_content,
            batch_id=batch_id,
            org_id=org_id,
            agent_type=agent_type,
        )
    except Exception:
        try:
            if fs.exists(file_id):
                fs.delete(file_id)
        except Exception:
            logger.warning("Failed to clean up GridFS file for failed batch upload batch_id=%s", batch_id)
        raise

    batch_doc = {
        "batch_id": batch_id,
        "org_id": org_id,
        "batch_name": normalized_batch_name,
        "agent_type": agent_type,
        "concurrency": DEFAULT_BATCH_CONCURRENCY,
        "original_filename": original_filename,
        "status": "uploaded",
        "execution_status": "not_started",
        "total_contacts": total_contacts,
        "valid_contacts": valid_contacts,
        "invalid_contacts": invalid_contacts,
        "attempted_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "source_file_id": str(file_id),
        "created_at": now,
        "updated_at": now,
    }
    try:
        batches.insert_one(batch_doc)
    except Exception as exc:
        contacts.delete_many({"batch_id": batch_id, "org_id": org_id})
        try:
            if fs.exists(file_id):
                fs.delete(file_id)
        except Exception:
            logger.warning("Failed to clean up GridFS file after batch insert failure batch_id=%s", batch_id)
        error_message = str(exc).lower()
        if "duplicate" in error_message and "org_batch_name_unique" in error_message:
            raise ValueError("Batch name already exists in this organization") from exc
        raise

    logger.info(
        "Batch uploaded batch_id=%s batch_name=%s org_id=%s agent_type=%s total=%s valid=%s invalid=%s",
        batch_id,
        normalized_batch_name,
        org_id,
        agent_type,
        total_contacts,
        valid_contacts,
        invalid_contacts,
    )
    return batch_doc


def list_batches(org_id: str, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
    db = get_database()
    table = db[BATCH_COLLECTION]
    query: Dict[str, Any] = {"org_id": org_id}
    if agent_type:
        query["agent_type"] = agent_type

    cursor = table.find(query).sort("created_at", -1)
    out: List[Dict[str, Any]] = []
    for doc in cursor:
        doc.pop("_id", None)
        batch_name = _normalize_batch_name(str(doc.get("batch_name") or ""))
        if not batch_name:
            fallback_batch_name = str(doc.get("original_filename") or "").strip()
            batch_name = fallback_batch_name or f"batch-{str(doc.get('batch_id') or '')[:8]}"
        doc["batch_name"] = batch_name
        doc["concurrency"] = (
            int(doc["concurrency"])
            if _is_valid_concurrency(doc.get("concurrency"))
            else DEFAULT_BATCH_CONCURRENCY
        )
        out.append(doc)
    return out


def delete_batch(org_id: str, batch_id: str) -> None:
    db = get_database()
    batches = db[BATCH_COLLECTION]
    contacts = db[BATCH_CONTACT_COLLECTION]
    fs = GridFS(db, collection=GRIDFS_COLLECTION)

    batch_doc = batches.find_one({"batch_id": batch_id, "org_id": org_id})
    if not batch_doc:
        raise BatchNotFoundError()

    source_file_id = batch_doc.get("source_file_id")
    if source_file_id:
        try:
            object_id = ObjectId(source_file_id)
            if fs.exists(object_id):
                fs.delete(object_id)
        except (InvalidId, TypeError):
            logger.warning(
                "Skipping GridFS delete: invalid source_file_id batch_id=%s source_file_id=%s",
                batch_id,
                source_file_id,
            )

    contacts.delete_many({"batch_id": batch_id, "org_id": org_id})
    batches.delete_one({"batch_id": batch_id, "org_id": org_id})


def _get_batch_for_org(org_id: str, batch_id: str) -> Dict[str, Any]:
    db = get_database()
    batch_doc = db[BATCH_COLLECTION].find_one({"batch_id": batch_id, "org_id": org_id})
    if not batch_doc:
        raise BatchNotFoundError()
    batch_doc.pop("_id", None)
    return batch_doc


def _ensure_batch_runnable(batch_doc: Dict[str, Any]) -> None:
    execution_status = batch_doc.get("execution_status", "not_started")
    if execution_status in {"running", "completed"}:
        raise BatchRunStateError(
            f"Batch cannot be started when execution_status is '{execution_status}'"
        )


def _mark_batch_running(org_id: str, batch_id: str) -> None:
    db = get_database()
    db[BATCH_COLLECTION].update_one(
        {"batch_id": batch_id, "org_id": org_id},
        {
            "$set": {
                "status": "processing",
                "execution_status": "running",
                "updated_at": _now_iso(),
            }
        },
    )


def _mark_batch_stopping(org_id: str, batch_id: str) -> None:
    db = get_database()
    db[BATCH_COLLECTION].update_one(
        {"batch_id": batch_id, "org_id": org_id},
        {"$set": {"execution_status": "stopping", "updated_at": _now_iso()}},
    )


def _mark_batch_finished(
    org_id: str,
    batch_id: str,
    *,
    status: str,
    execution_status: str,
    error_message: Optional[str] = None,
) -> None:
    db = get_database()
    update_payload: Dict[str, Any] = {
        "status": status,
        "execution_status": execution_status,
        "updated_at": _now_iso(),
    }
    if error_message:
        update_payload["error_message"] = error_message[:1000]
    else:
        update_payload["error_message"] = None

    db[BATCH_COLLECTION].update_one(
        {"batch_id": batch_id, "org_id": org_id},
        {"$set": update_payload},
    )


def _acquire_next_contact(batch_id: str, org_id: str) -> Optional[Dict[str, Any]]:
    db = get_database()
    contacts = db[BATCH_CONTACT_COLLECTION]
    doc = contacts.find_one_and_update(
        {"batch_id": batch_id, "org_id": org_id, "is_valid": True, "status": "queued"},
        {"$set": {"status": "dialing", "updated_at": _now_iso()}},
        sort=[("row_number", 1)],
        return_document=ReturnDocument.AFTER,
    )
    if doc:
        doc.pop("_id", None)
    return doc


def _get_agent_call_config(org_id: str, agent_type: str) -> Dict[str, Any]:
    db = get_database()
    agent = db[AGENT_COLLECTION].find_one({"org_id": org_id, "agent_type": agent_type})
    if not agent:
        raise ValueError("Agent not found for organization")

    agent_id = agent.get("agent_id")
    if not agent_id:
        raise ValueError("Agent has no agent_id configured")

    caller_id = agent.get("phone_number")
    if not caller_id:
        raise ValueError("Please attach a phone number to this agent before running batch calls")

    return {
        "agent_id": agent_id,
        "caller_id": caller_id,
    }


def _finalize_contact_result(
    *,
    batch_id: str,
    org_id: str,
    row_number: int,
    ok: bool,
    error: Optional[str],
) -> None:
    db = get_database()
    contacts = db[BATCH_CONTACT_COLLECTION]
    batches = db[BATCH_COLLECTION]
    now = _now_iso()

    contact_status = "called" if ok else "failed"
    contact_update: Dict[str, Any] = {
        "status": contact_status,
        "updated_at": now,
    }
    if error:
        contact_update["error_message"] = error[:1000]
    else:
        contact_update["error_message"] = None

    contacts.update_one(
        {"batch_id": batch_id, "org_id": org_id, "row_number": row_number},
        {"$set": contact_update},
    )

    batch_increments = {"attempted_calls": 1}
    if ok:
        batch_increments["successful_calls"] = 1
    else:
        batch_increments["failed_calls"] = 1

    batches.update_one(
        {"batch_id": batch_id, "org_id": org_id},
        {"$inc": batch_increments, "$set": {"updated_at": now}},
    )


def run_batch(
    org_id: str,
    batch_id: str,
    agent_type: Optional[str] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    batch_doc = _get_batch_for_org(org_id, batch_id)
    _ensure_batch_runnable(batch_doc)
    valid_contacts = int(batch_doc.get("valid_contacts", 0) or 0)
    if valid_contacts == 0:
        raise BatchRunStateError(
            "This batch has no valid contacts. Please upload a CSV with valid contact numbers."
        )
    already_attempted = int(batch_doc.get("attempted_calls", 0) or 0)
    queued_contacts = get_database()[BATCH_CONTACT_COLLECTION].count_documents(
        {
            "batch_id": batch_id,
            "org_id": org_id,
            "is_valid": True,
            "status": "queued",
        }
    )
    if queued_contacts == 0 and already_attempted > 0:
        raise BatchRunStateError(
            "This batch is already processed. Upload a new batch to place calls again."
        )

    selected_agent_type = (agent_type or batch_doc.get("agent_type") or "").strip()
    if not selected_agent_type:
        raise BatchRunStateError("Agent selection is required to run this batch")
    if not validate_agent_for_org(org_id, selected_agent_type):
        raise BatchRunStateError("Invalid agent selected for this organization")
    _get_agent_call_config(org_id, selected_agent_type)

    if concurrency is not None:
        if not _is_valid_concurrency(concurrency):
            raise BatchRunStateError(
                f"Concurrency must be between {MIN_BATCH_CONCURRENCY} and {MAX_BATCH_CONCURRENCY}"
            )
        selected_concurrency = concurrency
    else:
        existing_concurrency = batch_doc.get("concurrency")
        selected_concurrency = (
            int(existing_concurrency)
            if _is_valid_concurrency(existing_concurrency)
            else DEFAULT_BATCH_CONCURRENCY
        )

    get_database()[BATCH_COLLECTION].update_one(
        {"batch_id": batch_id, "org_id": org_id},
        {
            "$set": {
                "agent_type": selected_agent_type,
                "concurrency": selected_concurrency,
                "updated_at": _now_iso(),
                "error_message": None,
            }
        },
    )
    _mark_batch_running(org_id, batch_id)
    return {
        "status": "success",
        "message": (
            f"Batch execution prepared with agent '{selected_agent_type}' "
            f"at concurrency {selected_concurrency}"
        ),
        "agent_type": selected_agent_type,
        "concurrency": selected_concurrency,
    }


def stop_batch(org_id: str, batch_id: str) -> Dict[str, Any]:
    batch_doc = _get_batch_for_org(org_id, batch_id)
    if batch_doc.get("execution_status") != "running":
        raise BatchRunStateError("Batch is not running")
    _mark_batch_stopping(org_id, batch_id)
    return {"status": "success", "message": "Batch stop requested"}


def claim_next_contact_for_execution(org_id: str, batch_id: str) -> Optional[Dict[str, Any]]:
    return _acquire_next_contact(batch_id=batch_id, org_id=org_id)


def report_contact_execution_result(
    *,
    org_id: str,
    batch_id: str,
    row_number: int,
    ok: bool,
    error: Optional[str],
) -> None:
    _finalize_contact_result(
        batch_id=batch_id,
        org_id=org_id,
        row_number=row_number,
        ok=ok,
        error=error,
    )


def get_agent_call_config_for_batch(org_id: str, agent_type: str) -> Dict[str, Any]:
    return _get_agent_call_config(org_id, agent_type)


def finalize_batch_execution(org_id: str, batch_id: str, stopped: bool = False) -> Dict[str, Any]:
    db = get_database()
    batch_snapshot = db[BATCH_COLLECTION].find_one(
        {"batch_id": batch_id, "org_id": org_id},
        {"attempted_calls": 1, "successful_calls": 1, "failed_calls": 1},
    ) or {}
    attempted_calls = int(batch_snapshot.get("attempted_calls", 0) or 0)
    successful_calls = int(batch_snapshot.get("successful_calls", 0) or 0)
    failed_calls = int(batch_snapshot.get("failed_calls", 0) or 0)

    if stopped:
        _mark_batch_finished(
            org_id,
            batch_id,
            status="uploaded",
            execution_status="stopped",
        )
        return {"status": "stopped"}

    if attempted_calls > 0 and successful_calls == 0 and failed_calls > 0:
        _mark_batch_finished(
            org_id,
            batch_id,
            status="failed",
            execution_status="failed",
            error_message=(
                "All outbound call attempts failed. "
                "Check voice server URL/connectivity and agent phone setup."
            ),
        )
        return {"status": "failed"}

    _mark_batch_finished(
        org_id,
        batch_id,
        status="completed",
        execution_status="completed",
    )
    return {"status": "completed"}


def mark_batch_start_failure(org_id: str, batch_id: str, error_message: str) -> None:
    _mark_batch_finished(
        org_id,
        batch_id,
        status="failed",
        execution_status="failed",
        error_message=error_message,
    )
