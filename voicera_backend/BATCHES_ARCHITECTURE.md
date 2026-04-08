# Batches Architecture (Frontend ↔ Backend)

This document explains how the **Batches** feature is connected across frontend and backend, how CSV uploads are stored, and what components are used.

---

## 1) End-to-end flow

1. User opens **Dashboard → Batches** page in frontend.
2. Frontend loads:
   - agents for org (`getAgents`)
   - batches list (`getBatches`)
3. User uploads a CSV for a selected `agent_type`.
4. Frontend sends multipart upload to Next.js API route:
   - `POST /api/batches`
5. Next.js proxies to backend:
   - `POST /api/v1/batches/upload`
6. Backend:
   - validates org/user, agent, file type/size/encoding
   - stores original CSV in GridFS
   - parses CSV and extracts `contact_number` rows
   - stores parsed rows in `BatchContacts`
   - stores metadata in `Batches`
7. Frontend reloads list and shows uploaded batch row.

---

## 2) Frontend pieces used

- UI page:  
  - `voicera_frontend/app/(dashboard)/batches/page.tsx`
- Frontend API helpers:  
  - `voicera_frontend/lib/api.ts`
  - `getBatches(agentType?)`
  - `uploadBatchCsv(file, orgId, agentType)`
  - `deleteBatch(batchId)`
- Next.js API proxy routes:  
  - `voicera_frontend/app/api/batches/route.ts`
  - `voicera_frontend/app/api/batches/[batchId]/route.ts`

Why proxy routes are used:
- keep backend base URL server-side
- forward auth header safely
- keep frontend calling internal `/api/*` routes

---

## 3) Backend pieces used

- Router:
  - `voicera_backend/app/routers/batches.py`
  - `GET /api/v1/batches`
  - `POST /api/v1/batches/upload`
  - `DELETE /api/v1/batches/{batch_id}`
- Service logic:
  - `voicera_backend/app/services/batch_service.py`
- Models:
  - `voicera_backend/app/models/schemas.py`
  - `BatchResponse`, `BatchUploadResponse`, `BatchDeleteResponse`
- Router registration:
  - `voicera_backend/app/main.py`

---

## 4) Storage model

### A) Raw immutable CSV file

- Stored in **Mongo GridFS** (collection namespace: `batch_csv_files`)
- Metadata includes:
  - `batch_id`
  - `org_id`
  - `agent_type`
  - `uploaded_at`

### B) Batch metadata

- Collection: `Batches`
- One document per uploaded CSV
- Key fields:
  - `batch_id`
  - `org_id`
  - `agent_type`
  - `original_filename`
  - `status` (currently `uploaded`)
  - `execution_status` (currently `not_started`)
  - `total_contacts`, `valid_contacts`, `invalid_contacts`
  - `source_file_id` (GridFS object id as string)
  - `created_at`, `updated_at`

### C) Parsed per-contact rows

- Collection: `BatchContacts`
- One document per CSV row
- Key fields:
  - `batch_id`, `org_id`, `agent_type`
  - `row_number`
  - `contact_number` (normalized)
  - `is_valid`
  - `status` (`queued` for valid, `invalid` for invalid rows)
  - `dynamic_fields` (all other CSV columns)
  - `created_at`, `updated_at`

---

## 5) Validation rules currently implemented

- only `.csv` files are accepted
- max file size: **10 MB**
- encoding must be UTF-8 (`utf-8-sig` accepted)
- CSV must contain required column: `contact_number`
- CSV must contain at least one data row
- `agent_type` must belong to the authenticated user’s organization
- user can only upload/delete/list within their own `org_id`

Phone normalization/validation:
- removes common separators (`space`, `-`, `(`, `)`, `.`)
- valid format regex: `+?` followed by `8..15` digits

---

## 6) Immutability and delete behavior

### Immutability
- Uploaded CSV is treated as immutable.
- No edit/replace API exists for an existing batch.
- To change data, user uploads a new batch.

### Delete
- `DELETE /api/v1/batches/{batch_id}` removes:
  - batch metadata from `Batches`
  - parsed contacts from `BatchContacts`
  - raw CSV from GridFS (if `source_file_id` is valid)

---

## 7) Database indexes used

Defined in `voicera_backend/app/database_init.py`:

- `Batches`
  - unique: `batch_id`
  - index: `org_id`
  - compound index: `(org_id, agent_type, created_at desc)`
- `BatchContacts`
  - unique compound: `(batch_id, row_number)`
  - compound index: `(org_id, agent_type)`
  - compound index: `(batch_id, status)`

These support fast listing, scoping by org/agent, and stable per-row uniqueness.

---

## 8) Current scope vs future batch-calling scope

### Implemented now
- upload CSV
- parse and persist contacts
- list uploaded batches
- delete batch

### Not implemented yet
- run/stop batch calling execution pipeline
- call attempt records per contact
- retries, throttling, scheduling windows
- download result files

The current data model is intentionally designed so batch-calling can later run from `BatchContacts` directly without re-reading CSV files.
