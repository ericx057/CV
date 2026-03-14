"""
Benchmark task corpus for the latent rollback test bench.

17 tasks across 6 types:
  multifile_refactor  — rename / add-param / interface-change / move-function (3-file each)
  cross_file_ref      — who-calls / type-provenance / import-chain
  single_hop          — return-type / param-type / constant-value
  double_hop          — transitive-return / inherited-method / field-access
  short_ctx           — single file, <800 tokens
  long_ctx            — 5-6 files, >10 000 tokens

Each BenchTask carries:
  - context              : full multi-file code string (the "document D")
  - question             : the natural language query
  - gold_answers         : list of acceptable answer strings
  - must_contain         : patterns expected in a correct refactor output
  - must_not_contain     : code-level forbidden patterns (not prose)
  - hop_count            : 1 or 2 (for F block strategy selection hints)
  - n_files              : number of synthetic files in context
  - context_tokens_approx: rough token estimate for amortization math

Import this module from any test file; no pytest infrastructure is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchTask:
    id: str
    task_type: str          # multifile_refactor | cross_file_ref | single_hop | double_hop | short_ctx | long_ctx
    context: str
    question: str
    gold_answers: list[str]
    instruction: str = ""   # for refactor tasks: what the model must do
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    hop_count: int = 1
    n_files: int = 1
    context_tokens_approx: int = 0

    def __post_init__(self) -> None:
        if self.context_tokens_approx == 0:
            # Approximate: words × 1.3 (subword factor)
            self.context_tokens_approx = max(1, int(len(self.context.split()) * 1.3))


# ---------------------------------------------------------------------------
# Shared short context (~300-400 tokens)
# ---------------------------------------------------------------------------

_SHORT_SINGLE_FILE = """\
# File: service.py
from typing import Optional
from dataclasses import dataclass

MAX_RETRIES: int = 3
DEFAULT_TIMEOUT: int = 30

@dataclass
class Config:
    host: str
    port: int
    timeout: int = DEFAULT_TIMEOUT

def fetch_config(host: str, port: int) -> Config:
    return Config(host=host, port=port)

def get_retry_limit() -> int:
    return MAX_RETRIES
"""

# ---------------------------------------------------------------------------
# Medium context helpers (~1500-3000 tokens per set)
# ---------------------------------------------------------------------------

_ORDERS_CTX = """\
# File: orders.py
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Order:
    id: str
    user_id: str
    total: float
    status: str

def process_order(order: Order, notify: bool = True) -> bool:
    \"\"\"Process an order: validate, charge, fulfil.\"\"\"
    if order.total <= 0:
        return False
    order.status = "processing"
    if notify:
        _send_confirmation(order)
    return True

def _send_confirmation(order: Order) -> None:
    print(f"Order {order.id} is being processed.")

def cancel_order(order: Order) -> bool:
    if order.status == "processing":
        return False
    order.status = "cancelled"
    return True

def get_order_total(orders: List[Order]) -> float:
    return sum(o.total for o in orders)

# File: router.py
from orders import process_order, cancel_order, Order

def handle_request(action: str, order: Order) -> dict:
    if action == "process":
        success = process_order(order)
        return {"ok": success}
    elif action == "cancel":
        success = cancel_order(order)
        return {"ok": success}
    return {"ok": False, "error": "unknown action"}

def bulk_process(orders: list) -> list:
    return [process_order(o) for o in orders]

def retry_failed(orders: list, max_retries: int = 3) -> int:
    processed = 0
    for o in orders:
        if o.status == "failed":
            if process_order(o):
                processed += 1
    return processed

# File: api.py
from router import handle_request
from orders import Order, process_order

def post_order(payload: dict) -> dict:
    order = Order(
        id=payload["id"],
        user_id=payload["user_id"],
        total=float(payload["total"]),
        status="pending",
    )
    return handle_request("process", order)

def direct_process(order_id: str, total: float) -> bool:
    order = Order(id=order_id, user_id="system", total=total, status="pending")
    return process_order(order)

def batch_post(payloads: list) -> list:
    return [post_order(p) for p in payloads]
"""

_MAILER_CTX = """\
# File: mailer.py
import smtplib
from typing import Optional

SMTP_HOST: str = "localhost"
SMTP_PORT: int = 587

def send_email(to: str, subject: str, body: str) -> bool:
    \"\"\"Send an email. Returns True on success.\"\"\"
    try:
        return True
    except smtplib.SMTPException:
        return False

def send_bulk(recipients: list, subject: str, body: str) -> int:
    sent = 0
    for r in recipients:
        if send_email(r, subject, body):
            sent += 1
    return sent

def preview_email(to: str, subject: str, body: str) -> dict:
    return {"to": to, "subject": subject, "body_len": len(body)}

# File: notification_service.py
from mailer import send_email

class NotificationService:
    def notify_user(self, user_email: str, message: str) -> bool:
        subject = "Notification"
        return send_email(user_email, subject, message)

    def notify_team(self, emails: list, message: str) -> list:
        return [send_email(e, "Team Update", message) for e in emails]

    def broadcast(self, emails: list, subject: str, body: str) -> int:
        return sum(1 for e in emails if send_email(e, subject, body))

# File: event_handler.py
from notification_service import NotificationService

_service = NotificationService()

def on_order_complete(user_email: str, order_id: str) -> None:
    msg = f"Order {order_id} is complete."
    _service.notify_user(user_email, msg)

def on_signup(user_email: str) -> None:
    _service.notify_user(user_email, "Welcome!")

def on_password_reset(user_email: str, token: str) -> None:
    _service.notify_user(user_email, f"Reset token: {token}")
"""

_STORAGE_CTX = """\
# File: storage_base.py
from abc import ABC, abstractmethod
from typing import Optional

class DataStore(ABC):
    @abstractmethod
    def get(self, key: str) -> str:
        \"\"\"Retrieve a value by key. Raises KeyError if not found.\"\"\"
        ...

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        ...

    def get_or_default(self, key: str, default: str) -> str:
        try:
            return self.get(key)
        except KeyError:
            return default

# File: redis_store.py
from storage_base import DataStore

class RedisStore(DataStore):
    def __init__(self, host: str = "localhost", port: int = 6379):
        self._data: dict = {}

    def get(self, key: str) -> str:
        return self._data[key]

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def flush(self) -> None:
        self._data.clear()

# File: file_store.py
from storage_base import DataStore

class FileStore(DataStore):
    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}

    def get(self, key: str) -> str:
        return self._data[key]

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def sync(self) -> None:
        pass  # write _data to self._path
"""

_HELPERS_CTX = """\
# File: helpers.py
import re
from typing import Optional

def validate_email(email: str) -> bool:
    \"\"\"Return True if email matches a basic email regex.\"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def format_name(first: str, last: str) -> str:
    return f"{first.strip()} {last.strip()}"

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

def truncate(text: str, max_len: int = 100) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."

# File: validators.py
import re

def validate_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))

def validate_phone(phone: str) -> bool:
    return bool(re.match(r'^\\+?[\\d\\s\\-]{7,15}$', phone))

def validate_zip(zipcode: str) -> bool:
    return bool(re.match(r'^\\d{5}(-\\d{4})?$', zipcode))

# File: signup.py
from helpers import validate_email, format_name

def register_user(email: str, first: str, last: str) -> dict:
    if not validate_email(email):
        return {"error": "invalid email"}
    name = format_name(first, last)
    return {"email": email, "name": name, "status": "registered"}

def update_profile(user_id: str, email: str, first: str, last: str) -> dict:
    if not validate_email(email):
        return {"error": "invalid email"}
    return {"user_id": user_id, "email": email, "name": format_name(first, last)}
"""

# ---------------------------------------------------------------------------
# Cross-file reference context
# ---------------------------------------------------------------------------

_NOTIFY_CTX = """\
# File: notifications.py
from typing import Optional

def notify_user(user_id: str, message: str, channel: str = "email") -> bool:
    \"\"\"Send a notification to a user.\"\"\"
    if not user_id or not message:
        return False
    print(f"[{channel}] -> {user_id}: {message}")
    return True

def notify_admin(message: str) -> bool:
    return notify_user("admin", message, channel="slack")

# File: order.py
from notifications import notify_user

def complete_order(order_id: str, user_id: str) -> dict:
    notify_user(user_id, f"Order {order_id} is complete.")
    return {"order_id": order_id, "status": "complete"}

def refund_order(order_id: str, user_id: str, amount: float) -> dict:
    notify_user(user_id, f"Refund of {amount} issued for order {order_id}.")
    return {"order_id": order_id, "refunded": amount}

def ship_order(order_id: str, user_id: str, tracking: str) -> dict:
    return {"order_id": order_id, "tracking": tracking}

# File: user.py
from notifications import notify_user

def register_user(user_id: str, email: str) -> dict:
    notify_user(user_id, "Welcome to the platform!")
    return {"user_id": user_id, "email": email}

def deactivate_user(user_id: str) -> bool:
    notify_user(user_id, "Your account has been deactivated.")
    return True

def reset_password(user_id: str, token: str) -> bool:
    notify_user(user_id, f"Password reset token: {token}")
    return True
"""

_STATUS_CTX = """\
# File: models.py
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"

class PaymentStatus(Enum):
    UNPAID = "unpaid"
    PAID = "paid"
    REFUNDED = "refunded"

# File: service.py
from models import OrderStatus, PaymentStatus
from typing import Optional

def transition_order(current: OrderStatus, action: str) -> Optional[OrderStatus]:
    if action == "process" and current == OrderStatus.PENDING:
        return OrderStatus.PROCESSING
    if action == "complete" and current == OrderStatus.PROCESSING:
        return OrderStatus.COMPLETE
    if action == "cancel":
        return OrderStatus.CANCELLED
    return None

def is_paid(status: PaymentStatus) -> bool:
    return status == PaymentStatus.PAID

# File: router.py
from service import transition_order
from models import OrderStatus

def handle_status_change(order_id: str, current_status: str, action: str) -> dict:
    current = OrderStatus(current_status)
    new_status = transition_order(current, action)
    if new_status is None:
        return {"error": "invalid transition"}
    return {"order_id": order_id, "status": new_status.value}
"""

_IMPORT_CHAIN_CTX = """\
# File: config.py
import os

DEFAULT_TIMEOUT: int = 30

def get_timeout() -> int:
    \"\"\"Return the configured HTTP timeout in seconds.\"\"\"
    return int(os.environ.get("HTTP_TIMEOUT", DEFAULT_TIMEOUT))

def get_base_url() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8000")

def get_api_key() -> str:
    return os.environ.get("API_KEY", "")

# File: service.py
from config import get_timeout

import urllib.request
import json

def fetch_resource(url: str) -> dict:
    timeout = get_timeout()
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())

def post_resource(url: str, data: dict) -> dict:
    timeout = get_timeout()
    return {}

# File: handler.py
from service import fetch_resource, post_resource

def handle_get(resource_id: str) -> dict:
    url = f"http://localhost:8000/resources/{resource_id}"
    return fetch_resource(url)

def handle_post(resource_id: str, payload: dict) -> dict:
    url = f"http://localhost:8000/resources/{resource_id}"
    return post_resource(url, payload)
"""

# ---------------------------------------------------------------------------
# Single-hop context
# ---------------------------------------------------------------------------

_USER_SERVICE_CTX = """\
# File: user_service.py
from typing import Optional
from dataclasses import dataclass

@dataclass
class UserDelta:
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None

@dataclass
class User:
    id: int
    name: str
    email: str
    role: str = "user"

MAX_RETRIES: int = 3
DEFAULT_PAGE_SIZE: int = 25

class UserService:
    def __init__(self) -> None:
        self._users: dict = {}

    def find_user(self, user_id: int) -> Optional[User]:
        \"\"\"Look up a user by ID.\"\"\"
        return self._users.get(user_id)

    def update_user(self, user_id: int, delta: UserDelta) -> Optional[User]:
        user = self.find_user(user_id)
        if user is None:
            return None
        if delta.name is not None:
            user.name = delta.name
        if delta.email is not None:
            user.email = delta.email
        return user

    def list_users(self, page: int = 0) -> list:
        start = page * DEFAULT_PAGE_SIZE
        return list(self._users.values())[start: start + DEFAULT_PAGE_SIZE]
"""

# ---------------------------------------------------------------------------
# Double-hop contexts
# ---------------------------------------------------------------------------

_SESSION_CTX = """\
# File: tokens.py
from dataclasses import dataclass
import secrets
import time

@dataclass
class SessionToken:
    value: str
    expires_at: float
    user_id: str

def generate_token(user_id: str, ttl: int = 3600) -> SessionToken:
    \"\"\"Generate a new session token for user_id.\"\"\"
    value = secrets.token_hex(32)
    expires_at = time.time() + ttl
    return SessionToken(value=value, expires_at=expires_at, user_id=user_id)

def validate_token(token: SessionToken) -> bool:
    return time.time() < token.expires_at

# File: auth.py
from tokens import generate_token, SessionToken

def create_session(user_id: str) -> SessionToken:
    \"\"\"Create a new authenticated session for user_id.\"\"\"
    return generate_token(user_id)

def refresh_session(token: SessionToken, ttl: int = 3600) -> SessionToken:
    return generate_token(token.user_id, ttl=ttl)

def invalidate_session(token: SessionToken) -> bool:
    token.expires_at = 0.0
    return True
"""

_INHERITANCE_CTX = """\
# File: base_user.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseUser:
    id: int
    name: str
    email: str
    role: str = "user"

    def save(self, name: str, email: str, role: str) -> None:
        \"\"\"Persist updated fields.\"\"\"
        self.name = name
        self.email = email
        self.role = role

    def deactivate(self) -> None:
        self.role = "inactive"

    def display(self) -> str:
        return f"{self.name} <{self.email}>"

# File: admin_user.py
from base_user import BaseUser

class AdminUser(BaseUser):
    \"\"\"Admin account — inherits BaseUser, adds permission management.\"\"\"

    def grant_permission(self, permission: str) -> None:
        pass

    def revoke_permission(self, permission: str) -> None:
        pass

    def list_permissions(self) -> list:
        return []
"""

_LOCATION_CTX = """\
# File: geography.py
from dataclasses import dataclass

@dataclass
class Location:
    street: str
    city: str
    country: str
    postal_code: str = ""
    region: str = ""

@dataclass
class GeoPoint:
    latitude: float
    longitude: float

def distance_km(a: GeoPoint, b: GeoPoint) -> float:
    \"\"\"Haversine distance between two points.\"\"\"
    return 0.0

# File: formatters.py
from geography import Location

def format_address(loc: Location) -> str:
    \"\"\"Format a Location for display.\"\"\"
    parts = [loc.street, loc.city, loc.country]
    return ", ".join(p for p in parts if p)

def format_short(loc: Location) -> str:
    return f"{loc.city}, {loc.country}"

def is_domestic(loc: Location, home_country: str = "US") -> bool:
    return loc.city != "" and loc.country == home_country
"""

# ---------------------------------------------------------------------------
# Long context (>10 000 tokens) — built by expanding the medium tasks
# ---------------------------------------------------------------------------

_LONG_REFACTOR_CTX = """\
# File: serializer.py
from typing import Any, Optional, Union
from dataclasses import dataclass, asdict
import json
import base64

ENCODING: str = "utf-8"
MAX_PAYLOAD_SIZE: int = 1_048_576  # 1 MB

@dataclass
class SerializationError(Exception):
    message: str
    field: Optional[str] = None

def serialize_payload(data: Any, compress: bool = False) -> bytes:
    \"\"\"
    Convert data to a canonical JSON byte string.

    Args:
        data: Any JSON-serializable Python object.
        compress: If True, apply base64 encoding (placeholder for real compression).

    Returns:
        UTF-8 encoded bytes of the JSON representation.

    Raises:
        SerializationError: If data is not JSON-serializable or exceeds MAX_PAYLOAD_SIZE.
    \"\"\"
    try:
        raw = json.dumps(data, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise SerializationError(message=str(exc))

    encoded = raw.encode(ENCODING)
    if len(encoded) > MAX_PAYLOAD_SIZE:
        raise SerializationError(
            message=f"Payload exceeds {MAX_PAYLOAD_SIZE} bytes"
        )

    if compress:
        return base64.b64encode(encoded)
    return encoded

def deserialize_payload(data: bytes, compressed: bool = False) -> Any:
    \"\"\"Reverse of serialize_payload.\"\"\"
    if compressed:
        data = base64.b64decode(data)
    return json.loads(data.decode(ENCODING))

def serialize_list(items: list, compress: bool = False) -> list:
    \"\"\"Serialize each item in a list independently.\"\"\"
    return [serialize_payload(item, compress=compress) for item in items]

def safe_serialize(data: Any) -> Optional[bytes]:
    \"\"\"Like serialize_payload but returns None on error instead of raising.\"\"\"
    try:
        return serialize_payload(data)
    except SerializationError:
        return None

# File: transport.py
from serializer import serialize_payload, deserialize_payload, SerializationError
from typing import Optional, Dict, Any
import urllib.request
import urllib.error

TRANSPORT_TIMEOUT: int = 10
RETRY_LIMIT: int = 3

class TransportError(Exception):
    pass

def send_payload(url: str, data: Any, headers: Optional[Dict[str, str]] = None) -> bytes:
    \"\"\"
    Serialize data and POST it to url.

    Uses serialize_payload internally; re-raises SerializationError if
    serialization fails.  Wraps network errors in TransportError.
    \"\"\"
    body = serialize_payload(data)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=TRANSPORT_TIMEOUT) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise TransportError(str(exc)) from exc

def send_batch(url: str, items: list) -> list:
    \"\"\"Send each item independently; return list of (success, response_or_error).\"\"\"
    results = []
    for item in items:
        try:
            resp = send_payload(url, item)
            results.append((True, resp))
        except (SerializationError, TransportError) as exc:
            results.append((False, str(exc)))
    return results

def receive_payload(raw: bytes) -> Any:
    return deserialize_payload(raw)

def health_check(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False

# File: pipeline.py
from transport import send_payload, send_batch, TransportError
from serializer import serialize_payload, SerializationError
from typing import Optional, List, Any

PIPELINE_VERSION: str = "1.0.0"

class Pipeline:
    \"\"\"
    End-to-end data pipeline: serialize, optionally transform, then transmit.
    \"\"\"

    def __init__(self, endpoint: str, retries: int = 3):
        self.endpoint = endpoint
        self.retries = retries

    def run(self, data: Any) -> bool:
        \"\"\"Run one item through the pipeline. Returns True on success.\"\"\"
        for attempt in range(self.retries):
            try:
                send_payload(self.endpoint, data)
                return True
            except (TransportError, SerializationError):
                if attempt == self.retries - 1:
                    return False
        return False

    def run_batch(self, items: List[Any]) -> List[bool]:
        return [self.run(item) for item in items]

    def dry_run(self, data: Any) -> Optional[bytes]:
        \"\"\"Serialize without transmitting. Useful for validation.\"\"\"
        try:
            return serialize_payload(data)
        except SerializationError:
            return None

    def validate(self, data: Any) -> bool:
        return self.dry_run(data) is not None

def build_pipeline(endpoint: str, retries: int = 3) -> Pipeline:
    return Pipeline(endpoint=endpoint, retries=retries)

# File: middleware.py
from pipeline import Pipeline, build_pipeline
from transport import send_payload
from serializer import serialize_payload
from typing import Any, Callable, Optional

class Middleware:
    \"\"\"
    Wraps a Pipeline with pre/post processing hooks.
    \"\"\"

    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline
        self._pre_hooks: list = []
        self._post_hooks: list = []

    def add_pre_hook(self, fn: Callable) -> None:
        self._pre_hooks.append(fn)

    def add_post_hook(self, fn: Callable) -> None:
        self._post_hooks.append(fn)

    def process(self, data: Any) -> bool:
        for hook in self._pre_hooks:
            data = hook(data)
        result = self._pipeline.run(data)
        for hook in self._post_hooks:
            hook(result)
        return result

def logging_middleware(pipeline: Pipeline) -> Middleware:
    mw = Middleware(pipeline)
    mw.add_pre_hook(lambda d: d)  # pass-through
    return mw

# File: api_gateway.py
from middleware import Middleware, logging_middleware
from pipeline import build_pipeline
from serializer import serialize_payload
from typing import Any

GATEWAY_VERSION: str = "2.1.0"

def create_gateway(endpoint: str) -> Middleware:
    \"\"\"
    Build the full stack: pipeline + logging middleware.
    Entry point for the serialize_payload → transport chain.
    \"\"\"
    pipeline = build_pipeline(endpoint)
    return logging_middleware(pipeline)

def dispatch(endpoint: str, data: Any) -> bool:
    gateway = create_gateway(endpoint)
    return gateway.process(data)

def preview(data: Any) -> Optional[bytes]:
    \"\"\"Preview what would be sent without transmitting. Uses serialize_payload.\"\"\"
    return serialize_payload(data)

# File: cli.py
from api_gateway import create_gateway, dispatch
from serializer import serialize_payload
import json
import sys

def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: cli.py <endpoint> <json_data>")
        return 1
    endpoint = sys.argv[1]
    data = json.loads(sys.argv[2])
    success = dispatch(endpoint, data)
    print("ok" if success else "failed")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
"""

_LONG_DOUBLE_HOP_CTX = """\
# File: result_types.py
from dataclasses import dataclass, field
from typing import Optional, List, Any
from enum import Enum

class ResultStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ProcessingResult:
    \"\"\"
    Unified result type returned by all processing pipelines.

    Fields:
        status:    SUCCESS | PARTIAL | FAILED | SKIPPED
        items:     processed items (may be partial on PARTIAL)
        errors:    list of error strings (empty on SUCCESS)
        metadata:  arbitrary key-value metadata
    \"\"\"
    status: ResultStatus
    items: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == ResultStatus.SUCCESS

@dataclass
class ValidationResult:
    valid: bool
    field_errors: dict = field(default_factory=dict)
    message: str = ""

# File: validator.py
from result_types import ValidationResult
from typing import Any

REQUIRED_FIELDS: List[str] = ["id", "type", "payload"]

def validate_item(item: Any) -> ValidationResult:
    \"\"\"Validate that item has required fields and correct types.\"\"\"
    if not isinstance(item, dict):
        return ValidationResult(valid=False, message="item must be a dict")
    errors = {}
    for f in REQUIRED_FIELDS:
        if f not in item:
            errors[f] = "missing"
    return ValidationResult(valid=len(errors) == 0, field_errors=errors)

def validate_batch(items: list) -> List[ValidationResult]:
    return [validate_item(i) for i in items]

def is_valid(item: Any) -> bool:
    return validate_item(item).valid

# File: processor.py
from result_types import ProcessingResult, ResultStatus
from validator import validate_item, validate_batch, ValidationResult
from typing import List, Any

BATCH_SIZE: int = 50

def process_item(item: Any) -> ProcessingResult:
    \"\"\"
    Process a single item.  Validates first; returns FAILED if invalid.
    \"\"\"
    vr = validate_item(item)
    if not vr.valid:
        return ProcessingResult(
            status=ResultStatus.FAILED,
            errors=[f"{k}: {v}" for k, v in vr.field_errors.items()]
        )
    return ProcessingResult(status=ResultStatus.SUCCESS, items=[item])

def process_batch(items: List[Any]) -> ProcessingResult:
    \"\"\"
    Process a batch. Returns PARTIAL if some fail, SUCCESS if all pass.
    \"\"\"
    results = [process_item(i) for i in items]
    successes = [r for r in results if r.ok]
    errors = [e for r in results for e in r.errors]
    if not results:
        return ProcessingResult(status=ResultStatus.SKIPPED)
    if len(successes) == len(results):
        return ProcessingResult(status=ResultStatus.SUCCESS, items=items)
    return ProcessingResult(
        status=ResultStatus.PARTIAL,
        items=[items[i] for i, r in enumerate(results) if r.ok],
        errors=errors,
    )

# File: order_processor.py
from processor import process_batch, process_item
from result_types import ProcessingResult, ResultStatus
from typing import List, Any

MAX_ORDER_SIZE: int = 100

class OrderProcessor:
    \"\"\"
    Orchestrates order processing: validates, processes in batches,
    and returns a unified ProcessingResult.
    \"\"\"

    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size

    def execute(self, orders: List[Any]) -> ProcessingResult:
        \"\"\"
        Process a list of orders.

        Splits into batches of self.batch_size, processes each,
        and merges the results.  Always returns ProcessingResult.
        \"\"\"
        if not orders:
            return ProcessingResult(status=ResultStatus.SKIPPED)
        if len(orders) > MAX_ORDER_SIZE:
            return ProcessingResult(
                status=ResultStatus.FAILED,
                errors=["batch exceeds MAX_ORDER_SIZE"]
            )

        all_items: List[Any] = []
        all_errors: List[str] = []

        for start in range(0, len(orders), self.batch_size):
            batch = orders[start: start + self.batch_size]
            result = process_batch(batch)
            all_items.extend(result.items)
            all_errors.extend(result.errors)

        if all_errors and not all_items:
            return ProcessingResult(status=ResultStatus.FAILED, errors=all_errors)
        if all_errors:
            return ProcessingResult(
                status=ResultStatus.PARTIAL,
                items=all_items,
                errors=all_errors,
            )
        return ProcessingResult(status=ResultStatus.SUCCESS, items=all_items)

    def execute_single(self, order: Any) -> ProcessingResult:
        return process_item(order)

# File: workflow.py
from order_processor import OrderProcessor
from result_types import ProcessingResult, ResultStatus
from typing import List, Any

def run_workflow(orders: List[Any], batch_size: int = 50) -> ProcessingResult:
    \"\"\"
    Top-level workflow entry point.  Creates an OrderProcessor and calls execute.
    \"\"\"
    processor = OrderProcessor(batch_size=batch_size)
    return processor.execute(orders)

def is_workflow_success(orders: List[Any]) -> bool:
    result = run_workflow(orders)
    return result.status == ResultStatus.SUCCESS
"""


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

BENCH_TASKS: list[BenchTask] = [

    # -----------------------------------------------------------------------
    # MULTIFILE REFACTOR
    # -----------------------------------------------------------------------

    BenchTask(
        id="rename_3file",
        task_type="multifile_refactor",
        context=_ORDERS_CTX,
        question="What changes are needed to rename process_order to handle_order across all files?",
        instruction="Rename `process_order` to `handle_order` in all three files.",
        gold_answers=["handle_order"],
        must_contain=["def handle_order", "handle_order(order)", "handle_order(o)"],
        must_not_contain=["def process_order", "process_order(order)", "process_order(o)"],
        hop_count=1,
        n_files=3,
    ),

    BenchTask(
        id="add_param_chain",
        task_type="multifile_refactor",
        context=_MAILER_CTX,
        question="How should dry_run: bool = False be added to send_email and propagated through NotificationService?",
        instruction=(
            "Add `dry_run: bool = False` to `send_email` in mailer.py and propagate it "
            "through NotificationService.notify_user and event_handler.on_order_complete."
        ),
        gold_answers=["dry_run"],
        must_contain=["dry_run: bool = False", "dry_run=dry_run"],
        must_not_contain=["def send_email(to: str, subject: str, body: str) -> bool"],
        hop_count=1,
        n_files=3,
    ),

    BenchTask(
        id="interface_change",
        task_type="multifile_refactor",
        context=_STORAGE_CTX,
        question="What is required to change DataStore.get to return Optional[str] instead of raising KeyError?",
        instruction=(
            "Change the return type of `DataStore.get` from `str` to `Optional[str]`, "
            "returning `None` when the key is absent. Update all implementations."
        ),
        gold_answers=["Optional[str]", "return None"],
        must_contain=["-> Optional[str]", "return None"],
        must_not_contain=["def get(self, key: str) -> str"],
        hop_count=1,
        n_files=3,
    ),

    BenchTask(
        id="move_function",
        task_type="multifile_refactor",
        context=_HELPERS_CTX,
        question="How do we move validate_email from helpers.py to validators.py and update signup.py?",
        instruction=(
            "Move `validate_email` from helpers.py to validators.py "
            "and update signup.py to import from validators."
        ),
        gold_answers=["from validators import validate_email"],
        must_contain=["from validators import validate_email"],
        must_not_contain=["from helpers import validate_email"],
        hop_count=1,
        n_files=3,
    ),

    # -----------------------------------------------------------------------
    # CROSS FILE REFERENCE
    # -----------------------------------------------------------------------

    BenchTask(
        id="who_calls",
        task_type="cross_file_ref",
        context=_NOTIFY_CTX,
        question="Which functions call notify_user?",
        gold_answers=["complete_order", "register_user", "refund_order", "deactivate_user",
                      "reset_password", "on_order_complete", "on_signup", "on_password_reset"],
        hop_count=1,
        n_files=3,
    ),

    BenchTask(
        id="type_provenance",
        task_type="cross_file_ref",
        context=_STATUS_CTX,
        question="In which file is OrderStatus defined?",
        gold_answers=["models.py", "models"],
        hop_count=1,
        n_files=3,
    ),

    BenchTask(
        id="import_chain",
        task_type="cross_file_ref",
        context=_IMPORT_CHAIN_CTX,
        question="What function from config.py does handler.py transitively use via service.py?",
        gold_answers=["get_timeout"],
        hop_count=2,
        n_files=3,
    ),

    # -----------------------------------------------------------------------
    # SINGLE HOP
    # -----------------------------------------------------------------------

    BenchTask(
        id="return_type_q",
        task_type="single_hop",
        context=_USER_SERVICE_CTX,
        question="What does UserService.find_user return?",
        gold_answers=["Optional[User]"],
        hop_count=1,
        n_files=1,
    ),

    BenchTask(
        id="param_type_q",
        task_type="single_hop",
        context=_USER_SERVICE_CTX,
        question="What type does UserService.update_user take as its delta parameter?",
        gold_answers=["UserDelta"],
        hop_count=1,
        n_files=1,
    ),

    BenchTask(
        id="constant_q",
        task_type="single_hop",
        context=_USER_SERVICE_CTX,
        question="What is the value of MAX_RETRIES?",
        gold_answers=["3"],
        hop_count=1,
        n_files=1,
    ),

    # -----------------------------------------------------------------------
    # DOUBLE HOP
    # -----------------------------------------------------------------------

    BenchTask(
        id="transitive_return",
        task_type="double_hop",
        context=_SESSION_CTX,
        question="What type does create_session return?",
        gold_answers=["SessionToken"],
        hop_count=2,
        n_files=2,
    ),

    BenchTask(
        id="inherited_method",
        task_type="double_hop",
        context=_INHERITANCE_CTX,
        question="What parameters does AdminUser.save take?",
        gold_answers=["name", "email", "role"],
        hop_count=2,
        n_files=2,
    ),

    BenchTask(
        id="field_access",
        task_type="double_hop",
        context=_LOCATION_CTX,
        question="What fields of Location does format_address access?",
        gold_answers=["city", "country", "street"],
        hop_count=2,
        n_files=2,
    ),

    # -----------------------------------------------------------------------
    # SHORT CONTEXT (<800 tokens)
    # -----------------------------------------------------------------------

    BenchTask(
        id="short_return",
        task_type="short_ctx",
        context=_SHORT_SINGLE_FILE,
        question="What does fetch_config return?",
        gold_answers=["Config"],
        hop_count=1,
        n_files=1,
    ),

    BenchTask(
        id="short_constant",
        task_type="short_ctx",
        context=_SHORT_SINGLE_FILE,
        question="What is the value of DEFAULT_TIMEOUT?",
        gold_answers=["30"],
        hop_count=1,
        n_files=1,
    ),

    # -----------------------------------------------------------------------
    # LONG CONTEXT (>10 000 tokens)
    # -----------------------------------------------------------------------

    BenchTask(
        id="long_refactor",
        task_type="long_ctx",
        context=_LONG_REFACTOR_CTX,
        question="What changes are needed to rename serialize_payload to encode_payload across all files?",
        instruction="Rename `serialize_payload` to `encode_payload` in all six files.",
        gold_answers=["encode_payload"],
        must_contain=["def encode_payload", "encode_payload("],
        must_not_contain=["def serialize_payload", ".serialize_payload("],
        hop_count=1,
        n_files=6,
    ),

    BenchTask(
        id="long_double_hop",
        task_type="long_ctx",
        context=_LONG_DOUBLE_HOP_CTX,
        question="What type does OrderProcessor.execute return?",
        gold_answers=["ProcessingResult"],
        hop_count=2,
        n_files=5,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_TYPES: tuple[str, ...] = (
    "multifile_refactor", "cross_file_ref", "single_hop",
    "double_hop", "short_ctx", "long_ctx",
)


def get_tasks_by_type(task_type: str) -> list[BenchTask]:
    return [t for t in BENCH_TASKS if t.task_type == task_type]


def get_task(task_id: str) -> BenchTask:
    for t in BENCH_TASKS:
        if t.id == task_id:
            return t
    raise KeyError(f"No task with id={task_id!r}")
