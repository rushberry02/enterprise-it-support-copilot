import runtime_labs

# Before the imports below: they build AI objects at import time, and
# RuntimeLabs only traces what is created after init().
runtime_labs.init()

from uuid import uuid4
from hashlib import sha256
import json
from typing import Literal
from fastapi import FastAPI, Header, HTTPException, Path, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from app.agent.graph import app as agent
from app.agent.meridian import MeridianContext
from app.services.meridian_auth import backend_request
from app.agent.graph import CHECKPOINT_DB_PATH
from app.services.conversation_history import ConversationHistory
from app.api.admin import router as admin_router
from app.api.admin_chat import create_router as admin_chat_router
from app.services.admin_actions import AdminActionStore
from app.api.browser_auth import router as browser_auth_router

app = FastAPI(title="IT Support Copilot API", version="1.0.0")
history = ConversationHistory(agent, CHECKPOINT_DB_PATH)
app.include_router(admin_router)
app.include_router(browser_auth_router)
admin_actions = AdminActionStore(CHECKPOINT_DB_PATH)
app.include_router(admin_chat_router(admin_actions, agent, history))


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = Field(default=None, max_length=100)
    request_id: str | None = Field(default=None, min_length=1, max_length=100, pattern=r'^[A-Za-z0-9._:-]+$')


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=1, max_length=254)
    password: SecretStr = Field(min_length=1, max_length=128)
    role: Literal["employee", "admin"] | None = None


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refresh_token: SecretStr = Field(min_length=1, max_length=256)


class PasswordSetup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: SecretStr = Field(min_length=32, max_length=256)
    password: SecretStr = Field(min_length=12, max_length=128)


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str | None = Field(default=None, min_length=1, max_length=80)
    archived: bool | None = None

    @model_validator(mode="after")
    def valid_changes(self):
        if not self.model_fields_set or ("archived" in self.model_fields_set and self.archived is None):
            raise ValueError("Provide a title or archive state")
        return self


def no_store(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


@app.exception_handler(RequestValidationError)
async def safe_auth_validation(request, exc):
    if request.url.path.startswith("/auth/"):
        return JSONResponse(status_code=422, content={"detail": "Enter a valid email and password"}, headers={"Cache-Control": "no-store"})
    return await request_validation_exception_handler(request, exc)


@app.get("/")
def home():
    return {"message": "Welcome to IT Support Copilot API"}


@app.post("/auth/login")
def login(data: LoginRequest, response: Response):
    no_store(response)
    body = {"email": data.email, "password": data.password.get_secret_value()}
    if data.role is not None:
        body["role"] = data.role
    return backend_request("/auth/login", method="POST", body=body)


@app.post("/auth/refresh")
def refresh(data: RefreshRequest, response: Response):
    no_store(response)
    return backend_request("/auth/refresh", method="POST", body={"refresh_token": data.refresh_token.get_secret_value()})


@app.post("/auth/setup-password")
def setup_password(data: PasswordSetup, response: Response):
    no_store(response)
    return backend_request("/auth/setup-password", method="POST", body={"token": data.token.get_secret_value(), "password": data.password.get_secret_value()})


@app.post("/auth/logout", status_code=204)
def logout(data: RefreshRequest):
    backend_request("/auth/logout", method="POST", body={"refresh_token": data.refresh_token.get_secret_value()})
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


@app.get("/auth/me")
def me(response: Response, authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in required")
    no_store(response)
    return backend_request("/auth/me", authorization=authorization)


def chat_identity(authorization, employee_key=None):
    if authorization:
        if not authorization.startswith("Bearer ") or not authorization[7:].strip():
            raise HTTPException(status_code=401, detail="Sign in required")
        identity = backend_request("/auth/me", authorization=authorization)
        credential = authorization[7:]
        scope = f"user:{identity['id']}"
    elif employee_key:
        identity = backend_request("/identity", legacy_key=employee_key)
        credential = employee_key
        scope = f"employee:{identity['employee_id']}"
    else:
        raise HTTPException(status_code=401, detail="Sign in required")
    if identity.get("role") not in {"employee", "admin"}:
        raise HTTPException(status_code=403, detail="Employee login required")
    return identity, credential, scope


@app.get("/conversations")
def list_conversations(response: Response, authorization: str | None = Header(default=None), offset: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=100), query: str = Query(default="", max_length=200), archived: bool = False):
    _, _, scope = chat_identity(authorization)
    no_store(response)
    return history.list(scope, offset, limit, query, archived)


@app.get("/conversations/{thread_id}")
def get_conversation(response: Response, thread_id: str = Path(min_length=1, max_length=100), authorization: str | None = Header(default=None)):
    _, _, scope = chat_identity(authorization)
    no_store(response)
    return history.get(scope, thread_id)


@app.patch("/conversations/{thread_id}")
def update_conversation(data: ConversationUpdate, response: Response, thread_id: str = Path(min_length=1, max_length=100), authorization: str | None = Header(default=None)):
    _, _, scope = chat_identity(authorization)
    no_store(response)
    return history.update(scope, thread_id, **data.model_dump(exclude_unset=True))


@app.post("/ask")
def ask_question(request: QuestionRequest, authorization: str | None = Header(default=None), employee_key: str | None = Header(default=None, alias="X-Meridian-Employee-Key")):
    identity, credential, scope = chat_identity(authorization, employee_key)
    thread_id = request.thread_id or str(uuid4())
    if history.is_archived(scope, thread_id):
        raise HTTPException(status_code=409, detail="Restore this conversation before continuing it")
    request_id = request.request_id or str(uuid4())
    message_id = sha256(json.dumps([thread_id, request_id]).encode()).hexdigest()
    result = agent.invoke(
        {"question": request.question, "messages": [HumanMessage(content=request.question, id=message_id)]},
        config={"configurable": {"thread_id": f"{scope}:{thread_id}"}},
        context=MeridianContext(employee_api_key=credential, operator_role=identity["role"]),
    )
    response = {"question": request.question, "answer": result["answer"], "sources": result.get("sources", []), "status": result.get("status", "unknown"), "thread_id": thread_id, "request_id": request_id}
    if result.get("status") == "admin_confirmation_required" and result.get("admin_proposal"):
        if identity["role"] != "admin" or "id" not in identity:
            raise HTTPException(status_code=403, detail="Administrator access required")
        response["pending_action"] = admin_actions.propose(scope, thread_id, result["admin_proposal"], authorization)
    return response
