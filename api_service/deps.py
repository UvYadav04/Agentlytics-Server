"""FastAPI dependencies shared across routers."""
import jwt
from fastapi import Cookie, HTTPException, status

from shared.auth import ACCESS_TOKEN_COOKIE_NAME, decode_access_token
from shared.db import get_db
from shared.models.chat import COLLECTION as CHATS
from shared.models.chat import Chat
from shared.models.file import COLLECTION as FILES
from shared.models.file import File
from shared.models.investigation import COLLECTION as INVESTIGATIONS
from shared.models.investigation import Investigation
from shared.models.user import COLLECTION as USERS
from shared.models.user import User
from shared.models.workspace import COLLECTION as WORKSPACES
from shared.models.workspace import Workspace


async def get_current_user(access_token: str | None = Cookie(default=None, alias=ACCESS_TOKEN_COOKIE_NAME)) -> User:
    if not access_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        payload = decode_access_token(access_token)
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired session")

    doc = await get_db()[USERS].find_one({"_id": payload.sub})
    user = User.from_mongo(doc)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return user


async def get_owned_workspace(workspace_id: str, user: User) -> Workspace:
    """Fetches a workspace and 404s (never 403s - don't leak existence) if it
    doesn't exist or doesn't belong to the current user."""
    doc = await get_db()[WORKSPACES].find_one({"_id": workspace_id, "user_id": user.id})
    workspace = Workspace.from_mongo(doc)
    if workspace is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Workspace not found")
    return workspace


async def get_owned_file(file_id: str, user: User) -> File:
    doc = await get_db()[FILES].find_one({"_id": file_id})
    file = File.from_mongo(doc)
    if file is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    # Ownership is transitive through the workspace.
    await get_owned_workspace(file.workspace_id, user)
    return file


async def get_owned_chat(chat_id: str, user: User) -> Chat:
    doc = await get_db()[CHATS].find_one({"_id": chat_id})
    chat = Chat.from_mongo(doc)
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found")
    await get_owned_workspace(chat.workspace_id, user)
    return chat


async def get_owned_investigation(investigation_id: str, user: User) -> Investigation:
    doc = await get_db()[INVESTIGATIONS].find_one({"_id": investigation_id})
    investigation = Investigation.from_mongo(doc)
    if investigation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Investigation not found")
    await get_owned_workspace(investigation.workspace_id, user)
    return investigation
