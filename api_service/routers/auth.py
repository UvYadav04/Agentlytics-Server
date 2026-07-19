import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from api_service.deps import get_current_user
from shared.auth import ACCESS_TOKEN_COOKIE_NAME, create_access_token, verify_google_id_token
from shared.config import get_settings
from shared.db import get_db
from shared.models.user import COLLECTION as USERS
from shared.models.user import User
from shared.models.workspace import COLLECTION as WORKSPACES
from shared.models.workspace import Workspace

logger = logging.getLogger("api.auth")

router = APIRouter(tags=["auth"])


class GoogleLoginRequest(BaseModel):
    id_token: str


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    picture: str | None = None


def _cookie_max_age() -> int:
    minutes = int(get_settings().get("JWT_EXPIRE_MINUTES", "43200") or "43200")
    return minutes * 60


def _set_auth_cookie(response: Response, token: str) -> None:
    # secure=True in production (HTTPS) - set via env-driven flag if you also
    # deploy dev over http. Left True here since prod is the deploy target
    # in Phase 11; flip to settings-driven if you need local http testing.
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=_cookie_max_age(),
        path="/",
    )


@router.post("/auth/google", response_model=UserOut)
async def google_login(body: GoogleLoginRequest, response: Response):
    try:
        profile = verify_google_id_token(body.id_token)
    except Exception as exc:
        # The client only ever sees the generic 401 below (don't leak verifier
        # internals) - but log the real reason, since "invalid token" collapses
        # a handful of very different root causes (wrong/missing
        # GOOGLE_CLIENT_ID, expired token, clock skew, wrong token type) into
        # one message that's impossible to debug blind.
        logger.warning("Google ID token verification failed: %s", exc)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Google token")

    db = get_db()
    existing = await db[USERS].find_one({"google_id": profile.google_id})

    if existing is None:
        user = User(
            google_id=profile.google_id,
            email=profile.email,
            name=profile.name,
            picture=profile.picture,
        )
        await db[USERS].insert_one(user.to_mongo())

        # First login: create the default workspace.
        workspace = Workspace(user_id=user.id, name="Workspace1")
        await db[WORKSPACES].insert_one(workspace.to_mongo())
    else:
        user = User.from_mongo(existing)

    token = create_access_token(user.id, user.email)
    _set_auth_cookie(response, token)
    return UserOut(id=user.id, email=user.email, name=user.name, picture=user.picture)


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, email=user.email, name=user.name, picture=user.picture)


@router.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie(ACCESS_TOKEN_COOKIE_NAME, path="/")
    return {"ok": True}
