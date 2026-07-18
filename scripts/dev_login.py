"""Mint a working session without going through Google OAuth, for curl/Postman
testing of Phases 0-4 before the frontend (and its real Google Sign-In
button) exists.

Usage (from the Server/ directory, with shared/.env pointing at a real Mongo
URI):

    python -m scripts.dev_login

Creates (or reuses) a dev user + their default workspace, prints a JWT and
ready-to-use curl commands.
"""
import asyncio

from shared.auth import ACCESS_TOKEN_COOKIE_NAME, create_access_token
from shared.db import get_db
from shared.models.user import COLLECTION as USERS
from shared.models.user import User
from shared.models.workspace import COLLECTION as WORKSPACES
from shared.models.workspace import Workspace

DEV_GOOGLE_ID = "dev-local-user"
DEV_EMAIL = "dev@example.com"
DEV_NAME = "Dev User"


async def main():
    db = get_db()

    existing = await db[USERS].find_one({"google_id": DEV_GOOGLE_ID})
    if existing is None:
        user = User(google_id=DEV_GOOGLE_ID, email=DEV_EMAIL, name=DEV_NAME)
        await db[USERS].insert_one(user.to_mongo())
        workspace = Workspace(user_id=user.id, name="Workspace1")
        await db[WORKSPACES].insert_one(workspace.to_mongo())
        print(f"Created dev user {user.id} + workspace {workspace.id}")
    else:
        user = User.from_mongo(existing)
        workspace_doc = await db[WORKSPACES].find_one({"user_id": user.id})
        workspace = Workspace.from_mongo(workspace_doc)
        print(f"Reusing existing dev user {user.id} + workspace {workspace.id if workspace else '(none found)'}")

    token = create_access_token(user.id, user.email)

    print("\nJWT:")
    print(token)
    print(f"\nCookie header value: {ACCESS_TOKEN_COOKIE_NAME}={token}")
    print("\nExample curl (adjust host/port):")
    print(f'  curl -s http://localhost:8000/me --cookie "{ACCESS_TOKEN_COOKIE_NAME}={token}"')
    if workspace:
        print(
            f'  curl -s http://localhost:8000/workspaces/{workspace.id}/files '
            f'--cookie "{ACCESS_TOKEN_COOKIE_NAME}={token}"'
        )


if __name__ == "__main__":
    asyncio.run(main())
