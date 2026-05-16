"""POST /v1/auth/keys — issue an API key, gated by bootstrap token."""

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.api_key import generate_key
from backend.config import backend_settings
from backend.deps import get_db
from backend.models.auth import IssueKeyRequest, IssueKeyResponse
from backend.repos.api_keys import create_api_key
from backend.repos.users import get_or_create_user
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/keys", response_model=IssueKeyResponse, status_code=status.HTTP_201_CREATED)
async def issue_key(
    body: IssueKeyRequest,
    x_bootstrap_token: str | None = Header(default=None, alias="X-Bootstrap-Token"),
    db: AsyncSession = Depends(get_db),
) -> IssueKeyResponse:
    """Mint a new API key. Requires X-Bootstrap-Token matching BACKEND_BOOTSTRAP_TOKEN."""
    expected = backend_settings.bootstrap_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Key issuance disabled (BACKEND_BOOTSTRAP_TOKEN unset)",
        )
    if not x_bootstrap_token or x_bootstrap_token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid bootstrap token",
        )

    try:
        full_key, key_prefix, key_hash = generate_key()
        user = await get_or_create_user(db, email=body.email)
        api_key = await create_api_key(
            db,
            user_id=user.user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=body.name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Key issuance failed | email=%s | error=%s", body.email, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Key issuance failed: {type(exc).__name__}",
        )

    logger.info(
        "Issued API key | user_id=%s | key_id=%s | prefix=%s",
        user.user_id, api_key.key_id, key_prefix,
    )

    return IssueKeyResponse(
        key=full_key,
        key_id=api_key.key_id,
        key_prefix=key_prefix,
        user_id=user.user_id,
        email=user.email,
    )
