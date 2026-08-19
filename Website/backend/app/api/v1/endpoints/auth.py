from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import verify_password, create_access_token, create_refresh_token
from app.schemas.user import Token
from datetime import timedelta

router = APIRouter()

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    # Dummy implementation for scaffolding
    if form_data.username == "test@test.com" and form_data.password == "test":
        access_token = create_access_token(data={"sub": form_data.username})
        refresh_token = create_refresh_token(data={"sub": form_data.username})
        return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
