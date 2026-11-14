"""建立新帳號（本機管理用，不需收驗證信）。

用法（在 backend 資料夾）：
    .venv\\Scripts\\python.exe create_user.py            # 互動輸入，預設建立管理員
    .venv\\Scripts\\python.exe create_user.py --officer  # 建立一般帳號（不能開始/停止監測）
密碼不會顯示在畫面上；需至少 12 碼、不可全為英文或數字、不可包含帳號或信箱前綴。
"""
import getpass
import re
import sys

from app.core.security import get_password_hash, validate_password_strength, verify_password
from app.database.session import Base, SessionLocal, engine
from app.models.user import User

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def ask(prompt: str, check, error: str) -> str:
    while True:
        value = input(prompt).strip()
        if check(value):
            return value
        print(f"✗ {error}")


def main() -> int:
    is_admin = "--officer" not in sys.argv[1:]
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        username = ask(
            "帳號：",
            lambda v: 3 <= len(v) <= 50 and db.query(User).filter(User.username == v).first() is None,
            "帳號需 3–50 字，且不可與既有帳號重複",
        )
        email = ask(
            "Email：",
            lambda v: bool(EMAIL_RE.match(v)) and db.query(User).filter(User.email == v).first() is None,
            "Email 格式錯誤，或已被使用",
        )
        while True:
            pw = getpass.getpass("密碼：")
            try:
                validate_password_strength(pw, username, email)
            except ValueError as e:
                print(f"✗ {e}，請重新輸入")
                continue
            if getpass.getpass("再輸入一次：") != pw:
                print("✗ 兩次輸入不同，請重新輸入")
                continue
            break

        user = User(
            username=username,
            email=email,
            hashed_password=get_password_hash(pw),
            is_admin=is_admin,
            is_active=True,
            email_verified=True,
            failed_login_count=0,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        assert verify_password(pw, user.hashed_password)
        role = "管理員" if is_admin else "一般帳號"
        print(f"✓ 已建立{role} {user.username}（{user.email}），可以直接登入")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
