"""重設使用者密碼（本機管理用）。

用法（在 backend 資料夾）：
    .venv\\Scripts\\python.exe reset_password.py ibibbii123
輸入的密碼不會顯示在畫面上；需符合系統規則：至少 12 碼、不可全為英文或數字、不可包含帳號或信箱前綴。
"""
import getpass
import sys

from app.core.security import get_password_hash, validate_password_strength, verify_password
from app.database.session import SessionLocal
from app.models.user import User


def main() -> int:
    if len(sys.argv) != 2:
        print("用法：python reset_password.py <帳號或信箱>")
        return 1
    ident = sys.argv[1]

    db = SessionLocal()
    try:
        user = db.query(User).filter((User.username == ident) | (User.email == ident)).first()
        if user is None:
            print(f"找不到使用者：{ident}")
            return 1

        while True:
            pw = getpass.getpass("新密碼：")
            try:
                validate_password_strength(pw, user.username, user.email)
            except ValueError as e:
                print(f"✗ {e}，請重新輸入")
                continue
            if getpass.getpass("再輸入一次：") != pw:
                print("✗ 兩次輸入不同，請重新輸入")
                continue
            break

        user.hashed_password = get_password_hash(pw)
        user.failed_login_count = 0
        db.commit()
        db.refresh(user)
        assert verify_password(pw, user.hashed_password)
        print(f"✓ 已重設 {user.username}（{user.email}）的密碼，可以登入了")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
