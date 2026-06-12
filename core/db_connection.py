import sqlite3
import os

# Đường dẫn CSDL SQLite (thay SQL Server bằng SQLite để không cần cài đặt thêm)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, 'database')
DB_PATH = os.path.join(DB_DIR, 'fish_cbir.db')


def get_connection():
    """Thiết lập và trả về đối tượng kết nối đến SQLite."""
    os.makedirs(DB_DIR, exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except sqlite3.Error as e:
        print(f"Lỗi kết nối CSDL: {e}")
        return None


def init_db():
    """Khởi tạo bảng Fish_Metadata nếu chưa tồn tại."""
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Fish_Metadata (
                    ID            INTEGER PRIMARY KEY AUTOINCREMENT,
                    Image_ID      TEXT NOT NULL UNIQUE,
                    Species_Label TEXT NOT NULL,
                    File_Path     TEXT NOT NULL
                )
            """)
            conn.commit()
            print("CSDL san sang.")
        except sqlite3.Error as e:
            print(f"Lỗi khi khởi tạo CSDL: {e}")
        finally:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    init_db()
    print(f"CSDL tại: {DB_PATH}")
