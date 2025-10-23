# pages/admin_dashboard.py
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from main import DB_FILE, hash_password, log_activity  # adjust import name if your main file isn't 'main.py'

def admin_dashboard():
    st.title("🛠️ Admin Dashboard")
    conn = sqlite3.connect(DB_FILE)

    st.subheader("👥 Registered Users")
    users_df = pd.read_sql_query("SELECT username, email, created_at, last_login, is_admin FROM users", conn)
    st.dataframe(users_df, use_container_width=True)

    csv_data = users_df.to_csv(index=False).encode()
    st.download_button("📥 Download User List (CSV)", csv_data, "users.csv", "text/csv")

    st.divider()
    st.subheader("❌ Delete a User")
    del_col1, del_col2 = st.columns([3,1])
    with del_col1:
        username_to_delete = st.text_input("Enter username to delete")
    with del_col2:
        if st.button("Delete", use_container_width=True):
            if not username_to_delete.strip():
                st.warning("Enter a username")
            else:
                c = conn.cursor()
                c.execute("DELETE FROM users WHERE username=?", (username_to_delete,))
                if c.rowcount == 0:
                    st.error(f"User '{username_to_delete}' not found.")
                else:
                    conn.commit()
                    st.success(f"User '{username_to_delete}' deleted.")
                    log_activity(st.session_state["user"], f"Deleted user '{username_to_delete}'")
                    st.rerun()

    st.divider()
    st.subheader("🧩 Promote / Demote User")
    username_role = st.text_input("Username to modify role")
    role_action = st.selectbox("Action", ["Promote to Admin", "Demote to Regular User"])
    if st.button("Apply Role Change"):
        new_status = 1 if role_action == "Promote to Admin" else 0
        c = conn.cursor()
        c.execute("UPDATE users SET is_admin=? WHERE username=?", (new_status, username_role))
        if c.rowcount == 0:
            st.error(f"User '{username_role}' not found.")
        else:
            conn.commit()
            st.success(f"{username_role} role updated.")
            log_activity(st.session_state["user"], f"Changed role for '{username_role}' to {'Admin' if new_status else 'User'}")
            st.rerun()

    st.divider()
    st.subheader("🔑 Reset User Password")
    user_to_reset = st.text_input("Enter username to reset password")
    new_pass = st.text_input("New password", type="password")
    if st.button("Reset Password (Admin)"):
        if not user_to_reset.strip() or not new_pass.strip():
            st.warning("Fill both fields")
        else:
            hashed = hash_password(new_pass)
            c = conn.cursor()
            c.execute("UPDATE users SET password=? WHERE username=?", (hashed, user_to_reset))
            if c.rowcount == 0:
                st.error(f"User '{user_to_reset}' not found.")
            else:
                conn.commit()
                st.success(f"Password for '{user_to_reset}' has been reset.")
                log_activity(st.session_state["user"], f"Reset password for '{user_to_reset}'")

    st.divider()
    st.subheader("📋 Activity Logs")
    logs_df = pd.read_sql_query("SELECT * FROM activity_log ORDER BY id DESC LIMIT 500", conn)
    if logs_df.empty:
        st.info("No activity logs yet.")
    else:
        st.dataframe(logs_df, use_container_width=True)
        csv_logs = logs_df.to_csv(index=False).encode()
        st.download_button("📥 Download Logs (CSV)", csv_logs, "activity_logs.csv", "text/csv")

    conn.close()

# Auto-run if accessed directly
if __name__ == "__main__":
    admin_dashboard()
