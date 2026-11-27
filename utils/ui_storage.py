def jalankan_penyimpanan_sekali(session_state, action_key: str, save_callback):
    """Mencegah callback penyimpanan dijalankan dua kali pada rerun Streamlit."""
    saved_key = "saved_prediction_actions"
    saved_actions = session_state.setdefault(saved_key, set())
    if action_key in saved_actions:
        return {"status": "already_saved", "record_ids": []}
    try:
        record_ids = save_callback()
        saved_actions.add(action_key)
        session_state[saved_key] = saved_actions
        return {"status": "saved", "record_ids": record_ids if isinstance(record_ids, list) else [record_ids]}
    except Exception as exc:
        return {"status": "storage_error", "error": str(exc), "record_ids": []}
