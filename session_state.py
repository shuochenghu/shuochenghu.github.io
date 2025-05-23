import streamlit as st
import uuid

def get_user_id():
    """Get or create a user ID for the current session"""
    if 'user_id' not in st.session_state:
        # Generate a unique ID for this session
        st.session_state['user_id'] = str(uuid.uuid4())
    
    return st.session_state['user_id']