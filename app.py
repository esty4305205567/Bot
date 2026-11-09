"""ממשק צ'אט בעברית (RTL) ב-Streamlit. הרצה: streamlit run app.py"""
import streamlit as st

import config as cfg
from rag import FamilyRAG

st.set_page_config(page_title="בוט ידע משפחתי לשמחות", page_icon="📖")

# כיוון ימין-לשמאל
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    .stMarkdown, [data-testid="stChatMessage"], [data-testid="stExpander"],
    [data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="טוען מודלים ומאגר...")
def load_rag() -> FamilyRAG:
    return FamilyRAG()


st.title("📖 בוט ידע משפחתי לשמחות")
st.caption("שאלות על תקציבים, ספקים, לקחים, מנהגים ולוחות זמנים. התשובות מבוססות על מסמכי המשפחה בלבד.")

try:
    rag = load_rag()
except Exception as e:
    error_str = str(e)
    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
        st.error("התקבלה שגיאת עומס משרתי השירות (מכסת הבקשות היומית הסתיימה). יש להמתין לאיפוס המכסה או לשדרג את מסלול ה-API.")
    else:
        st.error(error_str)
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

if question := st.chat_input("כתבו שאלה על השמחות..."):
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("מחפש במסמכים..."):
            try:
                ans = rag.ask(question)
                response_text = ans.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    response_text = "התקבלה שגיאת עומס משרתי השירות (מכסת הבקשות היומית הסתיימה). יש להמתין לאיפוס המכסה או לשדרג את מסלול ה-API."
                else:
                    response_text = f"שגיאה: {e}"
        st.markdown(response_text)
    st.session_state.history.append({"role": "assistant", "text": response_text})