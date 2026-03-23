import streamlit as st
from inference import predict_review_comment

st.set_page_config(page_title="PeerReview Triage", page_icon="📝", layout="wide")

st.title("PeerReview Triage Assistant")
st.write("Paste a reviewer comment to predict its actionability, type and recommended response strategy.")

review_comment = st.text_area(
    "Reviewer Comment",
    height=90,
    placeholder="Paste a reviewer comment here..."
)

analyze = st.button("Analyze Comment")

if analyze:
    if not review_comment.strip():
        st.warning("Please enter a reviewer comment.")
    else:
        result = predict_review_comment(review_comment)

        st.subheader("Analysis Result")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Actionability", result["actionability"])
        col2.metric("Comment Type", result["comment_type"])
        col3.metric("Priority", result["priority"])
        col4.metric("Response Mode", result["response_mode"])

        st.write("**Guidance**")
        st.info(result["guidance"])

        st.write("**Confidence Scores**")
        st.write(f"Actionability score: {result['actionability_score']}")
        if result["comment_type_score"] is not None:
            st.write(f"Comment type score: {result['comment_type_score']}")
        else:
            st.write("Comment type score: Not applicable for non-actionable comments")

        st.write("**Original Comment**")
        st.code(result["review_comment"])