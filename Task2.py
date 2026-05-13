import streamlit as st
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class TicketAI:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            st.error("Model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        self.model = self._train_model()

    def _train_model(self):
        data = {
            'text': [
                "My password is not working for login", 
                "I need a refund for my last order", 
                "The website is down and showing 404 error", 
                "How do I change my profile picture", 
                "Payment failed but money was deducted",
                "The screen is cracked on delivery",
                "Where is my tracking number",
                "System is crashed and urgent fix needed",
                "Subscription renewal failed",
                "Cannot access my account dashboard"
            ],
            'label': [
                "Technical", "Billing", "Technical", "Account", "Billing", 
                "Hardware", "Billing", "Technical", "Billing", "Account"
            ]
        }
        df = pd.DataFrame(data)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', MultinomialNB())
        ])
        return pipeline.fit(df['text'], df['label'])

    def analyze(self, text):
        category = self.model.predict([text])[0]
        probs = self.model.predict_proba([text])[0]
        confidence = max(probs) * 100
        
        priority = "LOW"
        critical_keywords = ['urgent', 'down', 'crash', 'fail', 'hacked', 'money', 'deducted', 'emergency']
        
        if any(word in text.lower() for word in critical_keywords):
            priority = "HIGH"
        elif confidence < 65:
            priority = "MEDIUM"
            
        return category, priority, confidence

def main():
    st.set_page_config(page_title="Enterprise Support AI", page_icon="🎫", layout="wide")
    
    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151; }
        div[data-testid="stExpander"] { border: 1px solid #374151; border-radius: 12px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎫 Enterprise Support Classifier")
    st.subheader("AI-Driven Ticket Routing & Priority Assignment")
    st.write("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📝 Ticket Entry")
        description = st.text_area(
            "Describe the issue in detail:", 
            height=200, 
            placeholder="Example: My payment was processed twice and I need a refund immediately..."
        )
        process_btn = st.button("Analyze Ticket", use_container_width=True, type="primary")

    if process_btn and description:
        ai = TicketAI()
        category, priority, conf = ai.analyze(description)

        with col2:
            st.markdown("### 🔍 AI Analysis Results")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Category", category)
            with m2:
                st.metric("Priority", priority)
            with m3:
                st.metric("AI Confidence", f"{conf:.1f}%")

            st.write(f"**Confidence Level:**")
            st.progress(conf / 100)
            
            st.write("---")
            
            with st.expander("✅ System Action Recommendation", expanded=True):
                if priority == "HIGH":
                    st.error(f"🚨 CRITICAL: Routing to **{category}** Senior Team for immediate response.")
                elif priority == "MEDIUM":
                    st.warning(f"⚠️ ATTENTION: Assigned to **{category}** queue with elevated priority.")
                else:
                    st.success(f"✔️ ROUTED: Ticket sent to **{category}** department.")

    elif process_btn:
        st.warning("Please enter ticket details to begin the analysis.")

if __name__ == "__main__":
    main()