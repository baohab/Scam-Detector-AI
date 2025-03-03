import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Sample dataset for scam detection
data = [
    ("Your bank account is compromised! Urgent action required.", "scam"),
    ("Congratulations! You've won a free trip. Claim now!", "scam"),
    ("This is your last chance to claim your lottery prize. Transfer $500 now.", "scam"),
    ("Hello, your recent payment to PayPal was unsuccessful. Verify now.", "scam"),
    ("Hi Mom, I lost my wallet. Can you send me money urgently?", "scam"),
    ("Reminder: Your electricity bill is due. Pay before 5th March to avoid penalties.", "legitimate"),
    ("Amazon Order Confirmation: Your order has been shipped and will arrive in 3 days.", "legitimate"),
    ("Meeting reminder: Your Zoom link for today's meeting at 3 PM is here.", "legitimate"),
    ("Your internet package expires soon. Renew now for continued service.", "legitimate"),
    ("Your recent transaction of $500 was successful. Contact support if unauthorized.", "legitimate")
]

# Convert dataset into DataFrame
df = pd.DataFrame(data, columns=["message", "label"])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.3, random_state=42)

# Create a text classification model
model_pipeline = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model_pipeline.fit(X_train, y_train)

# Function to classify messages
def detect_scam(message):
    prediction = model_pipeline.predict([message])[0]
    return f"🚨 Scam Alert! This message looks like a SCAM." if prediction == "scam" else "✅ This message appears LEGITIMATE."

# Create Gradio interface
scam_detector_ui = gr.Interface(
    fn=detect_scam,
    inputs=gr.Textbox(label="Enter a message for scam detection:"),
    outputs=gr.Textbox(label="Detection Result"),
    title="AI Scam Detector",
    description="Enter a message, and the AI will determine if it is a scam or legitimate."
)

# Launch the app
scam_detector_ui.launch()
