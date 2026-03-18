import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from groq import Groq

st.set_page_config(page_title="Gbekie Trap AI", layout="wide")
st.title("Gbekie Trap Demonstrator")
st.markdown("Transformer predicts Ramadan study performance → multi-agent Language Trap → auto-damping (Gbekie Condition) → stability")

# Data upload
uploaded_file = st.file_uploader("Upload ramadan_full_model.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded!")
else:
    st.info("Using sample data")
    df = pd.DataFrame({
        'Sleep_hours': np.random.normal(5.0, 1.2, 30),
        'Fasting_drag_F': np.linspace(1.0, 0.88, 30),
        'Prayer_penalty_P': np.random.normal(0.92, 0.04, 30),
        'Hydration_drag_H': np.linspace(1.0, 0.78, 30),
        'Study_performance': np.random.uniform(1.0, 5.5, 30)
    })

features = ['Sleep_hours', 'Fasting_drag_F', 'Prayer_penalty_P', 'Hydration_drag_H']
X = torch.tensor(df[features].values, dtype=torch.float32)
y = torch.tensor(df['Study_performance'].values, dtype=torch.float32).unsqueeze(1)

class GbekieTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(4, 1)
        self.attn_weights = None

    def forward(self, x):
        def hook(module, inp, out):
            self.attn_weights = module.self_attn.attn_output_weights.detach().mean(dim=1).cpu().numpy()
        h = self.encoder.layers[0].self_attn.register_forward_hook(hook)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        out = self.fc(x.squeeze(1))
        h.remove()
        return out

model = GbekieTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

if st.button("Train Transformer (10 epochs)"):
    with st.spinner("Training..."):
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
    st.success("Training complete!")

with torch.no_grad():
    preds = model(X).squeeze().numpy()

st.subheader("Predictions vs Actual")
fig, ax = plt.subplots()
ax.plot(df['Study_performance'], label="Actual", color="lime")
ax.plot(preds, label="Predicted", color="cyan", linestyle="--")
ax.legend()
st.pyplot(fig)

# Chatbot
st.subheader("Chat with Gbekie Trap AI")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are Gbekie Trap AI. Expert on Language Trap, Gbekie Condition (damping α), Ramadan model, Transformer predictions."}
    ]

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            stream = client.chat.completions.create(
                messages=st.session_state.messages,
                model="mixtral-8x7b-32768",
                temperature=0.7,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
