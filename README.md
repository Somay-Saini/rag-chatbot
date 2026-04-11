# 🤖 RAG Knowledge Base Chatbot

Upload your company PDFs → employees and customers get instant, accurate answers from your own documents. No hallucination. Sources cited.

**Built by [Somay Saini](https://github.com/Somay-Saini) | B.Tech AI & Data Science | IIT Jodhpur**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-blue?style=flat-square)

---

## 🎯 Problem It Solves

HR teams answer the same policy questions daily. Customer support teams repeat the same product answers. New employees waste hours searching through documents.

This chatbot reads your PDFs once and answers any question from them → instantly, accurately, 24/7.

---

## ✨ Features

- 📄 Upload multiple PDFs at once
- 🧠 RAG architecture → answers **only** from your documents
- 💬 Conversational memory → remembers chat history
- 🎨 Clean Streamlit web interface → no coding needed to use
- 📍 Cites the source document and page for every answer
- 🔒 Runs locally → your documents never leave your machine

---

## 🚀 Quick Start

```bash
git clone https://github.com/Somay-Saini/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
streamlit run rag_chatbot.py
```

Open browser → Upload your PDFs → Start asking questions.

---

## 💬 Example

```
User: What is the annual leave policy?

Bot:  According to the HR Policy document (Page 12),
      employees are entitled to 18 days of annual leave
      per year, with up to 5 days carry-forward allowed.
      📄 Source: HR_Policy_2024.pdf → Page 12
```

---

## 🔧 Use Cases

- **HR departments** → instant policy Q&A for employees
- **Customer support** → product manual chatbot
- **Onboarding** → new employee guide bot
- **Legal teams** → contract and compliance Q&A

---

## 📁 Project Structure

```
rag-chatbot/
├── rag_chatbot.py
├── requirements.txt
└── README.md
```

---

*💡 Need a custom chatbot for your documents? [Connect on LinkedIn](https://www.linkedin.com/in/somay-saini-3907b9390)*
