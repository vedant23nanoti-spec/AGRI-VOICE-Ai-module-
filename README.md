# 🌾 Agri Voice

**A voice-first AI assistant for farmers.** Ask a farming question in **Hindi, Marathi or English** and hear the answer spoken back. It can also check a crop leaf for disease and tell you today's weather.

> Final-year B.Tech Data Science project (Palloti Project 2).

---

## ✨ Features
- 🎤 **Voice input**: speak naturally, language is auto-detected
- 📚 **Crop advice**: answers come from real crop guides (RAG), not guesswork
- 🍃 **Leaf disease check**: upload a leaf photo, get the disease and confidence
- 🌦️ **Live weather**: free Open-Meteo API
- 🔊 **Spoken replies**: in the farmer's own language
- 💸 **100% free & open-source**

## 🧠 How it works
```
🎤 Voice → Whisper (speech-to-text) → Router
            ├─ Weather → Open-Meteo API
            └─ Advice  → ChromaDB search → LLM (Ollama) → Answer
                                                         ↓
                                                 gTTS → 🔊 Voice reply
```

## 🛠️ Tech stack
| Part | Tool |
|---|---|
| Speech-to-text | faster-whisper |
| Knowledge search | ChromaDB + multilingual-e5-small |
| LLM | Ollama (llama3) |
| Leaf disease | MobileNetV2 (Hugging Face) |
| Text-to-speech | gTTS |
| Weather | Open-Meteo |

## 🚀 Quick start
1. **Clone**
```bash
   git clone https://github.com/<your-username>/agri-voice.git
   cd agri-voice
```
2. **Install Ollama** from [ollama.com](https://ollama.com), then:
```bash
   ollama pull llama3
```
3. **Open** `agri_voice.ipynb` in VS Code (Python 3.10+).
4. **Run cells top to bottom.** Restart the kernel once after Cell 1.
5. *(Optional)* Put crop PDFs in `data/pdfs/` for better answers.
6. Run `ask_voice()` and speak, e.g. *"How to control pink bollworm in cotton?"*

## 📁 Project structure
```
agri-voice/
├── agri_voice.ipynb   # main notebook
├── data/pdfs/         # crop guides (optional)
├── leaf.jpg           # sample leaf for testing
├── .gitignore
└── README.md
```

## ⚠️ Troubleshooting
- **No mic detected:** use `ask_voice(audio_path="sample.wav")`
- **Ollama not running:** the app still works and shows raw knowledge-base text
- **First run is slow:** models download once (~1 GB)

## ⚡ Limitations
- Marathi speech recognition is good, not perfect (use `medium` Whisper for better results)
- Leaf model covers PlantVillage crops only
- Advice is informational. Consult your local KVK for serious cases.

## 🔮 Future work
- Mandi price lookup (Agmarknet)
- Source citations in answers
- Streamlit web app + Azure deployment
- WhatsApp / mobile interface

