# 🤖 AI Assistant - Local & Free

A professional AI assistant built with FastAPI and Ollama, demonstrating modern AI application development with local models.

**Built by:** Fatemeh Shahrokhshahi  
**Education:** Master of Computer Engineering, Istanbul Aydin University  
**Specialization:** Large Language Models and AI Reasoning

## ✨ Features

- 🏠 **100% Local AI** - Runs entirely on your computer
- 💰 **Completely Free** - No API costs or rate limits
- 🔒 **Privacy First** - Your data never leaves your machine  
- ⚡ **Fast Responses** - Optimized with TinyLLaMA model
- 📚 **Auto Documentation** - Interactive API docs with Swagger UI
- 🛠️ **Professional Code** - Clean, maintainable architecture
- 🐳 **Production Ready** - Proper virtual environment and dependency management

## 🚀 Live Demo

- **API Documentation:** http://127.0.0.1:8001/docs
- **Home Page:** http://127.0.0.1:8001/
- **System Status:** http://127.0.0.1:8001/test

## 🏗️ Tech Stack

- **Backend:** FastAPI (Modern Python web framework)
- **AI Engine:** Ollama (Local AI model runner)
- **Model:** TinyLLaMA (Fast, efficient language model)
- **Development:** Virtual Environment, Git version control
- **Documentation:** Automatic OpenAPI/Swagger generation

## 📋 Prerequisites

- Python 3.7+
- Git
- 4GB+ RAM (for AI model)
- Windows/Mac/Linux

## 🔧 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/my-ai-assistant.git
cd my-ai-assistant
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate.bat
# Mac/Linux:
source venv/bin/activate

# Install dependencies  
pip install -r requirements.txt
```

### 3. Install Ollama & AI Model
```bash
# Install Ollama from https://ollama.ai
# Then pull the AI model:
ollama pull tinyllama
```

### 4. Start the Application
```bash
# Start Ollama server (in one terminal)
ollama serve

# Start FastAPI server (in another terminal)
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

### 5. Test Your AI Assistant
- Open http://127.0.0.1:8001/docs
- Try the **POST /chat** endpoint
- Ask: "Hello! Tell me about artificial intelligence."

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome page with project info |
| `/chat` | POST | Chat with AI assistant |
| `/test` | GET | System status and health check |
| `/models` | GET | Available AI models |
| `/about` | GET | Creator and project information |
| `/docs` | GET | Interactive API documentation |

### Example Usage

**Chat with AI:**
```bash
curl -X POST "http://127.0.0.1:8001/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Explain machine learning in simple terms"}'
```

**Response:**
```json
{
  "response": "Machine learning is like teaching computers to learn patterns from examples, similar to how humans learn from experience...",
  "model_used": "ollama/tinyllama",
  "status": "success"
}
```

## 📁 Project Structure

```
my-ai-assistant/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
└── venv/              # Virtual environment (not committed)
```

## 🔄 Development Workflow

```bash
# Daily development cycle:

# 1. Activate environment
venv\Scripts\activate.bat

# 2. Start services
ollama serve                    # Terminal 1
uvicorn main:app --reload      # Terminal 2

# 3. Make changes to main.py

# 4. Test at http://127.0.0.1:8001/docs

# 5. Commit changes
git add .
git commit -m "Add new feature"
git push origin main
```

## 🧪 Testing

Test all endpoints:
```bash
# System health
curl http://127.0.0.1:8001/test

# AI chat
curl -X POST http://127.0.0.1:8001/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello AI!"}'
```

## 🎯 Learning Outcomes

This project demonstrates:

- ✅ **FastAPI Development** - Modern Python web APIs
- ✅ **AI Integration** - Local language model deployment  
- ✅ **API Design** - RESTful endpoints with proper HTTP methods
- ✅ **Documentation** - Auto-generated interactive docs
- ✅ **Virtual Environments** - Isolated dependency management
- ✅ **Git Workflow** - Version control best practices
- ✅ **Production Concepts** - Error handling, logging, structure

## 🚨 Troubleshooting

**Common Issues:**

| Issue | Solution |
|-------|----------|
| `uvicorn not found` | Activate virtual environment first |
| `Ollama connection error` | Run `ollama serve` in separate terminal |
| `AI responses timing out` | Try `ollama pull tinyllama` for faster model |
| `Port already in use` | Change port: `--port 8002` |

**Debug Commands:**
```bash
# Check Ollama status
ollama list

# Check if virtual environment is active
pip list

# View server logs
# (Check terminal where uvicorn is running)
```

## 🔮 Future Enhancements

- [ ] Web UI frontend (React/HTML)
- [ ] Conversation memory/history
- [ ] Multiple AI model support
- [ ] File upload and processing
- [ ] Chain of Logic reasoning implementation
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 About the Developer

**Fatemeh Shahrokhshahi**
- 🎓 Master of Computer Engineering Student at Istanbul Aydin University
- 🔬 Research Focus: Large Language Models, AI Reasoning, Domain Clustering
- 🏆 Published Research: "Chain of Logic" methodology (21.44% improvement over Chain of Thought)
- 💼 Aspiring LLM Developer and AI Engineer

**Research Projects:**
- Chain of Logic for LLM Reasoning Enhancement
- Balanced K-Means for Domain Discovery in Language Models  
- Earthquake Prediction Using Machine Learning

**Connect:**
- GitHub: [@fatemeshahrokhshahi](https://github.com/fatemeshahrokhshahi)
- LinkedIn: [FatemeShahrokhshahi](https://www.linkedin.com/in/fatemeh-shahrokhshahi-180435219/)

## 📈 Project Stats

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AI](https://img.shields.io/badge/AI-Ollama-orange?style=for-the-badge)

---

**⭐ If this project helped you learn FastAPI and AI development, please give it a star!**