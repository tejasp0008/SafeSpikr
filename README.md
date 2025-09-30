# 🛡️ SafeSpikr

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16+-green.svg)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-success.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-blue.svg)](https://react.dev/)

**SafeSpikr** is an **AI-powered safety monitoring system** that combines machine learning, real-time dashboards, and hardware acceleration to detect and predict safety-related events.  
It brings together AI models, a React frontend, a Python/FastAPI backend, and FPGA/Verilog modules into one unified platform.

---

## ✨ Features

- ✅ AI-powered prediction using custom models  
- ✅ Real-time safety dashboard built with React  
- ✅ REST API backend powered by FastAPI  
- ✅ Verilog/FPGA integration for hardware acceleration  
- ✅ Cleanly structured for easy maintenance and future extensions.
- ✅ Modular architecture for easy extension  

---

## 📂 Project Structure

```
SafeSpikr/
├── frontend/                         # React frontend dashboard
├── backend/                          # Python FastAPI backend
├── model/                            # AI/ML models and training scripts
├── verilog/                          # FPGA / hardware logic (Vivado support)
├── Data/                             # Image dataset (ignored in git, .gitkeep used)
├── Data_/                            # Alternative dataset storage
├── data_unified/                     # Unified dataset (only .gitkeep committed)
├── ddd/                              # Additional dataset (only .gitkeep committed)
├── venv/                             # Python virtual environment (ignored in git)
├── user_identification_module        # For user identification and fetching weights
├── sleep_detection...                # To utilize AWS rekognition for sleep rekognition
├── .gitignore                        # Ignore unnecessary files
└── README.md                         # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/safespikr.git
cd safespikr
```

### 2. Setup Backend (Python + FastAPI)

```bash
python -m venv venv
# On Unix/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

**Run backend server:**

```bash
uvicorn main:app --reload
```

### 3. Setup Frontend (React)

```bash
cd frontend
npm install
npm start
```

Open in browser: [http://localhost:3000](http://localhost:3000)

---

## 📊 Data Handling

- `Data/imgs/` and `Data_/imgs/` are ignored in git, only `.gitkeep` is preserved.
- `data_unified/` and `ddd/` only commit `.gitkeep` files to keep folder structure.
- Large datasets should be stored locally or on cloud (not in git).
- Example ignored file: `frame.jpg`

---

## 🛠️ Development Notes

- **AI Models:** Developed in Python (PyTorch / TensorFlow)
- **Backend:** FastAPI serving trained models with REST APIs
- **Frontend:** React dashboard consuming APIs in real-time
- **FPGA/Verilog:** Acceleration & RTL simulations supported in Vivado

---

## 🚀 Roadmap / Future Enhancements

- 📹 Real-time video streaming integration
- 🤖 Improved AI models with larger & diverse datasets
- ⚡ FPGA acceleration for ultra-low latency inference
- 🐳 Dockerized deployment for production environments
- 🌐 Cloud-hosted demo with live safety monitoring

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch (`feature-xyz`)
3. Commit your changes
4. Push to your branch
5. Open a Pull Request 🚀

---

## 📜 License

This project is licensed under the MIT License.
