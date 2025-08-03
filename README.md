# Quantum AI Biomarker Discovery & Treatment Recommendation System

This project integrates **Quantum AI, Hybrid Workflows, and Interactive Visualization** to recommend personalized treatments based on quantum biomarkers.

## 🚀 Features

- **Quantum & Classical Hybrid Computing**  
  - Utilizes IBM Quantum Cloud & Classical ML for optimal drug discovery.
- **Automated Job Scheduling**  
  - Efficient distribution of quantum/classical computations.
- **Real-Time Interactive Visualization**  
  - Web and CLI interfaces for monitoring patient data & treatments.
- **Fully Containerized**  
  - Deploy using **Docker**.

---

## 🏰️ Installation

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-repo/QuantumAI-Biomarker
cd QuantumAI-Biomarker
```

### 2️⃣ **Setup Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # MacOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🔥 Running the System

### **1️⃣ CLI Mode**
```bash
python interfaces/cli.py --recommend
```
👤 **User Input:**  
`0.8, 1.2, 0.5, 0.9, 1.1, 0.3, 0.7, 0.6`

✅ **Output:**  
`Recommended Treatment: Targeted Therapy`

---

### **2️⃣ Web Interface**
```bash
python interfaces/web_app/app.py
```
🖥 **Access Web App:**  
🔍 Open **`http://127.0.0.1:5000/`** in a browser.

---

## 🐳 Deploying with Docker

### **1️⃣ Build Docker Image**
```bash
docker build -t quantum-ai-biomarker .
```

### **2️⃣ Run Container**
```bash
docker run -p 5000:5000 quantum-ai-biomarker

if echo "QISKIT_IBM_TOKEN=your_ibm_quantum_token_here" > .env
docker run --env-file .env -p 5000:5000 quantum-ai-biomarker
```

🖥 **Access Web App:**  
🔍 Open **`http://127.0.0.1:5000/`**

---

## 🛠️ Development & Testing

### **Run PyTests**
```bash
pytest -v --disable-warnings tests/ai_tests.py
pytest -v --disable-warnings tests/quantum_tests.py
pytest -v --disable-warnings tests/treatment_recommendation_tests.py
pytest -v --disable-warnings tests/workflow_tests.py
```

---

## 💼 IBM Quantum Cloud Integration

**Setup IBM Quantum Account**  
1. Create an account at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Get your API **Token** from **My Account**
3. Set it in your environment:
```bash
export QISKIT_IBM_TOKEN="your_ibm_quantum_token"
```

---

## 🗂️ Project Structure

```
├── interfaces/           # User Interfaces (CLI & Web)
│   ├── cli.py           # CLI Interface
│   ├── web_app/
│   │   ├── app.py       # Flask Web App
│   │   ├── templates/   # HTML Templates
│   │   │   ├── index.html
│
├── workflow/             # Quantum & Job Scheduling
│   ├── workflow_manager.py
│   ├── job_scheduler.py
│
├── treatment_recommendation/
│   ├── recommendation_engine.py
│   ├── patient_data_integration.py
│
├── tests/                # Automated Tests
│   ├── workflow_tests.py
│   ├── treatment_recommendation_tests.py
│
├── models/               # Saved Quantum ML Models
├── data/                 # Patient Profiles & Biomarker Data
├── requirements.txt      # Python Dependencies
├── Dockerfile            # Containerization
├── README.md
```
# Personalized-Medicine-QML
