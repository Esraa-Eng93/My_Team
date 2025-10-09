# 🩺 CuraAI — Hackathon Project

## 🧠 Project Overview
**CuraAI** is a web-based student well-being solution developed during the **startAD x Google.org Hackathon**.  

Our platform leverages **Artificial Intelligence (AI)** to support student monitoring and intervention:  
- **Decision Trees** analyze structured student data such as attendance, academic performance, well-being scores, counseling sessions, and behavior incidents.  
- The AI generates **Recommended Actions** for school staff to take, ranging from routine monitoring to immediate referral for support.  

This system helps schools identify students at risk and take timely, data-driven actions to improve student well-being.


---

## 🎯 Objective
To create an interactive web platform that allows school staff to input or access student data,  
analyze it using AI models, and receive actionable recommendations to support student well-being.  

The system aims to:  
- Identify students at risk based on attendance, academic performance, behavior, and well-being scores.  
- Suggest targeted interventions, ranging from routine monitoring to urgent referral.  
- Help schools make **data-driven decisions** to improve student outcomes.


---

## 🛠️ Key Features
- ✅ Web interface for entering and visualizing patient data  
- ✅ AI-powered predictions using **Decision Tree** models  
- ✅ Natural Language Processing for Arabic medical text using **AraBERT**  
- ✅ Interactive dashboards and result visualization  
- ✅ Secure data handling  

---

## 🧩 Tech Stack
| Category | Technology |
|-----------|-------------|
| Frontend | HTML, CSS, Streamlit (Python-based Web Interface) |
| Backend | Python |
| Machine Learning | Scikit-learn (Decision Tree), HuggingFace Transformers (AraBERT) |
| Data | Synthetic data generated with Pandas & NumPy |
| Tools | Git, Google Drive, Google Colab |
---

## 🧪 How to Run the Project
1. Clone the repository:
   ```bash
   git clone git@github.com:Esraa-Eng93/My_Team.git

2. Navigate to the project folder:
      ```bash
   cd My_Team

3.Install dependencies (ensure Python 3.8+ is installed):
   ```bash
   pip install -r requirements.txt
   ```
4.Run the web application using Streamlit:
   ```bash
  streamlit run app.py
   ```
5. The browser will automatically open the website, or visit:
   ```bash
   http://localhost:8501
   ```
