# Breast_Cancer_Detection
🎯 Automated Breast Cancer Cell Detection using Deep Learning This capstone project leverages deep learning models to accurately classify breast cancer histopathological images. Trained on a publicly available dataset and deployed using Streamlit, the application offers a clean, user-friendly interface for real-time cancer prediction.

# 🧠 Breast Cancer Cell Classification using Deep Learning

A deep learning-based web application that detects breast cancer from histopathological images using a custom CNN and ResNet50 model. This is my capstone project at GITAM School of Technology, deployed via Streamlit.

## 🚀 Live Demo
👉 [Click here to try the deployed app](https://breast-cancer-classification-kbctfhvtuetb8wz6yz8fdz.streamlit.app/)

---

## 📌 Features

- 🔬 Classifies histopathological images as **Benign** or **Malignant**
- 🧠 Trained using both a **custom CNN** and **ResNet50 (transfer learning)**
- 🖼️ Accepts image uploads and displays predictions instantly
- 🌐 Hosted using **Streamlit** — serverless and fast
- 📊 Built with simplicity and accessibility in mind

---

## 📁 Project Structure

📁 Detect_BreastCancer.ipynb # Data loading, preprocessing, evaluation
📁 train_CustomModel_32_conv_20k.ipynb # CNN model training (32 conv layers, 20k steps)
📁 train_ResNet50_32_20k.ipynb # ResNet50 training notebook
📁 app.py # Streamlit app code (UI + model inference)
📁 requirements.txt # Dependencies for running the app


---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **OpenCV & NumPy**
- **Matplotlib, Seaborn (visualization)**
- **Streamlit** (for web UI and deployment)

---

## 🧪 Dataset

- **BreaKHis**: Breast Cancer Histopathological Image Classification Dataset
- Includes 7,909 images across 8 tumor subclasses under 4 magnification levels (40X, 100X, 200X, 400X)
- Classes: **Benign** and **Malignant**
- Source: [Kaggle - BreaKHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)

---

## 📈 Model Performance

| Model         | Accuracy | Notes                          |
|---------------|----------|---------------------------------|
| Custom CNN    | ~92%     | 32 Convolutional layers, trained for 20,000 steps |
| ResNet50      | ~95%     | Transfer learning model with fine-tuning |

---

## 👨‍🎓 Academic Context

- 🎓 **Capstone Project**  
- 🏫 **GITAM School of Technology**  
- 👨‍🏫 Guided by **Prof. Sangameshwar**  
- 📆 Final Year B.Tech (Computer Science) — Class of 2025  
- 👨‍💻 Developed by **Chowda Reddy N** and **N Swaroopa**

---

## 🧑‍💻 How to Run Locally

  1. **Clone the repository**
     ```bash
     git clone https://github.com/your-username/breast-cancer-classification.git
     cd breast-cancer-classification
     
  2. **Create a virtual environment (optional but recommended)**
    
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
  
  3. **Install dependencies**
    
    pip install -r requirements.txt
  
  4. **Run the Streamlit app**
   
    streamlit run app.py

📦 requirements.txt

streamlit  
tensorflow  
opencv-python  
numpy  
pillow  
matplotlib  

🤝 Acknowledgements  
Kaggle - BreaKHis Dataset  
GITAM School of Technology, Bangalore  
Prof. Sangameshwar, for academic guidance  

📜 License  
This project is open-source and available under the MIT License.  

🙋‍♂️ Connect With Me  
Chowda Reddy N  
Final Year B.Tech CSE | Full Stack Enthusiast | AI Explorer  
💬 Email: reddyreddy44477@gmail.com
