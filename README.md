# 📰 Fake News Detection System

## 🚀 Overview

This project is a **Fake News Detection System** that uses Machine Learning and Deep Learning techniques to classify news as **Real or Fake**.
It is built using **Python, NLP, and BiLSTM model**, and provides a simple web interface for users to check news authenticity.

---

## 🎯 Features

* 🔍 Detect whether news is **Real or Fake**
* 🧠 Uses **BiLSTM Deep Learning model**
* 📊 Text preprocessing using NLP techniques
* 🌐 Simple web interface using Flask
* 📝 History tracking of predictions

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** Flask
* **Libraries:** NumPy, Pandas, Scikit-learn, TensorFlow/Keras
* **NLP:** Tokenization, Padding, Label Encoding
* **Frontend:** HTML, CSS

---

## 📂 Project Structure

```
FakeNewsDetection/
│── app/                
│── templates/          
│── Notebook/           
│── models/             
│── data/               
│── app.py              
│── train_model.py      
│── .gitignore
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/adity7a/FakeNewsDetection.git
cd FakeNewsDetection
```

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the application

```
python app.py
```

---

## 📊 Model Details

* Model: **BiLSTM (Bidirectional LSTM)**
* Input: News text
* Output: Real / Fake classification
* Preprocessing:

  * Tokenization
  * Padding
  * Label Encoding

---

## ⚠️ Note

* Dataset and trained model files are **not included** due to large size.
* You can retrain the model using `train_model.py`.

---

## 🤝 Contributing

Feel free to fork this repository and contribute!

---

## 📧 Contact

**Aditya Kumar**
GitHub: https://github.com/adity7a
