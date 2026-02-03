
# Sentiment Analysis of Real-time Flipkart Product Reviews

## 📌 Project Overview
This project performs **sentiment analysis** on real-time Flipkart product reviews to classify customer feedback as **Positive** or **Negative**. The goal is to understand customer satisfaction and identify pain points from negative reviews using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The model is deployed as a **Streamlit web application** for real-time sentiment prediction and can be hosted on an **AWS EC2 instance**.

---

## 🎯 Objective
- Classify customer reviews into **positive** or **negative**
- Analyze customer dissatisfaction from negative reviews
- Build an end-to-end **ML + NLP + Deployment** pipeline
- Enable real-time inference through a web application

---

## 📊 Dataset
- **Source:** Flipkart (scraped by Data Engineering team)
- **Product:** YONEX MAVIS 350 Nylon Shuttle
- **Total Reviews:** 8,518
- **Key Features:**
  - Reviewer Name  
  - Rating  
  - Review Title  
  - Review Text  
  - Review Date  
  - Location  
  - Up Votes / Down Votes  

> ⚠️ Note: Web scraping was not performed in this project. The provided dataset was used directly.

---

## 🧹 Data Preprocessing
- Removal of special characters and punctuation
- Stopword removal
- Text normalization using **lemmatization**
- Handling missing values
- Sentiment labeling based on rating:
  - Rating ≥ 4 → Positive  
  - Rating < 4 → Negative  

---

## 🧠 Feature Engineering
- **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization
- Vocabulary size limited to top 5,000 features for efficiency

---

## 🤖 Model Building
- **Algorithm:** Logistic Regression
- **Evaluation Metric:** F1-Score
- **Why F1-Score?**
  - Handles class imbalance effectively
  - Balances precision and recall

---

## 🌐 Model Deployment
- **Framework:** Streamlit
- **Functionality:**
  - User inputs a review
  - Model predicts sentiment in real time
- **Deployment Platform:** AWS EC2

---

## 🗂 Project Structure
```

flipkart-sentiment-analysis/
│
├── data/
│   └── data.csv
│
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── preprocess.py
├── train.py
├── app.py
├── requirements.txt
└── README.md

````

---

## ⚙️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Train the Model

```bash
python train.py
```

### 3️⃣ Run the Web App

```bash
streamlit run app.py
```

---

## ☁️ AWS EC2 Deployment (Summary)

```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Access the app via:

```
http://<EC2-PUBLIC-IP>:8501
```

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK
* Streamlit
* AWS EC2

---

## 📈 Key Learnings

* End-to-end NLP pipeline design
* Text vectorization using TF-IDF
* Model evaluation using F1-score
* Real-time ML model deployment
* Production-ready ML project structure

---

## 🚀 Future Enhancements

* BERT-based sentiment classification
* Aspect-based sentiment analysis
* Negative review pain-point extraction
* REST API using Flask/FastAPI
* Model monitoring and logging

---

## 👤 Author

**Kayala Durga Nagendra**
Aspiring Data Scientist | Machine Learning | NLP | Deployment

---

⭐ If you found this project useful, feel free to star the repository!


Just say the word 👌
```
