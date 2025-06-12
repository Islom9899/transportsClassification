
# 🚗 Transport Classification App

This is a deep learning-based web application that classifies different types of transport vehicles (such as cars, airplanes, bicycles, etc.) from images. The project is built using the **fastai** framework for model training and **Streamlit** for the web interface.

You can try the app live here 👉 **[Live Demo](https://transportsclassification.streamlit.app)**

---

## 📌 Project Objective

The goal of this project is to demonstrate how deep learning models can be integrated into a real-time interactive web interface to solve image classification tasks. This app allows users to upload an image of a transport and receive the predicted class along with the confidence score.

---

## 🖼️ Features

- Upload an image of a transport vehicle.
- Real-time prediction using a trained fastai model.
- Visual confidence chart using Plotly.
- User-friendly interface via Streamlit.
- Deployed on Streamlit Cloud — no setup required to use.

---

## 🚀 Live Demo

🔗 **[Click to Open the App](https://transportsclassification.streamlit.app)**  
📸 Try uploading an image of a car, bike, or plane!

---

## 🧠 Model Info

- Framework: `fastai`
- Dataset: Custom dataset with labeled transport images
- Architecture: Pretrained CNN (e.g., ResNet34 or similar)
- Exported as: `transport_model.pkl`
- Trained locally and deployed to the cloud

---

## 🛠️ Tech Stack

| Tool        | Description                                  |
|-------------|----------------------------------------------|
| Python      | Core programming language                    |
| fastai      | For training and loading the deep learning model |
| Streamlit   | Web app framework                            |
| Plotly      | To visualize prediction probabilities        |
| Git & GitHub| Version control and open-source hosting      |

---

## 📁 Project Structure

```
transportsClassification/
│
├── app.py                  # Main Streamlit application
├── transport_model.pkl     # Trained model file
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

---

## 📦 Installation (Run Locally)

1. Clone the repository:

```bash
git clone https://github.com/Islom9899/transportsClassification.git
cd transportsClassification
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app locally:

```bash
streamlit run app.py
```

---

## 👤 Author

**Islom aka (Islom9899)**  
🎓 Computer Engineering graduate — Yeungnam University  
🎯 AI | Deep Learning | Web App Integration  
📫 [Add your email or LinkedIn here for portfolio]

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Thanks to the `fastai` and `Streamlit` communities.
- Inspired by real-world applications of AI in transport analytics.
