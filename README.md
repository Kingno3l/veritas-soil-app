# DEEP LEARNING-ENABLED INTERNET OF THINGS FOR INTELLIGENT SOIL HEALTH MONITORING IN PRECISION AGRICULTURAL SYSTEMS API (Backend)

**Project Code:** Victoria Uzuegbu Veritas MSc Final Year Project
**Tech Stack:** Python, FastAPI, TensorFlow, NumPy

This is the intelligence core of the Soil Health Monitoring System. It hosts the Convolutional Neural Network (CNN) and provides a REST API to process simulated IoT sensor data and predict biological soil markers.

---

## 📂 Features

* **Deep Learning Inference:** Loads a pre-trained CNN (`soil_soc_cnn.keras`) to predict Soil Organic Carbon (SOC), POX-C, and Enzyme activities.
* **Physics-Constraint Layer:** Implements logic to "clamp" predictions (e.g., Bulk Density) to scientifically valid ranges to prevent hallucinations.
* **FastAPI Architecture:** High-performance, asynchronous API handling.
* **IoT Data Simulation:** Accepts raw physical parameters (Temp, Moisture, pH, EC) mimicking ESP32 sensor nodes.

---

## 🚀 Setup & Installation

Follow these steps to run the backend server locally.

### 1. Prerequisites
* Python 3.9, 3.10, or 3.11 (Recommended for TensorFlow compatibility).

### 2. Navigate to the Directory
Ensure you are in the `backend` folder:
```bash
cd backend

```

### 3. Create a Virtual Environment

It is best practice to isolate dependencies.

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

### 4. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## ⚡ Running the Server

Start the local development server using Uvicorn:

```bash
uvicorn main:app --reload

```

* **Success Message:** You should see `Uvicorn running on http://127.0.0.1:8000`.
* **Auto-Reload:** The server will automatically restart if you save changes to the code.

---

## 📖 API Documentation

Once the server is running, you can access the interactive documentation:

* **Swagger UI:** [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)
* **ReDoc:** [http://127.0.0.1:8000/redoc](https://www.google.com/search?q=http://127.0.0.1:8000/redoc)

### Main Endpoint: `/predict`

* **Method:** `POST`
* **Input (JSON):**
```json
{
  "device_id": "FYP-NODE-LAB",
  "Soil_Temperature": 25.5,
  "Water_Content": 22.0,
  "pH": 6.8,
  "EC": 1.2
}

```


* **Output:** Returns predicted biological markers and health status.

---

## 🏗️ Project Structure

```text
backend/
├── models/
│   ├── cnn/soil_soc_cnn.keras  # The Trained Deep Learning Model
│   ├── scaler.pkl              # Scaler for normalizing inputs
│   └── target_columns.pkl      # List of output labels
├── main.py                     # The API Application Entry Point
├── train_cnn_model.py          # Script used to train the AI
├── prepare_data.py             # Data preprocessing logic
└── requirements.txt            # Python dependencies

```

---

## ⚠️ Troubleshooting

**1. "ModuleNotFoundError"**
Make sure you activated your virtual environment (Step 3) before running the install command.

**2. TensorFlow Version Errors**
If you encounter errors regarding TensorFlow versions on Render or locally, ensure you are using a compatible Python version (3.9 - 3.11). Python 3.13 is currently not fully supported by standard TensorFlow releases.
