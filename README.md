# Histology-classifier

## 🧠 Model Training: Histology Classifier Core

This section describes the training pipeline used to build the core models for the **Histology Classifier App**, leveraging transfer learning with MobileNetV2.

---

### 🔍 Overview

The classification model is built using **transfer learning** on the **MobileNetV2** architecture, pre-trained on ImageNet. We fine-tuned the last few layers and added custom dense layers for classifying 5 histology categories.

![ChatGPT Image Jul 1, 2025, 10_04_10 PM](https://github.com/user-attachments/assets/432d33dd-8598-436a-a17a-7459aeaa15e3)


---

### 📊 Dataset

The dataset consists of histology images structured into class-labeled directories. Images are automatically split into **training** and **validation** subsets using Keras' `ImageDataGenerator`.

---

### 🔁 Data Augmentation

We apply real-time data augmentation to increase generalization and robustness:

- Rotation (`±20°`)
- Horizontal shifts (`±10%`)
- Vertical shifts (`±10%`)
- Shearing and zooming
- Horizontal flipping
- Rescaling (`1./255`)

---

### 🏗️ Model Architecture

<p>Input Images (224x224x3)<br>
↓<br>
MobileNetV2 (pretrained, last 4 layers trainable)<br>
↓<br>
GlobalAveragePooling2D<br>
↓<br>
Dense(1024, ReLU)<br>
↓<br>
Dense(18, Softmax)</p>


---

### ⚙️ Training Configuration

- **Optimizer:** Adam (`lr=0.0001`)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Batch Size:** 32
- **Epochs:** 30
- **Early Stopping:** Monitors validation loss with patience of 3
- **Model Checkpoint:** Saves the best model based on validation loss

---

### 💾 Output Artifacts

- `best_model.h5`: Best performing model (Keras format)
- `model5.tflite`: Converted TensorFlow Lite model for deployment
- `labels.txt`: Class-label to index mapping

---

### 🧪 TensorFlow Lite Conversion

The final Keras model is converted into a `.tflite` format using TensorFlow Lite, enabling efficient mobile or embedded deployment.

![image](https://github.com/user-attachments/assets/e1860e04-1037-4271-b971-55f101e14d51)



---

### 📁 Directory Structure
```markdown /HistologyClassifier/ ├── train_model.py ├── best_model.h5 ├── model5.tflite ├── labels.txt └── README.md ```


---

> 💡 **Tip:** To retrain the model or adapt it to a new dataset, edit the data path and class count in the training script.
