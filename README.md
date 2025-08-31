# 🌿 Leaf Classification - Precision Agriculture  

This project is a **Leaf Classification System** built using **TensorFlow** and **Streamlit**.  
It predicts the species of a plant based on an uploaded leaf image.  

The model is trained on the [Leaf Images Dataset](https://huggingface.co/datasets/yusuf802/leaf-images) from Hugging Face.  
This dataset contains labeled images of plant leaves, useful for **plant disease detection, crop monitoring, and precision agriculture applications**.  

The StreamLit webpafe is live @ [Leaf Species Classification](https://leaf-classification.streamlit.app/)
---

## 🚀 Features
- Upload a leaf image through a **Streamlit Web App**  
- Pre-trained CNN model (**MobileNetV2**) for classification  
- Displays **predicted plant species** with **confidence score**  
- Clean and simple **web interface**  

---

## 📂 Dataset
We used the **[yusuf802/leaf-images](https://huggingface.co/datasets/yusuf802/leaf-images)** dataset from Hugging Face.  

- **Train set**: 56,842 images  
- **Test set**: 10,032 images  
- Each image is labeled with its corresponding plant species.  

This dataset is primarily used for:  
- Leaf classification  
- Precision agriculture  
- Plant disease research  
- Agricultural AI/ML applications  

---

## 🖥️ How to Run  

1. Clone this repository:
   ```bash
   git clone https://github.com/rakesh001n/Leaf-Classification---precision-Agriculture.git
   cd Leaf-Classification---precision-Agriculture
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Example Prediction
- Input: Uploaded image of a **Tomato leaf**  
- Output:  
  ```
  Predicted Class: Tomato
  Confidence: 92.5%
  ```

---

## 🔗 Links
- 📂 Dataset: [yusuf802/leaf-images](https://huggingface.co/datasets/yusuf802/leaf-images)  
- 💻 GitHub Repo: [Leaf Classification - Precision Agriculture](https://github.com/rakesh001n/Leaf-Classification---precision-Agriculture)  
- 📒 Kaggle Notebook: [leaf classification-precision agriculture](https://www.kaggle.com/code/rockybhai001n/leaf-classification-precision-agriculture/)
---

## ✨ Future Improvements
- Add **disease classification** along with species detection  
- Deploy on **Hugging Face Spaces** or **Streamlit Cloud**  
- Expand dataset for better generalization  

---
## 👨‍💻 Developed By
- Rakesh [LinkedIn](https://www.linkedin.com/in/rakesh-sd/)
- Nishanth [LinkedIn](https://www.linkedin.com/in/nishanth-krishnamoorthy24/)
- Josika [LinkedIn](https://www.linkedin.com/in/josika-ponnusamy-a71b86294/)
---

## 📜 License
This project is licensed under the **MIT License**.  
