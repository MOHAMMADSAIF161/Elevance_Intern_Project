# 📝 Task 3 - Text Preprocessing using Hugging Face Transformers

## 📌 Project Overview

This project demonstrates how text descriptions are preprocessed using the **Hugging Face Transformers** library and then used as inputs for a **Stable Diffusion v1.5** text-to-image model.

The input text is first tokenized using the **CLIP Tokenizer**, encoded into high-dimensional **CLIP text embeddings**, and finally passed to the Stable Diffusion pipeline to generate images. This ensures that the text is represented accurately before image generation.

---

## 🎯 Objective

- Preprocess text descriptions using Hugging Face Transformers.
- Generate tokenized representations of text prompts.
- Encode text into CLIP embeddings.
- Save generated embeddings and token IDs.
- Visualize the embedding representation.
- Use the generated embeddings as inputs to Stable Diffusion v1.5.
- Generate images from the encoded text.

---

## 🚀 Features

- ✅ Text Tokenization using CLIP Tokenizer
- ✅ Text Encoding using CLIP Text Encoder
- ✅ Prompt Embedding Generation
- ✅ Embedding Visualization
- ✅ Stable Diffusion v1.5 Integration
- ✅ Image Generation from Embeddings
- ✅ Modular Python Code
- ✅ Easy to Extend

---

## 🛠 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Diffusers
- Stable Diffusion v1.5
- NumPy
- Matplotlib

---

## 📂 Project Structure

```
Task3_TextPreprocessing/
│
├── src/
│   ├── preprocess.py
│   ├── generate.py
│   ├── visualize.py
│   └── main.py
│
├── outputs/
│   ├── Generated_Images/
│   ├── Generated_Tokens/
│   ├── Generated_Embeddings/
│   └── embedding_heatmap.png
│
├── sample_prompts.txt
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Workflow

```
User Prompt
      │
      ▼
CLIP Tokenizer
      │
      ▼
Token IDs
      │
      ▼
CLIP Text Encoder
      │
      ▼
Prompt Embeddings
      │
      ├── Save Token IDs
      ├── Save Embeddings
      ├── Visualize Embeddings
      │
      ▼
Stable Diffusion v1.5
      │
      ▼
Generated Image
```

---

## 📋 Sample Prompts

```
A red rose in a garden.

A sunflower in a field.

A butterfly on a flower.

## 📊 Outputs

The project generates the following outputs:

### Generated Token IDs

- JSON format
- Saved inside `outputs/Generated_Tokens/`

### Prompt Embeddings

- NumPy (.npy) format
- Saved inside `outputs/Generated_Embeddings/`

### Generated Images

Images generated using Stable Diffusion v1.5 from the generated prompt embeddings.

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Elevance_Intern_Project.git
```

Go to the project

```bash
cd Task3_TextPreprocessing
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python src/main.py
```

---

## 📷 Results

The project successfully:

- Converted text prompts into token IDs.
- Generated CLIP text embeddings.
- Saved token IDs and embeddings.
- Visualized embeddings.
- Used prompt embeddings as inputs to Stable Diffusion v1.5.
- Generated images from text embeddings.

---

