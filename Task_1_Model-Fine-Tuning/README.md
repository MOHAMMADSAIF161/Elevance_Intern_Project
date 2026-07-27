# 🌸 Task 1 - Flower Image Generation using LoRA Fine-Tuning

> Fine-tuning Stable Diffusion v1.5 using Low-Rank Adaptation (LoRA) on the Oxford 102 Flowers Dataset to generate high-quality flower images from text prompts.

---

# 📌 Project Overview

This project demonstrates parameter-efficient fine-tuning of the Stable Diffusion v1.5 model using **LoRA (Low-Rank Adaptation)**.

Instead of training the entire Stable Diffusion model, LoRA updates only a small number of trainable parameters, making the training process significantly faster while reducing GPU memory usage.

The model was trained on the **Oxford 102 Flowers Dataset** and can generate realistic flower images from natural language prompts.

---

# 🎯 Objectives

- Fine-tune Stable Diffusion v1.5 using LoRA.
- Learn parameter-efficient fine-tuning.
- Generate realistic flower images.
- Compare image quality before and after fine-tuning.
- Build an optimized image generation pipeline.

---

# 🚀 Features

- Stable Diffusion v1.5 Fine-Tuning
- LoRA-based Training
- Oxford 102 Flowers Dataset
- FP16 Mixed Precision Training
- Hugging Face Diffusers
- Accelerate Training
- PEFT Integration
- Custom Image Generation

---

# 🛠 Technologies Used

- Python
- PyTorch
- Hugging Face Diffusers
- Transformers
- Accelerate
- PEFT
- Stable Diffusion v1.5
- LoRA
- Google Colab
- NumPy
- Pillow
- Matplotlib

---

# 📂 Dataset

### Dataset Name

Oxford 102 Flowers Dataset

### Dataset Details

- 102 Flower Categories
- 8,189 Images
- High-quality flower photographs
- Used for image generation fine-tuning

---

# 📁 Project Structure

```text
Task_1_Model-Fine-Tuneing/

├── README.md
├── requirements.txt
├── train.py
├── inference.py
├── config.py
├── .gitignore

├── datasets/
│   ├── README.md
│   └── sample_images/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── image3.jpg

├── notebooks/
│   └── Flower_FineTuning.ipynb

├── outputs/
│   ├── generated_images/
│   └── comparison/

├── model/
│   ├── README.md
│   └── download_model.md

├── images/

└── docs/
```

---

# ⚙ Training Configuration

| Parameter | Value |
|------------|-------|
| Base Model | Stable Diffusion v1.5 |
| Fine-Tuning Method | LoRA |
| Dataset | Oxford 102 Flowers |
| Epochs | 1 |
| Max Training Steps | 1000 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Learning Rate | 1e-4 |
| Mixed Precision | FP16 |
| Framework | Hugging Face Diffusers |

---

# 📈 Training Progress

Training completed successfully.

Saved checkpoints:

- ✅ checkpoint-250
- ✅ checkpoint-500
- ✅ checkpoint-750
- ✅ checkpoint-1000

Final LoRA weights:

```
pytorch_lora_weights.safetensors
```

---

# 💻 Installation

### Clone the repository

```bash
git clone https://github.com/<your-github-username>/AI_Intern_Project.git
```

### Go to Task 1

```bash
cd AI_Intern_Project/Task_1_Model-Fine-Tuneing
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

Run the training script:

```bash
python train.py
```

---

# 🎨 Inference

Generate images using the fine-tuned model:

```bash
python inference.py
```

---

# 📝 Example Prompts

```
A beautiful sunflower in a garden

A red rose with green leaves

A white daisy blooming in nature

A colorful tulip field during spring

A realistic lotus flower floating on water
```

---

# 🖼 Results

## Before Fine-Tuning

*(Insert image here)*

Example:

```
outputs/comparison/before.png
```

---

## After Fine-Tuning

*(Insert image here)*

Example:

```
outputs/comparison/after.png
```

---

## Generated Images

Some generated examples:

- 🌻 Sunflower
- 🌹 Rose
- 🌷 Tulip
- 🌼 Daisy

Images are available inside:

```
outputs/generated_images/
```

---

# 📦 Model

The fine-tuned LoRA weights are not included in this repository due to GitHub file size limitations.

You can download them using the instructions inside:

```
model/download_model.md
```

---

# 📚 Future Improvements

- Train on more epochs.
- Support Stable Diffusion XL (SDXL).
- Improve prompt engineering.
- Train using larger flower datasets.
- Deploy using Gradio or Hugging Face Spaces.
- Add web interface for image generation.

---

# 📊 Learning Outcomes

Through this project, I learned:

- Stable Diffusion Architecture
- Parameter-Efficient Fine-Tuning (PEFT)
- LoRA Training
- Hugging Face Diffusers
- Accelerate Framework
- Image Generation Pipeline
- Dataset Preparation
- Model Inference

---

# 👨‍💻 Author

**Mohammad Saif**

- AI/ML Undergraduate
- AI/ML Intern @ ElevanceSkills
- Python Developer
- Generative AI Enthusiast

### Connect with Me

**GitHub:** https://github.com/MOHAMMADSAIF161

**LinkedIn:** *(Add your LinkedIn profile link)*

---

# ⭐ Acknowledgements

- Hugging Face
- Stability AI
- Oxford Visual Geometry Group (102 Flowers Dataset)
- Google Colab

---

# 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project useful, consider giving it a star on GitHub!