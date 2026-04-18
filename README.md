# 🤖 Aplicación de LLMs para Análisis de Sentimiento en IMDB

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
![Visitas](https://komarev.com/ghpvc/?username=SebastianDeghi&color=blue&style=flat)

### 👥 Colaboradores: **Emmanuel Gonzalez Gomez, Sebastián Deghi, Dalma Márquez.**

---

## 📝 Descripción General

Clasificación binaria de sentimientos (positivo/negativo) en 50,000 reseñas de películas IMDB, comparando cuatro enfoques arquitectónicos: **BiLSTM con atención**, **SLMs autoregresivos (TinyLlama, GPT-Neo)** en zero-shot, y **BERT** fine-tuneado. El objetivo es evaluar el equilibrio entre precisión, interpretabilidad, costo computacional y capacidad de generalización.

---

## 📁 Estructura del Proyecto

La organización de los archivos en este repositorio es la siguiente:

```
imdb-llm-sentiment-analysis/
├── IMDB_LLM_Sentiment_Analysis.ipynb    # Notebook principal con todos los modelos
├── requirements.txt                     # Dependencias de Python
├── LICENSE                              # Licencia MIT
├── .gitignore                           # Archivos ignorados
├── saved_models/                        # Modelos guardados (Word2Vec, BiLSTM, BERT)
│   ├── word2vec_trained.model
│   ├── best_model.pth
│   ├── resultados_tinyllama_1.1B.csv
│   ├── resultados_gptneo_1.3B.csv
│   └── bert_imdb_checkpoint/
└── README.md                            # Este archivo
```

---

## 📊 Descripción del Dataset

El dataset **IMDB Dataset of 50K Movie Reviews** contiene 50,000 reseñas de películas etiquetadas como positivas o negativas.

| Propiedad | Valor |
|-----------|-------|
| **Tamaño** | 50,000 reseñas |
| **Distribución** | 25,000 positivas / 25,000 negativas |
| **Columnas** | `review` (texto), `sentiment` (positive/negative) |
| **Tipo de problema** | Clasificación binaria de sentimientos |

**Fuente:** [IMDB Dataset en Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🚀 Modelos Implementados

### 1. BiLSTM + Atención + Word2Vec

Modelo secuencial bidireccional con mecanismo de atención entrenado desde cero sobre embeddings Word2Vec.

| Componente | Descripción |
|------------|-------------|
| **Embeddings** | Word2Vec entrenado sobre el corpus (100 dimensiones) |
| **Arquitectura** | BiLSTM (4 capas, hidden_dim=32) + Atención |
| **Regularización** | Dropout (0.3), Early Stopping |
| **Ventaja** | Alta interpretabilidad (pesos de atención visualizables) |

### 2. TinyLlama 1.1B (SLM autoregresivo)

Modelo pequeño tipo ChatGPT, utilizado en **zero-shot** mediante prompting.

| Propiedad | Valor |
|-----------|-------|
| **Parámetros** | 1.1B |
| **Arquitectura** | LLaMA 2-style (RoPE + SwiGLU + FlashAttention) |
| **Cuantización** | 4-bit (bitsandbytes) |
| **VRAM requerida** | ~4-6 GB |
| **Ventaja** | Eficiente, sin fine-tuning necesario |

### 3. GPT-Neo 1.3B (LLM ligero)

Modelo autoregresivo tipo GPT-3, utilizado en **zero-shot** con prompt engineering.

| Propiedad | Valor |
|-----------|-------|
| **Parámetros** | 1.3B |
| **Arquitectura** | Transformer Decoder-only (GPT-3 style) |
| **Dataset** | The Pile (825 GB) |
| **VRAM requerida** | ~6-8 GB |
| **Ventaja** | Mayor coherencia y creatividad |

### 4. BERT (DistilBERT fine-tuneado)

Modelo bidireccional encoder fine-tuneado específicamente para clasificación.

| Propiedad | Valor |
|-----------|-------|
| **Parámetros** | ~66M (DistilBERT) |
| **Arquitectura** | Transformer Encoder (bidireccional) |
| **Fine-tuning** | 20 épocas, learning rate 2e-5 |
| **Ventaja** | Máxima precisión en tareas discriminativas |

---

## 📈 Resultados Comparativos

| Modelo | Tipo | Enfoque | F1-score | Precisión | Recall |
|--------|------|---------|----------|-----------|--------|
| **BERT (DistilBERT)** | Encoder (fine-tuned) | Discriminativo | **~0.93** | ~0.93 | ~0.93 |
| **BiLSTM + Atención** | RNN + Atención | Discriminativo | ~0.88 | ~0.88 | ~0.88 |
| **GPT-Neo 1.3B** | Decoder (zero-shot) | Generativo | ~0.88 | ~0.87 | ~0.89 |
| **TinyLlama 1.1B** | Decoder (zero-shot) | Generativo | ~0.78 | ~0.77 | ~0.79 |

### Conclusión Clave

- **BERT fine-tuneado** logra el mejor rendimiento (F1 ~0.93), ideal para aplicaciones donde la precisión es prioritaria.
- **BiLSTM + Atención** ofrece el mejor equilibrio entre precisión e interpretabilidad, permitiendo visualizar qué palabras influyen en cada decisión.
- **GPT-Neo** alcanza rendimiento competitivo en zero-shot (F1 ~0.88), demostrando la potencia de los LLMs sin necesidad de entrenamiento específico.
- **TinyLlama**, aunque menos preciso (F1 ~0.78), es ideal para entornos con recursos limitados y aplicaciones rápidas.

---

## 🔧 Instalación y Uso

Sigue estos pasos para clonar el repositorio, configurar el entorno y ejecutar el análisis en tu máquina local.

### 1. Clonar el repositorio

```bash
git clone https://github.com/SebastianDeghi/imdb-llm-sentiment-analysis.git
cd imdb-llm-sentiment-analysis
```

### 2. Crear y activar un entorno virtual (recomendado)

- En **Windows**:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- En **macOS/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Nota:** Las dependencias incluyen `torch`, `transformers`, `datasets`, `accelerate`, `bitsandbytes`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`, `nltk`, `gensim`.

### 4. Ejecutar el notebook

```bash
jupyter notebook IMDB_LLM_Sentiment_Analysis.ipynb
```

### 5. (Opcional) Configuración de GPU

Para ejecutar TinyLlama o GPT-Neo, se recomienda una GPU con al menos 6 GB de VRAM. El notebook detectará automáticamente CUDA si está disponible.

---

## 📚 Referencias y Recursos

- **Artículo BiLSTM + Atención:** Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*.
- **Word2Vec:** Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space*.
- **TinyLlama:** TinyLlama Team (2024). *TinyLlama: An Open-Source Small Language Model*.
- **GPT-Neo:** EleutherAI (2021). *GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow*.
- **BERT:** Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- **Dataset:** [IMDB Dataset of 50K Movie Reviews en Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Para más detalles, consulta el archivo [`LICENSE`](LICENSE) en la raíz del repositorio.

---

## 🙏 Agradecimientos

- A **Lakshmi N Pathi** por publicar el dataset en Kaggle.
- A **Hugging Face** por la infraestructura de modelos y tokenizadores.
- A **EleutherAI** por GPT-Neo y a **TinyLlama Team** por TinyLlama.
- A la comunidad de código abierto por `pytorch`, `transformers`, `bitsandbytes`, `scikit-learn` y `nltk`.

---

## 👤 Autor

**Sebastián Deghi**
- GitHub: [@SebastianDeghi](https://github.com/SebastianDeghi)
- LinkedIn: [@sebastian-deghi](https://www.linkedin.com/in/sebastian-deghi/)
- Google Scholar: [@Sebastian E. Deghi](https://scholar.google.com/citations?user=3Nq5hTIAAAAJ&hl=en)