# Maithili Nano Poet

A full-stack AI-powered application that generates poetry in the Maithili language. The project uses a dual-model approach, combining a custom-trained transformer and a fine-tuned Gemma 2B model to create culturally rich and contextually relevant poems.

The application features a modern web interface where users can input a word, receive two distinct poems, read cultural insights, and listen to an audio narration.

## ✨ Features

*   **Dual AI Models**: Leverages two distinct models for diverse poetic styles:
    *   **MaithiliNano**: A lightweight, custom-built transformer trained specifically on Maithili poetry.
    *   **Gemma 2B**: A fine-tuned version of Google's Gemma model, providing more complex and creative outputs.
*   **Smart Translation**: Automatically translates input words from any language into Maithili (Devanagari script) with fallbacks to culturally similar languages like Nepali and Hindi.
*   **Cultural Insights**: Provides contextual notes about the significance of the input word in Maithili culture, art, and literature.
*   **Text-to-Speech**: Generates an audio narration of the poem, supporting both Maithili and Hindi pronunciations.
*   **Modern UI**: A responsive and intuitive web interface built with Next.js and Tailwind CSS, featuring:
    *   Side-by-side comparison of poems from both models.
    *   Client-side history of generated poems stored in the browser.
    *   Options to copy, listen to, and select a preferred poem.

## 🛠️ Tech Stack

*   **Backend**: Python, FastAPI, PyTorch, Hugging Face Transformers, PEFT (LoRA), `deep-translator`, `gtts`.
*   **Frontend**: Next.js, React, TypeScript, Tailwind CSS.
*   **AI Models**:
    *   Custom `nn.Module` Transformer (MaithiliNano).
    *   `google/gemma-2-2b-it` with a LoRA adapter.

## 📂 Project Structure

```
maithali-poem/
├── backend/
│   ├── main.py         # FastAPI application logic
│   ├── model/          # Directory for AI model weights and tokenizers
│   │   ├── finetuned_final.pt
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   └── gemma_2b/   # LoRA adapter for Gemma 2B
│   └── requirements.txt
└── frontend/
    ├── app/            # Next.js App Router
    │   └── page.tsx    # Main UI component
    ├── package.json
    └── README.md
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   Node.js 18+ and npm/yarn/pnpm
*   A Hugging Face account and an access token (for downloading the Gemma model).

### 1. Backend Setup

The backend serves the FastAPI API that runs the models.

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install required Python packages
pip install -r requirements.txt

# 4. Set up Hugging Face authentication
# You need an HF token to download the Gemma-2B model.
# The app will look for an HF_TOKEN environment variable.
export HF_TOKEN="your_hugging_face_token_here"

# 5. Ensure model files are in place
# The custom MaithiliNano model and tokenizer should be in `backend/model/`.
# The Gemma LoRA adapter should be in `backend/model/gemma_2b/`.
# The application will download the base Gemma model on first run.
```

### 2. Frontend Setup

The frontend is a Next.js application.

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install Node.js dependencies
npm install
```

### 3. Running the Application

You need to run both the backend and frontend servers simultaneously.

1.  **Run the Backend Server:**
    In your terminal, from the `backend` directory:
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

2.  **Run the Frontend Server:**
    In a *new* terminal, from the `frontend` directory:
    ```bash
    npm run dev
    ```
    Open http://localhost:3000 in your browser to use the application.

## 📝 API Endpoints

*   `POST /generate`: The main endpoint for generating poems.
    *   **Body**: `{ "word": "your_word_here", "models": ["nano", "gemma"] }`
    *   **Returns**: A JSON object with the original input, translated input, poems from both models, a cultural note, and an audio URL.
*   `GET /health`: A health check endpoint to verify that the API is running and which models are loaded.






uvicorn main:app --reload

 python3 -m venv venv
 source venv/bin/activate

 pip install -r requirements.txt
    