# Mr.🆖 SpeakAI 🗣️

<p style="text-align:center; font-size: 1.2em;">
  <strong>🪄 Turn text from documents and images into high-quality audio with one click. ✨ Convert articles, stories, and other content into clear speech for easy listening. 💡 Ideal for students to practice listening and speaking skills by turning any text into natural-sounding audio.</strong>
</p>

## Overview

SpeakAI is an AI-powered web application that converts text from various sources — including pasted text, uploaded documents (PDF, DOCX, TXT), images, or URLs — into high-quality audio speech using advanced text-to-speech technology. It leverages OpenAI's TTS API for natural-sounding voices and supports concurrent processing for fast audio generation. The app features a user-friendly Gradio-based web interface and includes OCR capabilities for extracting text from images via OpenAI's Vision API.

## Features

- **Multi-Input Sources:** Accept text from:
  - Direct text input (paste your content)
  - File uploads (PDF, DOCX, TXT documents)
  - Images (JPG, JPEG, PNG) with automatic text extraction using AI vision
  - URLs (web articles and pages)
- **High-Quality Audio:** Generates lifelike speech using OpenAI's text-to-speech models
- **Multiple Voices:** Choose from a variety of voices including English (female/male) and Cantonese options
- **Concurrent Processing:** Efficiently processes large texts by splitting into chunks and generating audio in parallel
- **Robust Text Handling:** Intelligently splits text at paragraph and sentence boundaries for natural pacing
- **User-Friendly Interface:** Built with Gradio for an intuitive web experience
- **API Integration:** Supports custom overlays for OpenAI and Vision API providers
- **Error Resilient:** Includes retries, logging with Sentry, and comprehensive error handling
- **Containerized Deployment:** Docker-ready for easy deployment

## Requirements

- **API Keys:**
  - OpenAI TTS API key (set as `TTS_API_KEY` environment variable)
  - OpenAI Vision API key (set as `VISION_API_KEY` environment variable, required for image processing)
- **Python:** 3.12 or higher
- **Dependencies:** Managed via `uv` package manager

## Installation

### Option 1: Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bentwnghk/speakAI.git
   cd speakAI
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   Create a `.env` file or export variables directly:
   ```bash
   export TTS_API_KEY="your-openai-tts-api-key"
   export VISION_API_KEY="your-openai-vision-api-key"
   ```

### Option 2: Docker Deployment

The project includes Docker support for easy containerized deployment.

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Or build manually:**
   ```bash
   docker build -t speakai .
   docker run -p 8000:8000 -e TTS_API_KEY="your-key" -e VISION_API_KEY="your-key" speakai
   ```

## Usage

1. **Start the application:**
   ```bash
   python main.py
   ```
   Or with Docker, the app will be available on port 8000.

2. **Access the web interface:**
   Open your browser and go to `http://localhost:8000`

3. **Generate audio:**
   - Choose your input method: Enter Text, Upload Files, or Provide a URL
   - For file uploads, select PDF, DOCX, TXT, or image files
   - Select your preferred voice from the dropdown
   - (Optional) Enter a Vision API key in Advanced Settings for image processing
   - Click "✨ Generate Audio" to start the process

4. **Download your audio:**
   The generated MP3 file will be available for download, along with the extracted text transcript

## Supported File Types

- **Documents:** PDF (.pdf), Word (.docx), Text (.txt)
- **Images:** JPG (.jpg, .jpeg), PNG (.png)
- **Web Content:** Any valid HTTP/HTTPS URL

## Configuration

Customize the application by setting these environment variables:

- `TTS_API_KEY`: Your OpenAI API key for text-to-speech
- `VISION_API_KEY`: Your OpenAI API key for vision/OCR
- `TTS_BASE_URL`: Custom base URL for TTS API (optional)
- `VISION_BASE_URL`: Custom base URL for Vision API (optional)
- `SENTRY_DSN`: Sentry DSN for error logging (optional)

## API Usage Details

- **Text Splitting:** Long texts are automatically split into 4000-character chunks
- **Voice Options:** Female 1 (nova), Male 1 (alloy), Female 2 (fable), Male 2 (echo), Cantonese Female (shimmer), Cantonese Male (onyx)
- **Cost Estimation:** Approximately $15 per million characters for TTS
- **Concurrent Workers:** Up to 10 parallel audio generations for optimal performance

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and enhancement requests.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more information.
