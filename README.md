# Mr.🆖 SpeakAI 🗣️

<p style="text-align:center; font-size: 1.2em;">
  <strong>🪄 Turn text from documents and images into high-quality audio with one click. ✨ Convert articles, stories, and other content into clear speech for easy listening. 💡 Ideal for students to practice listening and speaking skills by turning any text into natural-sounding audio.</strong>
</p>

## Overview

SpeakAI is an AI-powered web application that converts text from various sources — including pasted text, uploaded documents (PDF, DOCX, TXT), or images — into high-quality audio speech using advanced text-to-speech technology. It leverages OpenAI's TTS API for natural-sounding voices and supports concurrent processing for fast audio generation. The app features a user-friendly Gradio-based web interface and includes OCR capabilities for extracting text from images via OpenAI's Vision API.

## Features

- **Multi-Input Sources:** Accept text from:
  - Direct text input (paste your content)
  - File uploads (PDF, DOCX, TXT documents)
  - Images (JPG, JPEG, PNG) with automatic text extraction using AI vision
- **High-Quality Audio:** Generates lifelike speech using OpenAI's text-to-speech models
- **Multiple Voices:** Choose from 6 voice options: Female 1 (nova), Male 1 (alloy), Female 2 (fable), Male 2 (echo), Female 3 (shimmer), Male 3 (onyx)
- **Speed Control:** Adjustable playback speed from 50% (slower) to 200% (faster) for customized listening experience
- **Concurrent Processing:** Efficiently processes large texts by splitting into chunks and generating audio in parallel
- **Audio History & Archives:** Browser-based storage system using IndexedDB to save, load, rename, and delete previously generated audio files
- **Cost Estimation:** Real-time calculation and display of TTS API costs (approximately $15 per million characters)
- **Robust Text Handling:** Intelligently splits text at paragraph and sentence boundaries for natural pacing
- **User-Friendly Interface:** Built with Gradio for an intuitive web experience with responsive design
- **API Integration:** Supports custom overlays for OpenAI and Vision API providers, plus Mr.🆖 AI Hub integration
- **Persistent API Keys:** Browser localStorage stores your API keys for convenience across sessions
- **Error Resilient:** Includes retries, logging with Sentry, and comprehensive error handling with user-friendly messages
- **Containerized Deployment:** Docker-ready with proper volume management for persistent data storage

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
    - Choose your input method: Enter Text, or Upload Files
    - For file uploads, select PDF, DOCX, TXT, or image files
    - Select your preferred voice from the dropdown (6 options available)
    - Adjust playback speed using the slider (50% to 200%)
    - (Optional) Enter your Mr.🆖 AI Hub API Key in Advanced Settings for enhanced functionality
    - Click "✨ Generate Audio" to start the process

4. **View results and manage audio:**
    - The generated MP3 file will be available for immediate playback and download
    - View the extracted text transcript with copy functionality
    - Estimated costs are displayed in real-time
    - Audio is automatically saved to browser history for future access

5. **Manage audio archives:**
    - Expand the "📜 Archives" section to view previously generated audio
    - Load any saved audio back into the player
    - Rename audio files in your history
    - Delete unwanted audio from your browser storage
    - Audio files are stored locally in your browser for offline access

## Supported File Types

- **Documents:** PDF (.pdf), Word (.docx), Text (.txt)
- **Images:** JPG (.jpg, .jpeg), PNG (.png)

## Configuration

Customize the application by setting these environment variables:

- `TTS_API_KEY`: Your OpenAI API key for text-to-speech
- `VISION_API_KEY`: Your OpenAI API key for vision/OCR
- `TTS_BASE_URL`: Custom base URL for TTS API (optional)
- `VISION_BASE_URL`: Custom base URL for Vision API (optional)
- `SENTRY_DSN`: Sentry DSN for error logging (optional)

## Browser Storage & Archives

The application includes a sophisticated browser-based storage system:

- **Local Storage:** Audio files are stored in your browser's IndexedDB for offline access
- **Automatic Saving:** Generated audio is automatically saved to your browser history
- **Archive Management:** Load, rename, or delete previously generated audio files
- **Persistent API Keys:** Your Mr.🆖 AI Hub API key is saved locally for convenience
- **Storage Cleanup:** Temporary audio files are automatically removed after 7 days
- **Cross-Session Access:** Access your audio history across browser sessions

**Note:** Audio files are stored locally in your browser only. Clearing browser data will remove all archived audio.

## API Usage Details

- **Text Splitting:** Long texts are automatically split into 4000-character chunks with intelligent paragraph and sentence boundary detection
- **Voice Options:** 6 high-quality voices available:
  - Female 1 (nova) - Default English female voice
  - Male 1 (alloy) - English male voice
  - Female 2 (fable) - Alternative English female voice
  - Male 2 (echo) - Alternative English male voice
  - Female 3 (shimmer) - Alternative English female voice
  - Male 3 (onyx) - Alternative English male voice
- **Speed Control:** Adjustable playback speed from 0.5x to 2.0x (50% to 200% in the interface)
- **Cost Estimation:** Real-time calculation at approximately $15 per million characters for TTS
- **Concurrent Workers:** Up to 10 parallel audio generations for optimal performance
- **Audio History:** Browser-based storage with IndexedDB for offline access and management
- **Error Handling:** Robust retry mechanism with exponential backoff for API failures

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and enhancement requests.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more information.
