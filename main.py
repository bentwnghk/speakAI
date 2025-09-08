# -*- coding: utf-8 -*-
import concurrent.futures as cf
import glob
import io
import os
import time
import datetime
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal, Optional, Dict, Any
import json
import gradio as gr
import sentry_sdk
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger
from openai import OpenAI, APIError
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from mimetypes import guess_type
import docx
import pytz

# --- Configuration ---
# Define available voices. "nova" is set as the default.
VOICE_MAP = {
    "Female 1": "nova",
    "Male 1": "alloy",
    "Female 2": "fable",
    "Male 2": "echo",
    "Cantonese Female": "shimmer",
    "Cantonese Male": "onyx",
}
OPENAI_VOICES = list(VOICE_MAP.keys())

if sentry_dsn := os.getenv("SENTRY_DSN"):
    sentry_sdk.init(sentry_dsn)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Core Functions ---

def split_text(text: str, max_chunk_size: int = 4000) -> List[str]:
    """
    Splits the text into chunks safe for the TTS API, handling paragraphs, sentences, and words.
    This more robust version correctly handles oversized paragraphs and sentences by breaking them down
    by sentences, and then by words if a sentence itself is too long.
    """
    import re
    chunks = []
    
    # First, split by paragraphs to preserve structure
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        if len(paragraph) <= max_chunk_size:
            chunks.append(paragraph)
        else:
            # If a paragraph is too long, split it into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_chunk = ""
            for sentence in sentences:
                if len(sentence) > max_chunk_size:
                    # If a sentence is too long, split it into words
                    words = sentence.split(' ')
                    for word in words:
                        if len(current_chunk) + len(word) + 1 > max_chunk_size:
                            chunks.append(current_chunk)
                            current_chunk = word
                        else:
                            current_chunk += (' ' if current_chunk else '') + word
                    if current_chunk: # Add the last part of the oversized sentence
                         chunks.append(current_chunk)
                         current_chunk = ""
                elif len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += (' ' if current_chunk else '') + sentence
            
            if current_chunk: # Add the last remaining chunk
                chunks.append(current_chunk)

    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(APIError) # More specific retry for OpenAI errors
)
def get_mp3(text: str, voice: str, api_key: str) -> bytes:
    """
    Generates MP3 audio for a single text chunk using the OpenAI TTS API.
    This function now uses the TTS_BASE_URL and the selected voice.
    It includes robust error handling and retries to ensure reliability.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("TTS_BASE_URL")
    )
    
    logger.debug(f"Requesting TTS. Voice: '{voice}', Text: '{text[:50]}...'")
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3",
        )
        
        # The response content is the binary audio data
        audio_content = response.read()
        logger.debug(f"TTS generation successful for chunk. Size: {len(audio_content)} bytes.")
        return audio_content

    except APIError as api_err:
        logger.error(f"OpenAI API error during TTS generation. Voice: '{voice}'. Error: {api_err}")
        raise  # Reraise to trigger tenacity retry
    except Exception as e:
        logger.error(f"Unexpected error during TTS generation. Voice: '{voice}'. Error: {e}")
        raise

def is_pdf(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    return filename.lower().endswith(".pdf") or (t or "").endswith("pdf")

def is_image(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    image_exts = (".jpg", ".jpeg", ".png")
    return filename.lower().endswith(image_exts) or (t or "").startswith("image")

def is_text(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    return filename.lower().endswith(".txt") or (t or "") == "text/plain"

def is_docx(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return filename.lower().endswith(".docx") or (t or "") == docx_mime

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
def extract_text_from_image_via_vision(image_file, vision_api_key=None):
    """Extracts text from an image using OpenAI Vision API, with retries."""
    client = OpenAI(
        api_key=vision_api_key or os.getenv("VISION_API_KEY"),
        base_url=os.getenv("VISION_BASE_URL"),
        timeout=120.0
    )
    logger.debug(f"Requesting Vision text extraction for image: {image_file}")
    try:
        with open(image_file, "rb") as f:
            data = f.read()
            mime_type = guess_type(image_file)[0] or "image/png"
            b64 = base64.b64encode(data).decode("utf-8")
            image_url = f"data:{mime_type};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "auto"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all computer-readable text from the provided image.\n\n"
                            "Instructions:\n"
                            "- Preserve original paragraph breaks by inserting two newline characters (`\\n\\n`) between paragraphs.\n"
                            "- Remove any numbers in square brackets (e.g., `[1]`, `[13]`) or parentheses (e.g., `(1)`) that appear at the beginning of any paragraph.\n"
                            "- Do not insert any additional line breaks within paragraphs.\n"
                            "- Use visual spacing and indentation to detect paragraph breaks.\n"
                            "- Return only the extracted text, without commentary, metadata, or formatting.\n"
                            "- Output the result as a plain text string."
                        )
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=32768,
            temperature=0,
        )
        extracted_text = response.choices[0].message.content.strip()
        logger.debug(f"Vision extraction successful for {image_file}. Text length: {len(extracted_text)}")
        return extracted_text
    except Exception as e:
        logger.error(f"Vision extraction failed for {image_file}. Error: {e}")
        raise

def generate_audio(
    input_method: str,
    files: Optional[List[str]],
    input_text: Optional[str],
    voice: str,
    vision_api_key: str = None,
) -> (str, str, str, str):
    """
    Generates audio from text or files. This function has been refactored
    to directly convert the input text to speech.
    """
    start_time = time.time()
        
    resolved_vision_api_key = vision_api_key or os.getenv("VISION_API_KEY")

    api_key = os.getenv("TTS_API_KEY")
    if not api_key:
        raise gr.Error("TTS_API_KEY environment variable not set.")

    full_text = ""
    gr.Info("📦 Processing input...")
    title_base = "Audio"

    # --- Input Processing ---
    if input_method == "Upload Files":
        if not files:
            raise gr.Error("Please upload at least one file.")
        texts = []
        file_names = []
        for file_path in files:
            actual_file_path = file_path.name if hasattr(file_path, 'name') else file_path
            file_path_obj = Path(actual_file_path)
            file_names.append(file_path_obj.stem)
            text = ""
            if is_pdf(str(file_path_obj)):
                try:
                    with open(actual_file_path, "rb") as f:
                        reader = PdfReader(f)
                        text = "\n\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                except Exception as e:
                    raise gr.Error(f"Error reading PDF: {e}")
            elif is_image(str(file_path_obj)):
                if not resolved_vision_api_key:
                    raise gr.Error(
                        "Vision API Key not found. Please provide it in Advanced Settings "
                        "or set the VISION_API_KEY environment variable to process images."
                    )
                try:
                    text = extract_text_from_image_via_vision(str(file_path_obj), resolved_vision_api_key)
                except Exception as e:
                    raise gr.Error(f"Error extracting text from image: {e}")
            elif is_text(str(file_path_obj)):
                with open(actual_file_path, "r", encoding="utf-8", errors='ignore') as f:
                    text = f.read()
            elif is_docx(str(file_path_obj)):
                doc = docx.Document(actual_file_path)
                text = "\n\n".join(p.text for p in doc.paragraphs if p.text)
            else:
                raise gr.Error(f"Unsupported file type: {file_path_obj.name}")
            texts.append(text)
        full_text = "\n\n".join(filter(None, texts))
        title_base = file_names[0] if len(file_names) == 1 else f"{len(file_names)} Files"

    elif input_method == "Enter Text":
        if not input_text or not input_text.strip():
            raise gr.Error("Please enter text.")
        full_text = input_text
        title_base = "Pasted Text"


    if not full_text.strip():
        raise gr.Error("No text content to process.")

    logger.info(f"Total input text length: {len(full_text)} characters.")

    # --- TTS Generation ---
    text_chunks = split_text(full_text)
    total_chunks = len(text_chunks)
    
    logger.info(f"Starting TTS generation for {total_chunks} chunks.")
    gr.Info(f"🪄 Generating audio for {total_chunks} text chunks...")

    audio_chunks = [None] * total_chunks
    processed_count = 0
    actual_voice = VOICE_MAP.get(voice, "nova") # Default to nova if not found

    with cf.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_chunk = {
            executor.submit(get_mp3, chunk, actual_voice, api_key): i
            for i, chunk in enumerate(text_chunks)
        }
        
        for future in cf.as_completed(future_to_chunk):
            try:
                chunk_index = future_to_chunk[future]
                audio_chunks[chunk_index] = future.result()
                processed_count += 1
                gr.Info(f"🪄 Generated audio for chunk {processed_count}/{total_chunks}...")
            except Exception as exc:
                logger.error(f"TTS generation failed for a chunk: {exc}")
                raise gr.Error("Failed to generate audio for a part of the text. Please check your API key and network.")

    # --- Finalization ---
    final_audio = b"".join(chunk for chunk in audio_chunks if chunk is not None)
 
    if not final_audio:
        raise gr.Error("Failed to generate any audio. Please check the TTS service status or your API key.")

    temporary_directory = "./gradio_cached_files/tmp/" 
    os.makedirs(temporary_directory, exist_ok=True)

    try:
        temp_file_path = None
        with NamedTemporaryFile(
            dir=temporary_directory,
            delete=False,
            suffix=".mp3",
            prefix="SpeakAI_audio_"
        ) as temp_file:
            temp_file.write(final_audio)
            temp_file_path = temp_file.name

        if temp_file_path:
            logger.info(f"Audio saved to temporary file: {temp_file_path}")
        else:
            raise IOError("Temporary file path was not obtained.")

    except Exception as e:
        logger.error(f"Failed to write temporary audio file: {e}")
        raise gr.Error("Failed to save the generated audio file.")

    try:
        for file in glob.glob(f"{temporary_directory}SpeakAI_audio_*.mp3"):
            if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 7 * 24 * 60 * 60: # Delete temp audio file after 7 days
                try:
                    os.remove(file)
                    logger.debug(f"Removed old temp audio file: {file}")
                except OSError as e_rem:
                    logger.warning(f"Could not remove old temp audio file {file}: {e_rem}")
    except Exception as e:
        logger.warning(f"Error during old temp audio file cleanup: {e}")

    # --- History and UI Update ---
    hk_now = datetime.datetime.now(pytz.timezone('Asia/Hong_Kong'))
    final_title = f"{title_base} - {hk_now.strftime('%Y-%m-%d %H:%M')}"
    
    characters = len(full_text)
    tts_cost = (characters / 1_000_000) * 15
    
    data_to_send = {
        "title": final_title,
        "audio_file_component_id": "temp_audio_file_url_holder",
        "transcript": full_text,
        "tts_cost": f"{tts_cost:.2f}"
    }
    json_data_string = json.dumps(data_to_send)
    
    total_duration = time.time() - start_time
    gr.Info(f"🎉 Audio generation complete! Total time: {total_duration:.2f} seconds.")
    gr.Info(f"💸 Estimated cost: US${tts_cost:.2f}.")

    return temp_file_path, full_text, json_data_string, temp_file_path


# --- Gradio UI Definition ---
allowed_extensions = [".txt", ".docx", ".pdf", ".jpg", ".jpeg", ".png"]

def read_file_content(filepath: str, default: str = "") -> str:
    try:
        return Path(filepath).read_text(encoding='utf-8')
    except (FileNotFoundError, Exception):
        return default

description_md = read_file_content("description.md", "AI-powered text-to-speech application.")
footer_md = read_file_content("footer.md", "")
head_html = read_file_content("head.html", "")

with gr.Blocks(theme="ocean", title="Mr.🆖 SpeakAI 🗣️" flagging_options=[], css="footer{display:none !important}") as demo:
    gr.Markdown(description_md)

    with gr.Row():
        # --- Left Column: Inputs ---
        with gr.Column(scale=1):
            input_method_radio = gr.Radio(
                ["Enter Text", "Upload Files"],
                label="📁 Source",
                value="Enter Text"
            )

            with gr.Group(visible=True) as text_input_group:
                text_input = gr.Textbox(
                    label="✍️ Enter Text",
                    lines=15,
                    placeholder="Paste or type your text here..."
                )
            
            with gr.Group(visible=False) as file_upload_group:
                file_input = gr.Files(
                    label="Upload TXT, DOCX, PDF, JPG, JPEG, or PNG Files",
                    file_types=allowed_extensions,
                    file_count="multiple",
                )
            
            voice_input = gr.Dropdown(
                label="🎤 Voice",
                choices=OPENAI_VOICES,
                value="Female 1",
            )

            GET_KEY_URL = "https://api.mr5ai.com"
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                gr.Markdown(
                    f"💡 Get your Mr.🆖 AI Hub API Key [here]({GET_KEY_URL})"
                )
                vision_api_key_input = gr.Textbox(
                    label="Mr.🆖 AI Hub API Key",
                    type="password",
                    placeholder="sk-xxx",
                    elem_id="mr_ng_ai_hub_vision_api_key_input"
                )

            submit_button = gr.Button("✨ Generate Audio", variant="primary")

        # --- Right Column: Outputs ---
        with gr.Column(scale=1):
            transcript_output = gr.Textbox(
                label="📃 Input Text", 
                lines=20, 
                show_copy_button=True, 
                autoscroll=False, 
                elem_id="audio_transcript_display"
            )
            audio_output = gr.Audio(
                label="🔊 Audio Output", 
                type="filepath", 
                elem_id="audio_player"
            )

    with gr.Accordion("📜 Archives (Stored in your browser)", open=False):
        gr.HTML("<ul id='audioHistoryList' style='list-style-type: none; padding: 0;'><li>Loading archives...</li></ul>")
        js_trigger_data_textbox = gr.Textbox(label="JS Trigger Data", visible=False, elem_id="js_trigger_data_textbox")
        temp_audio_file_output_for_url = gr.File(label="Temp Audio File URL Holder", visible=False, elem_id="temp_audio_file_url_holder")

    # --- UI Logic ---
    def switch_input_method(choice):
        return {
            text_input_group: gr.update(visible=choice == "Enter Text"),
            file_upload_group: gr.update(visible=choice == "Upload Files")
        }

    input_method_radio.change(
        fn=switch_input_method,
        inputs=input_method_radio,
        outputs=[text_input_group, file_upload_group]
    )

    submit_button.click(
        fn=generate_audio,
        inputs=[
            input_method_radio,
            file_input,
            text_input,
            voice_input,
            vision_api_key_input
        ],
        outputs=[audio_output, transcript_output, js_trigger_data_textbox, temp_audio_file_output_for_url],
        api_name="generate_audio"
    )


    gr.Markdown(footer_md)
    demo.head = (os.getenv("HEAD", "") or "") + head_html


# --- App Setup & Launch ---
demo = demo.queue(max_size=20, default_concurrency_limit=5)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    os.makedirs("./gradio_cached_files/tmp/", exist_ok=True)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
