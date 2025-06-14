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
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from mimetypes import guess_type
import docx # Added for DOCX support
import requests # Added for URL fetching
from bs4 import BeautifulSoup # Added for HTML parsing
import pytz

MINIMAX_CANTONESE_VOICE_MAPPINGS = {
    "female-1": "English_captivating_female1",
    "male-1": "Cantonese_Narrator",
    "female-2": "Cantonese_GentleLady",
    "male-2": "English_Comedian",
}

MINIMAX_ENGLISH_VOICE_MAPPINGS = {
    "female-1": "English_captivating_female1",
    "male-1": "English_Magnetic_Male_2",
    "female-2": "English_Kind-heartedGirl",
    "male-2": "English_Lively_Male_11",
}

MINIMAX_CHINESE_VOICE_MAPPINGS = {
    "female-1": "Chinese (Mandarin)_Crisp_Girl",
    "male-1": "Chinese (Mandarin)_Gentleman",
    "female-2": "Chinese (Mandarin)_Warm-HeartedAunt",
    "male-2": "Chinese (Mandarin)_Sincere_Adult",
}

if sentry_dsn := os.getenv("SENTRY_DSN"):
    sentry_sdk.init(sentry_dsn)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class DialogueItem(BaseModel):
    text: str
    speaker: Literal["female-1", "male-1", "female-2", "male-2"]

    def voice(self, language: str = "English"): # Add language parameter, remove @property
        if language == "English":
            return MINIMAX_ENGLISH_VOICE_MAPPINGS[self.speaker]
        if language == "Chinese":
            return MINIMAX_CHINESE_VOICE_MAPPINGS[self.speaker]
        if language == "Cantonese":
            return MINIMAX_CANTONESE_VOICE_MAPPINGS[self.speaker]
        # If language is not one of the above, this will implicitly return None or raise KeyError
        # depending on whether self.speaker exists in a non-existent map.
        # This path should ideally not be reached with current UI constraints.
        logger.error(f"Unsupported language '{language}' encountered in DialogueItem.voice(). Falling back to MiniMax English voices as a default, but this indicates an issue.")
        return MINIMAX_ENGLISH_VOICE_MAPPINGS[self.speaker] # Defaulting to English if language is unexpected


class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]


# Add retry mechanism to MiniMax TTS calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
def get_mp3_minimax(text: str, voice: str, language: str, group_id: str = None, api_key: str = None) -> bytes:
    """Generates MP3 audio for the given text using MiniMax TTS, with retries."""
    resolved_group_id = group_id or os.getenv("MINIMAX_GROUP_ID")
    resolved_api_key = api_key or os.getenv("MINIMAX_API_KEY")

    if not resolved_group_id or not resolved_api_key:
        logger.error("MiniMax Group ID or API Key not configured.")
        raise ValueError("MiniMax Group ID or API Key not configured.")

    url = f"https://api.minimax.io/v1/t2a_v2?GroupId={resolved_group_id}"

    language_boost_map = {
        "English": "English",
        "Chinese": "Chinese",
        "Cantonese": "Chinese,Yue"
    }
    language_boost_value = language_boost_map.get(language, "English") # Default to English if language not in map

    payload = json.dumps({
      "model":"speech-02-turbo",
      "text": text,
      "stream": False,
      "voice_setting":{
        "voice_id": voice,
        "speed": 1,
        "vol": 2,
        "pitch": 0
      },
      "audio_setting":{
        "sample_rate": 44100,
        "bitrate": 128000,
        "format": "mp3",
        "channel": 1
      },
      "language_boost": language_boost_value
    })
    headers = {
      'Authorization': f'Bearer {resolved_api_key}',
      'Content-Type': 'application/json'
    }
    logger.debug(f"Requesting MiniMax TTS for voice '{voice}', text: '{text[:50]}...'")
    try:
        response = requests.request("POST", url, stream=True, headers=headers, data=payload, timeout=60.0)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Check if response content type is JSON, otherwise it might be an error page or unexpected format
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            logger.error(f"MiniMax TTS returned non-JSON response. Status: {response.status_code}. Content: {response.text[:200]}")
            raise requests.exceptions.RequestException(f"MiniMax TTS returned non-JSON response. Status: {response.status_code}")

        parsed_json = response.json() # Use response.json() for direct parsing

        if parsed_json.get("base_resp", {}).get("status_code") != 0:
            error_msg = parsed_json.get("base_resp", {}).get("status_msg", "Unknown MiniMax API error")
            logger.error(f"MiniMax TTS API error: {error_msg}. Full response: {parsed_json}")
            raise Exception(f"MiniMax TTS API error: {error_msg}")

        audio_hex = parsed_json.get('data', {}).get('audio')
        if not audio_hex:
            logger.error(f"No audio data in MiniMax TTS response. Full response: {parsed_json}")
            raise Exception("No audio data in MiniMax TTS response")
            
        audio_bytes = bytes.fromhex(audio_hex)
        logger.debug(f"MiniMax TTS generation successful for voice '{voice}', text: '{text[:50]}...'")
        return audio_bytes
    except requests.exceptions.HTTPError as e:
        logger.error(f"MiniMax TTS HTTP error: {e}. Response: {e.response.text[:200] if e.response else 'No response body'}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"MiniMax TTS request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"MiniMax TTS generation failed for voice '{voice}', text: '{text[:50]}...'. Error: {e}")
        raise # Reraise exception to trigger tenacity retry

def is_pdf(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    # Check extension and guessed MIME type
    return filename.lower().endswith(".pdf") or (t or "").endswith("pdf")

def is_image(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    image_exts = (".jpg", ".jpeg", ".png")
    # Check extension and guessed MIME type
    return filename.lower().endswith(image_exts) or (t or "").startswith("image")

def is_text(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    # Check extension and guessed MIME type
    return filename.lower().endswith(".txt") or (t or "") == "text/plain"

def is_docx(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    # Check extension and guessed MIME type
    return filename.lower().endswith(".docx") or (t or "") == docx_mime

# Add retry mechanism to Vision calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
def extract_text_from_image_via_vision(image_file, openai_api_key=None):
    """Extracts text from an image using OpenAI Vision API, with retries."""
    client = OpenAI(
        api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=120.0 # Longer timeout for vision potentially
    )
    logger.debug(f"Requesting Vision text extraction for image: {image_file}")
    try:
        with open(image_file, "rb") as f:
            data = f.read()
            mime_type = guess_type(image_file)[0] or "image/png" # Default to png if guess fails
            b64 = base64.b64encode(data).decode("utf-8")
            image_url = f"data:{mime_type};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "auto" # Use 'auto' or 'high' based on needs
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract all the computer-readable text from this image as accurately as possible. Avoid commentary, return only the extracted text."
                    },
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4.1-mini", # Ensure this model supports vision
            messages=messages,
            max_tokens=32768,
            temperature=0,
        )
        extracted_text = response.choices[0].message.content.strip()
        logger.debug(f"Vision extraction successful for {image_file}. Text length: {len(extracted_text)}")
        return extracted_text
    except Exception as e:
        logger.error(f"Vision extraction failed for {image_file}. Error: {e}")
        raise # Reraise for retry

# Helper function to extract text from URL
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), retry=retry_if_exception_type(requests.exceptions.RequestException))
def extract_text_from_url(url: str) -> str:
    """Fetches content from a URL and extracts text using BeautifulSoup."""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url # Basic correction for missing scheme
    logger.info(f"Fetching content from URL: {url}")
    try:
        headers = { # Add headers to mimic a browser visit
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

        # Attempt to decode using UTF-8, then fall back to apparent_encoding
        try:
            html_content = response.content.decode('utf-8')
        except UnicodeDecodeError:
            html_content = response.text # relies on apparent_encoding

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text: try common main content tags first, then fall back to body
        # This is a heuristic and might need adjustment for specific site structures
        main_content_tags = ['article', 'main', '.main-content', '#main', '#content', '.post-content', '.entry-content']
        text_parts = []
        found_main_content = False

        for tag_selector in main_content_tags:
            elements = soup.select(tag_selector)
            if elements:
                for element in elements:
                    text_parts.append(element.get_text(separator='\n', strip=True))
                found_main_content = True
                break # Stop if main content is found

        if not found_main_content and soup.body:
            text_parts.append(soup.body.get_text(separator='\n', strip=True))
        
        extracted_text = "\n\n".join(filter(None, text_parts))

        if not extracted_text.strip():
            logger.warning(f"No significant text extracted from URL: {url}")
            return "" # Return empty string if no text is found
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from URL: {url}")
        return extracted_text

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching URL {url}: {e}")
        raise gr.Error(f"Failed to fetch content from URL: {url}. Server returned: {e.response.status_code}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error fetching URL {url}: {e}")
        raise gr.Error(f"Failed to connect to URL: {url}. Please check the URL and your internet connection.")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching URL {url}")
        raise gr.Error(f"Fetching content from URL {url} timed out. The website might be slow or unresponsive.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        raise gr.Error(f"An error occurred while trying to fetch the URL: {url}. Details: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {e}")
        raise gr.Error(f"An unexpected error occurred while processing the URL {url}. Error: {e}")


def generate_audio(
    input_method: str,
    files: Optional[List[str]],
    input_text: Optional[str],
    url_input: Optional[str], # Added url_input
    language: str = "English",
    openai_api_key: str = None,
) -> (str, str, str, str): # Added 4th str for the hidden gr.File component
    """Generates podcast audio from uploaded files, direct text input, or URL."""
    start_time = time.time()
    minimax_group_id = os.getenv("MINIMAX_GROUP_ID")
    minimax_api_key = os.getenv("MINIMAX_API_KEY")
    
    # API Key Check
    # MiniMax keys are required for all currently supported TTS languages
    if language in ["English", "Chinese", "Cantonese"]:
        if not (minimax_group_id and minimax_api_key):
            logger.error(f"MINIMAX_GROUP_ID and MINIMAX_API_KEY must be set as environment variables for {language} TTS.")
            raise gr.Error(f"MiniMax Group ID and API Key are required for {language} TTS. Please set them as environment variables (MINIMAX_GROUP_ID, MINIMAX_API_KEY).")
    
    # OpenAI API key is needed for dialogue generation and vision, regardless of TTS choice.
    if not (openai_api_key or os.getenv("OPENAI_API_KEY")):
        raise gr.Error("Mr.🆖 AI Hub API Key is required. Please provide it in Advanced Settings.")

    # Resolve OpenAI API key and Base URL once (used for dialogue generation)
    resolved_openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    resolved_openai_base_url = os.getenv("OPENAI_BASE_URL")
    
    # MiniMax keys are resolved within get_mp3_minimax using os.getenv

    full_text = ""
    gr.Info("📦 Processing input...")
    podcast_title_base = "Podcast" # Default title base

    if input_method == "Upload Files":
        if not files:
            raise gr.Error("Please upload at least one file or switch to another input method.")
        texts = []
        file_names = []
        for file_path in files:
            if not file_path:
                logger.warning("Received an empty file path in the list, skipping.")
                continue
            actual_file_path = file_path.name if hasattr(file_path, 'name') else file_path
            file_path_obj = Path(actual_file_path)
            file_names.append(file_path_obj.stem)
            logger.info(f"Processing file: {file_path_obj.name}")
            text = ""

            if is_pdf(str(file_path_obj)):
                try:
                    with file_path_obj.open("rb") as f:
                        reader = PdfReader(f)
                        if reader.is_encrypted:
                             logger.warning(f"Skipping encrypted PDF: {file_path_obj.name}")
                             raise gr.Error(f"Cannot process password-protected PDF: {file_path_obj.name}")
                        page_texts = [page.extract_text() for page in reader.pages if page.extract_text()]
                        text = "\n\n".join(page_texts) if page_texts else ""
                        if not text: logger.warning(f"No text extracted from PDF: {file_path_obj.name}")
                except Exception as e:
                    logger.error(f"Error reading PDF {file_path_obj.name}: {e}")
                    if "PdfReadError" in str(type(e)):
                         raise gr.Error(f"Error reading PDF file: {file_path_obj.name}. It might be corrupted or improperly formatted.")
                    else:
                         raise gr.Error(f"Error processing PDF file: {file_path_obj.name}.")
            elif is_image(str(file_path_obj)):
                try:
                    text = extract_text_from_image_via_vision(str(file_path_obj), resolved_openai_api_key)
                except Exception as e:
                    logger.error(f"Error processing image {file_path_obj.name} with Vision API: {e}")
                    raise gr.Error(f"Error extracting text from image: {file_path_obj.name}. Check API key, file format, and OpenAI status. Error: {e}")
            elif is_text(str(file_path_obj)):
                try:
                    with open(actual_file_path, "r", encoding="utf-8", errors='ignore') as f: 
                        text = f.read()
                except Exception as e:
                    logger.error(f"Error reading text file {file_path_obj.name}: {e}")
                    raise gr.Error(f"Error reading text file: {file_path_obj.name}. Check encoding. Error: {e}")
            elif is_docx(str(file_path_obj)):
                try:
                    doc = docx.Document(actual_file_path)
                    paragraphs = [p.text for p in doc.paragraphs if p.text]
                    text = "\n\n".join(paragraphs)
                    if not text: logger.warning(f"No text extracted from DOCX: {file_path_obj.name}")
                except Exception as e:
                    logger.error(f"Error reading DOCX file {file_path_obj.name}: {e}")
                    if "PackageNotFoundError" in str(type(e)):
                        raise gr.Error(f"Error reading DOCX file: {file_path_obj.name}. It might be corrupted, not a valid DOCX format, or password-protected.")
                    else:
                        raise gr.Error(f"Error processing DOCX file: {file_path_obj.name}.")
            else:
                try:
                   f_size = file_path_obj.stat().st_size
                   if f_size > 0:
                       raise gr.Error(f"Unsupported file type: {file_path_obj.name}. Please upload TXT, PDF, DOCX, or image file (JPG, JPEG, PNG). Note: Older .doc format is not supported.")
                   else:
                       logger.warning(f"Skipping empty or placeholder file: {file_path_obj.name}")
                       text = ""
                except FileNotFoundError:
                    logger.warning(f"File not found during processing, likely a temporary file issue: {actual_file_path}")
                    text = ""
                except Exception as e:
                     logger.error(f"Error checking file status for {file_path_obj.name}: {e}")
                     raise gr.Error(f"Error accessing file: {file_path_obj.name}.")
            texts.append(text)
        full_text = "\n\n".join(filter(None, texts))
        if not full_text.strip():
             raise gr.Error("Could not extract any text from the uploaded file(s). Please check the files or try different ones.")
        if file_names:
            podcast_title_base = file_names[0] if len(file_names) == 1 else f"{len(file_names)} Files"


    elif input_method == "Enter Text":
        if not input_text or not input_text.strip():
            raise gr.Error("Please enter text or switch to another input method.")
        full_text = input_text
        podcast_title_base = "Pasted Text"

    elif input_method == "URL": # New block for URL input
        if not url_input or not url_input.strip():
            raise gr.Error("Please enter a URL or switch to another input method.")
        try:
            full_text = extract_text_from_url(url_input)
            if not full_text.strip():
                raise gr.Error(f"Could not extract any meaningful text from the URL: {url_input}. The page might be empty, primarily image-based without OCR, or protected.")
            podcast_title_base = url_input.split('//')[-1].split('/')[0] # Domain as base title
        except gr.Error as e: # Catch Gradio errors from extract_text_from_url
            raise e # Re-raise to display in UI
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during URL processing for {url_input}: {e}")
            raise gr.Error(f"An unexpected error occurred while processing the URL: {url_input}. Please try again or use a different URL.")

    else:
        raise gr.Error("Invalid input method selected.")

    logger.info(f"Total input text length: {len(full_text)} characters.")
    if not full_text.strip(): # Double check after all input methods
        raise gr.Error("No text content to process. Please provide valid input.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15), retry=retry_if_exception_type(ValidationError))
    @llm(
        model="openai/mr5-podcast-ai", # This LLM call might still use OpenAI-compatible endpoint
        api_key=resolved_openai_api_key, # Dialogue generation still uses OpenAI key
        base_url=resolved_openai_base_url,
        temperature=0.5,
        max_tokens=16384
    )
    def generate_dialogue(text: str, language: str) -> Dialogue:
        """
        Your task is to take the input text provided and turn it into an engaging, informative podcast dialogue. The input text may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points and interesting facts that could be discussed in a podcast.

        Important: The ENTIRE podcast dialogue (including brainstorming, scratchpad, and actual dialogue) should be written in {language}. If 'Chinese' or 'Cantonese', use correct idiomatic Traditional Chinese (繁體中文) suitable for a Hong Kong audience.

        Here is the input text you will be working with:

        <input_text>
        {text}
        </input_text>

        First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for an audio podcast.

        <scratchpad>
        Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

        Keep in mind that your podcast should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

        Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

        Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.
        </scratchpad>

        Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.

        <podcast_dialogue>
        Write your engaging, informative podcast dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience. Use made-up names for the hosts and guests to create a more engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio. Assign appropriate speakers (female-1, male-1, female-2, male-2) to each line, varying them for a natural conversation. Ensure the output strictly adheres to the required format: a list of objects, each with 'text' and 'speaker' fields.

        Make the dialogue as long and detailed as possible, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

        At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.
        </podcast_dialogue>
        """

    try:
        gr.Info("✨ Generating dialogue script with AI...")
        llm_start_time = time.time()
        llm_output = generate_dialogue(full_text, language)
        logger.info(f"Dialogue generation took {time.time() - llm_start_time:.2f} seconds.")

    except ValidationError as e:
        logger.error(f"LLM output validation failed after retries: {e}")
        raw_output = getattr(e, 'llm_output', str(e)) 
        raise gr.Error(f"The AI model returned an unexpected format even after retries. Please try again or simplify the input. Raw output hint: {str(raw_output)[:500]}...")
    except Exception as e:
        logger.error(f"Error during dialogue generation: {e}")
        error_str = str(e).lower()
        if "authentication" in error_str:
             raise gr.Error("Authentication error with API. Please check your API key.")
        elif "rate limit" in error_str:
             raise gr.Error("API rate limit exceeded. Please wait and try again, or check your usage tier.")
        elif "base_url" in error_str or "connection" in error_str: # This refers to OpenAI connection for dialogue
            dialogue_base_url = resolved_openai_base_url or 'the configured OpenAI endpoint for dialogue generation'
            raise gr.Error(f"Could not connect to {dialogue_base_url}. Please check the URL and network connection.")
        elif "invalid request" in error_str and "image" in error_str:
             raise gr.Error("Error processing image with Vision API. The image might be invalid, unsupported, or the model doesn't support image input.")
        else:
            raise gr.Error(f"An error occurred during dialogue generation: {e}")

    if not llm_output or not llm_output.dialogue:
        raise gr.Error("The AI failed to generate a dialogue script. The input might be too short or unclear.")

    characters = 0
    total_lines = len(llm_output.dialogue)
    logger.info(f"Starting TTS generation for {total_lines} dialogue lines.")
    gr.Info(f"🪄 Generating audio for {total_lines} dialogue lines... (this may take a while)")

    results = [None] * total_lines

    with cf.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {}
        if language in ["English", "Chinese", "Cantonese"]: # All these now use MiniMax
            if not (minimax_group_id and minimax_api_key): # Check again before submitting tasks
                 raise gr.Error(f"MiniMax Group ID and API Key environment variables (MINIMAX_GROUP_ID, MINIMAX_API_KEY) are required for {language} TTS generation but not found.")
            future_to_index = {
                executor.submit(get_mp3_minimax, line.text, line.voice(language), language): i # Pass language to get_mp3_minimax
                for i, line in enumerate(llm_output.dialogue) if line.text.strip()
            }
        # The 'else' block for OpenAI TTS has been removed as all current languages use MiniMax.
        # If new languages are added that don't use MiniMax, this section would need to be revisited.
        
        for line in llm_output.dialogue:
            if line.text.strip():
                characters += len(line.text)

        processed_count = 0
        tts_start_time = time.time()
        for future in cf.as_completed(future_to_index):
            index = future_to_index[future]
            line_obj = llm_output.dialogue[index]
            transcript_line = f"{line_obj.speaker}: {line_obj.text}"
            try:
                audio_chunk = future.result()
                results[index] = (transcript_line, audio_chunk)
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == total_lines: 
                     gr.Info(f"🪄 Generated audio for {processed_count}/{total_lines} lines...")
            except Exception as exc:
                 logger.error(f'TTS generation failed for line {index+1} after retries: {exc}')
                 error_msg = f"[TTS Error: Failed audio for line {index+1}]"
                 results[index] = (transcript_line, error_msg) 

    logger.info(f"TTS generation took {time.time() - tts_start_time:.2f} seconds.")
    logger.info(f"Total characters for TTS: {characters}")

    gr.Info("🧩 Combining audio segments...")
    final_audio_chunks = []
    final_transcript_lines = []
    successful_lines = 0
    for i, result in enumerate(results):
        line_obj = llm_output.dialogue[i] 
        if result is None:
            if line_obj.text.strip(): 
                logger.error(f"Result missing for non-empty line {i+1}. Original text: '{line_obj.text[:50]}...' Skipping.")
                final_transcript_lines.append(f"[Internal Error: Audio result missing for line {i+1}] {line_obj.speaker}: {line_obj.text}")
            continue 

        transcript_part, audio_part = result
        final_transcript_lines.append(transcript_part) 
        if isinstance(audio_part, bytes):
            final_audio_chunks.append(audio_part)
            successful_lines += 1
    
    if not final_audio_chunks:
        if any("[TTS Error" in line for line in final_transcript_lines):
             raise gr.Error("Failed to generate audio for all lines. Please check the transcript for details and review API key/status.")
        else:
             raise gr.Error("Failed to generate any audio, although dialogue script was created. Check TTS service status or API key.")

    audio = b"".join(final_audio_chunks)
    transcript = "\n\n".join(final_transcript_lines)

    logger.info(f"Successfully generated audio for {successful_lines}/{total_lines} lines.")

    temporary_directory = "./gradio_cached_files/tmp/" 
    os.makedirs(temporary_directory, exist_ok=True)

    try:
        temp_file_path = None
        with NamedTemporaryFile(
            dir=temporary_directory,
            delete=False, 
            suffix=".mp3",
            prefix="podcast_audio_"
        ) as temp_file:
             temp_file.write(audio)
             temp_file_path = temp_file.name 

        if temp_file_path:
             logger.info(f"Audio saved to temporary file: {temp_file_path}")
        else:
             raise IOError("Temporary file path was not obtained.")

    except Exception as e:
        logger.error(f"Failed to write temporary audio file: {e}")
        raise gr.Error("Failed to save the generated audio file.")

    try:
        for file in glob.glob(f"{temporary_directory}podcast_audio_*.mp3"):
            if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60: 
                try:
                    os.remove(file)
                    logger.debug(f"Removed old temp file: {file}")
                except OSError as e_rem:
                     logger.warning(f"Could not remove old temp file {file}: {e_rem}") 
    except Exception as e: 
        logger.warning(f"Error during old temp file cleanup: {e}")

    total_duration = time.time() - start_time
    gr.Info(f"🎉 Podcast generation complete! Total time: {total_duration:.2f} seconds.")

    # Prepare podcast title for history
    # Get current time in UTC
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    # Define Hong Kong timezone
    hk_tz = pytz.timezone('Asia/Hong_Kong')
    # Convert UTC time to Hong Kong time
    hk_now = utc_now.astimezone(hk_tz)
    # Format the Hong Kong time
    final_podcast_title = f"{podcast_title_base} - {hk_now.strftime('%Y-%m-%d %H:%M')}"
    
    # Escape transcript for JavaScript string literal
    escaped_transcript = transcript.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

    # Create JavaScript to call the save function in head.html
    # The audio file path needs to be accessible by the client's browser.
    # Gradio serves files from a temporary location. We need to ensure this path is correct.
    # If temp_file_path is absolute, we might need to make it relative or ensure it's served.
    # For Gradio, files in `file_output` are typically served under `/file=...`
    # We assume `temp_file_path` as returned by `gr.Audio` is directly fetchable.
    
    # The audio_output component in Gradio will have a URL like /file=/path/to/temp_file.mp3
    # We need to pass this server-relative path to the JS function.
    # `temp_file_path` is the absolute path on the server.
    # Gradio's `gr.Audio(type="filepath")` returns the absolute path.
    # When this path is set as the value of an `gr.Audio` output, Gradio makes it accessible via a `/file=` URL.
    # The JS in `head.html` will fetch this URL.

    # Construct the web-accessible URL for the temporary file
    # Gradio serves files from gr.Audio(type="filepath") via /file=<path>
    # Ensure the path is properly URL encoded if it contains special characters, though Gradio might handle this.
    # For simplicity, we'll assume basic paths for now. If issues persist, URL encoding might be needed.
    # gradio_file_url = f"/file={temp_file_path}" # Original absolute path approach

    temp_file_path_obj = Path(temp_file_path)
    try:
        # Path.cwd() in the context of the uvicorn server (likely /app in Docker)
        app_root = Path.cwd()
        relative_temp_path = temp_file_path_obj.relative_to(app_root)
        gradio_file_url = f"/file={relative_temp_path}"
        logger.info(f"Constructed relative gradio_file_url: {gradio_file_url}")
    except ValueError:
        # Fallback if relative_to fails (e.g. different drives or not a subpath)
        gradio_file_url = f"/file={temp_file_path}" # Keep the absolute path
        logger.warning(f"Could not make path relative to CWD ('{app_root}'). Using absolute path for gradio_file_url: {gradio_file_url}")


    # The audio_url will now be fetched by JS from a hidden gr.File component
    # We pass the raw temp_file_path to that component.
    # The JSON will just contain a marker or the title to correlate.
    data_to_send = {
        "title": final_podcast_title,
        # "audio_url": gradio_file_url, # REMOVED - JS will get this from hidden gr.File
        "audio_file_component_id": "temp_audio_file_url_holder", # ID of the hidden gr.File
        "transcript": transcript
    }
    json_data_string = json.dumps(data_to_send)
    
    logger.debug(f"Returning JSON data for JS trigger (no audio_url, JS will fetch from component): {json_data_string[:200]}...")

    return temp_file_path, transcript, json_data_string, temp_file_path # 4th item for hidden gr.File


# --- Gradio UI Definition ---

allowed_extensions = [
    ".txt", ".pdf", ".docx", ".jpg", ".jpeg", ".png" 
]

examples_dir = Path("examples")
examples = [
    [ # Input method, files, text, url, language, api_key
        "Upload Files", [str(examples_dir / "Intangible cultural heritage item.pdf")], "", "", "English", None
    ],
    [
        "Upload Files", [str(examples_dir / "JUPAS Guide.jpg")], "", "", "Chinese", None
    ],
    [
        "URL", None, "", "https://geographical.co.uk/culture/uncontacted-tribes-around-the-world", "Cantonese", None
    ]
]

def read_file_content(filepath: str, default: str = "") -> str:
    try:
        return Path(filepath).read_text(encoding='utf-8') 
    except FileNotFoundError:
        logger.warning(f"{filepath} not found, using default content.")
        return default
    except Exception as e:
         logger.error(f"Error reading file {filepath}: {e}. Using default.")
         return default


description_md = read_file_content("description.md", "Generate a podcast from text, documents, or a URL.")
footer_md = read_file_content("footer.md", "")
head_html = read_file_content("head.html", "")


with gr.Blocks(theme="ocean", title="Mr.🆖 PodcastAI 🎙️🎧") as demo: # Reverted allowed_paths
    gr.Markdown(description_md)

    with gr.Row():
        input_method_radio = gr.Radio(
            ["Upload Files", "Enter Text", "URL"], # Added "URL"
            label="📁 Sources",
            value="Upload Files"
        )

    with gr.Group(visible=True) as file_upload_group:
        file_input = gr.Files(
            label="Upload TXT, PDF, DOCX, or Image Files",
            file_types=allowed_extensions,
            file_count="multiple",
        )

    with gr.Group(visible=False) as text_input_group: 
        text_input = gr.Textbox(
            label="✍️ Enter Text",
            lines=10,
            placeholder="Paste or type your text here..."
        )
    
    with gr.Group(visible=False) as url_input_group: # New URL input group
        url_input_field = gr.Textbox( # Renamed to avoid conflict if needed later
            label="🔗 Enter URL",
            lines=1,
            placeholder="https://example.com/article"
        )

    lang_input = gr.Radio(
            label="🌐 Podcast Language",
            choices=["English", "Chinese", "Cantonese"],
            value="English",
        )

    API_KEY_URL = "https://api.mr5ai.com"
    with gr.Accordion("⚙️ Advanced Settings", open=False):
        gr.Markdown(
            f"💡 Get your Mr.🆖 AI Hub API Key [here]({API_KEY_URL})"
        )
        api_key_input = gr.Textbox(
                label="Mr.🆖 AI Hub API Key",
                type="password",
                placeholder="sk-xxx",
                elem_id="mr_ng_ai_hub_api_key_input"
        )
        # gr.Markdown(
        #     "For **Cantonese** TTS, ensure `MINIMAX_GROUP_ID` and `MINIMAX_API_KEY` are set as environment variables."
        # )
        # Future improvement: Add input fields for MiniMax keys if desired
        # minimax_group_id_input = gr.Textbox(label="MiniMax Group ID (for Cantonese)", type="password", placeholder="Enter MiniMax Group ID")
        # minimax_api_key_input = gr.Textbox(label="MiniMax API Key (for Cantonese)", type="password", placeholder="Enter MiniMax API Key")

    submit_button = gr.Button("✨ Generate Podcast", variant="primary")

    with gr.Column():
        audio_output = gr.Audio(label="Podcast Audio", type="filepath", elem_id="podcast_audio_player") # Keep existing elem_id
        transcript_output = gr.Textbox(label="📃 Transcript", lines=15, show_copy_button=True, autoscroll=False, elem_id="podcast_transcript_display") # Keep existing elem_id

    with gr.Accordion("📜 History (Stored in your browser)", open=False): # Keep existing Accordion
        # This HTML component will be populated by JavaScript from head.html
        podcast_history_display = gr.HTML("<ul id='podcastHistoryList' style='list-style-type: none; padding: 0;'><li>Loading history...</li></ul>")
        # Hidden Textbox component to pass JSON data to JavaScript
        js_trigger_data_textbox = gr.Textbox(label="JS Trigger Data", visible=False, elem_id="js_trigger_data_textbox")
        # Hidden File component to get a Gradio-served URL for the audio
        temp_audio_file_output_for_url = gr.File(label="Temp Audio File URL Holder", visible=False, elem_id="temp_audio_file_url_holder")


    def switch_input_method(choice):
        """Updates visibility and clears the inactive input fields."""
        is_upload = choice == "Upload Files"
        is_text = choice == "Enter Text"
        is_url = choice == "URL" # New condition

        # Determine visibility updates
        file_vis = is_upload
        text_vis = is_text
        url_vis = is_url # New visibility

        # Determine value updates (clear hidden fields)
        # gr.update() means no change to value
        file_val_update = gr.update(value=None) if not is_upload else gr.update()
        text_val_update = gr.update(value="") if not is_text else gr.update()
        url_val_update = gr.update(value="") if not is_url else gr.update() # New value update

        return {
            file_upload_group: gr.update(visible=file_vis),
            text_input_group: gr.update(visible=text_vis),
            url_input_group: gr.update(visible=url_vis), # Update URL group visibility
            file_input: file_val_update,
            text_input: text_val_update,
            url_input_field: url_val_update, # Update URL field value
        }

    input_method_radio.change(
        fn=switch_input_method,
        inputs=input_method_radio,
        outputs=[
            file_upload_group, 
            text_input_group, 
            url_input_group, # Add url_input_group to outputs
            file_input, 
            text_input,
            url_input_field  # Add url_input_field to outputs
        ]
    )

    submit_button.click(
        fn=generate_audio,
        inputs=[ # Order must match generate_audio parameters
            input_method_radio,
            file_input,
            text_input,
            url_input_field, # Added url_input_field
            lang_input,
            api_key_input
        ],
        outputs=[audio_output, transcript_output, js_trigger_data_textbox, temp_audio_file_output_for_url], # Added temp_audio_file_output_for_url
        api_name="generate_podcast"
    )

    gr.Examples(
        examples=examples,
        inputs=[ # Ensure order matches generate_audio parameters for examples
            input_method_radio, 
            file_input, 
            text_input, 
            url_input_field, # Added url_input_field
            lang_input, 
            api_key_input
        ],
        # Examples won't trigger the history save directly unless we adapt the example fn or outputs
        # For now, history save is only for manual generation.
        outputs=[audio_output, transcript_output, js_trigger_data_textbox, temp_audio_file_output_for_url], # Added temp_audio_file_output_for_url
        fn=generate_audio,
        cache_examples=True,
        run_on_click=True,
        label="Examples (Click for Demo)"
    )

    gr.Markdown(footer_md)
    demo.head = (os.getenv("HEAD", "") or "") + head_html

# --- App Setup & Launch ---

demo = demo.queue(
    max_size=20,
    default_concurrency_limit=5, 
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    examples_dir.mkdir(exist_ok=True)
    example_files = [
        "Intangible cultural heritage item.pdf",
        "JUPAS Guide.jpg"
    ]
    for fname in example_files:
        fpath = examples_dir / fname
        if not fpath.is_file():
            logger.warning(f"Example file {fpath} not found. Creating empty placeholder.")
            try:
                fpath.touch()
            except OSError as e:
                logger.error(f"Failed to create placeholder file {fpath}: {e}")

    os.makedirs("./gradio_cached_files/tmp/", exist_ok=True)

    logger.info("Starting Gradio application via Uvicorn...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)