# -*- coding: utf-8 -*-
import concurrent.futures as cf
import glob
import io
import os
import time
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal, Optional, Dict, Any

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

if sentry_dsn := os.getenv("SENTRY_DSN"):
    sentry_sdk.init(sentry_dsn)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class DialogueItem(BaseModel):
    text: str
    speaker: Literal["female-1", "male-1", "female-2", "male-2"]

    @property
    def voice(self):
        return {
            "female-1": "nova",
            "male-1": "alloy",
            "female-2": "shimmer",
            "male-2": "echo",
        }[self.speaker]


class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]


# Add retry mechanism to TTS calls for resilience
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
def get_mp3(text: str, voice: str, api_key: str = None) -> bytes:
    """Generates MP3 audio for the given text using OpenAI TTS, with retries."""
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=60.0 # Add timeout
    )
    logger.debug(f"Requesting TTS for voice '{voice}', text: '{text[:50]}...'")
    try:
        # Use the non-streaming version for simplicity within retry logic
        response = client.audio.speech.create(
            model="tts-1", # Consider tts-1-hd for higher quality if needed
            voice=voice,
            input=text,
            response_format="mp3"
        )
        logger.debug(f"TTS generation successful for voice '{voice}', text: '{text[:50]}...'")
        return response.content
    except Exception as e:
        logger.error(f"TTS generation failed for voice '{voice}', text: '{text[:50]}...'. Error: {e}")
        raise # Reraise exception to trigger tenacity retry

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
                        "text": "Extract all the computer-readable text from this image as accurately as possible. Preserve line breaks where appropriate. Avoid commentary, return only the extracted text."
                    },
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4.1-mini", # Ensure this model supports vision
            messages=messages,
            max_tokens=8192,
            temperature=0,
        )
        extracted_text = response.choices[0].message.content.strip()
        logger.debug(f"Vision extraction successful for {image_file}. Text length: {len(extracted_text)}")
        return extracted_text
    except Exception as e:
        logger.error(f"Vision extraction failed for {image_file}. Error: {e}")
        raise # Reraise for retry


def generate_audio(
    input_method: str,
    files: Optional[List[str]],
    input_text: Optional[str],
    language: str = "English",
    openai_api_key: str = None,
) -> (str, str):
    """Generates podcast audio from either uploaded files or direct text input."""
    start_time = time.time()
    if not (os.getenv("OPENAI_API_KEY") or openai_api_key):
        raise gr.Error("Mr.🆖 AI Hub API Key is required")

    # Resolve API key and Base URL once
    resolved_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    resolved_base_url = os.getenv("OPENAI_BASE_URL")

    full_text = ""
    gr.Info("Processing input...")
    if input_method == "Upload Files":
        if not files:
            raise gr.Error("Please upload at least one file or switch to text input.")
        texts = []
        for file_path in files:
            if not file_path:
                logger.warning("Received an empty file path in the list, skipping.")
                continue
            # Gradio >= 4 often returns temp file objects, get the path using .name
            actual_file_path = file_path.name if hasattr(file_path, 'name') else file_path
            file_path_obj = Path(actual_file_path)
            logger.info(f"Processing file: {file_path_obj.name}")
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
                    text = extract_text_from_image_via_vision(str(file_path_obj), resolved_api_key)
                except Exception as e:
                    logger.error(f"Error processing image {file_path_obj.name} with Vision API: {e}")
                    raise gr.Error(f"Error extracting text from image: {file_path_obj.name}. Check API key, file format, and OpenAI status. Error: {e}")
            elif is_text(str(file_path_obj)):
                try:
                    # Use open(actual_file_path...) instead of file_path_obj.open()
                    # as file_path_obj might refer to the TempPath object not the name string
                    with open(actual_file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except Exception as e:
                    logger.error(f"Error reading text file {file_path_obj.name}: {e}")
                    raise gr.Error(f"Error reading text file: {file_path_obj.name}. Check encoding. Error: {e}")
            else:
                try:
                   f_size = file_path_obj.stat().st_size
                   if f_size > 0:
                       raise gr.Error(f"Unsupported file type: {file_path_obj.name}. Please upload TXT, PDF, or image file (JPG, JPEG, PNG).")
                   else:
                       logger.warning(f"Skipping empty or placeholder file: {file_path_obj.name}")
                       text = ""
                except FileNotFoundError:
                    # This might happen if the temp file was cleaned up prematurely
                    logger.warning(f"File not found during processing, likely a temporary file issue: {actual_file_path}")
                    text = ""
                except Exception as e:
                     logger.error(f"Error checking file status for {file_path_obj.name}: {e}")
                     raise gr.Error(f"Error accessing file: {file_path_obj.name}.")
            texts.append(text)
        full_text = "\n\n".join(filter(None, texts))
        if not full_text.strip():
             raise gr.Error("Could not extract any text from the uploaded file(s). Please check the files or try different ones.")

    elif input_method == "Enter Text":
        if not input_text or not input_text.strip():
            raise gr.Error("Please enter text or switch to file upload.")
        full_text = input_text
    else:
        raise gr.Error("Invalid input method selected.")

    logger.info(f"Total input text length: {len(full_text)} characters.")

    # LLM Call needs Pydantic Models defined in scope
    # Add retry logic to LLM call as well
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15), retry=retry_if_exception_type(ValidationError))
    @llm(
        model="gpt-4.1-mini",
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        temperature=0.5, # Slightly increased temperature for potentially more engaging dialogue
        max_tokens=8192 # Explicitly set max_tokens for dialogue generation
    )
    def generate_dialogue(text: str, language: str) -> Dialogue:
        """
        Your task is to take the input text provided and turn it into an engaging, informative podcast dialogue. The input text may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points and interesting facts that could be discussed in a podcast.

        Important: The ENTIRE podcast dialogue (including brainstorming, scratchpad, and actual dialogue) should be written in {language}. If 'Chinese', use correct idiomatic Traditional Chinese (繁體中文) suitable for a Taiwanese audience.

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
        gr.Info("Generating dialogue script with AI...")
        llm_start_time = time.time()
        llm_output = generate_dialogue(full_text, language)
        logger.info(f"Dialogue generation took {time.time() - llm_start_time:.2f} seconds.")

    except ValidationError as e:
        logger.error(f"LLM output validation failed after retries: {e}")
        # Try to parse the raw output if possible to give user feedback
        raw_output = getattr(e, 'llm_output', str(e)) # Access raw output if promptic stores it
        raise gr.Error(f"The AI model returned an unexpected format even after retries. Please try again or simplify the input. Raw output hint: {str(raw_output)[:500]}...")
    except Exception as e:
        logger.error(f"Error during dialogue generation: {e}")
        error_str = str(e).lower()
        if "authentication" in error_str:
             raise gr.Error("Authentication error with API. Please check your API key.")
        elif "rate limit" in error_str:
             raise gr.Error("API rate limit exceeded. Please wait and try again, or check your usage tier.")
        elif "base_url" in error_str or "connection" in error_str:
             base_url = resolved_base_url or 'the configured OpenAI endpoint'
             raise gr.Error(f"Could not connect to {base_url}. Please check the URL and network connection.")
        elif "invalid request" in error_str and "image" in error_str:
             raise gr.Error("Error processing image with Vision API. The image might be invalid, unsupported, or the model doesn't support image input.")
        else:
            raise gr.Error(f"An error occurred during dialogue generation: {e}")

    if not llm_output or not llm_output.dialogue:
        raise gr.Error("The AI failed to generate a dialogue script. The input might be too short or unclear.")

    # --- Audio Generation (Order-Preserving) ---
    characters = 0
    total_lines = len(llm_output.dialogue)
    logger.info(f"Starting TTS generation for {total_lines} dialogue lines.")
    gr.Info(f"Generating audio for {total_lines} dialogue lines... (this may take a while)")

    # List to store results in order: [(transcript_line, audio_chunk_bytes | error_message), ...]
    results = [None] * total_lines

    with cf.ThreadPoolExecutor(max_workers=10) as executor: # Adjust max_workers as needed
        future_to_index = {
            executor.submit(get_mp3, line.text, line.voice, resolved_api_key): i
            for i, line in enumerate(llm_output.dialogue) if line.text.strip() # Only submit non-empty lines
        }
        # Also track characters for non-empty lines submitted
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
                if processed_count % 10 == 0 or processed_count == total_lines: # Update progress periodically
                     gr.Info(f"Generated audio for {processed_count}/{total_lines} lines...")
            except Exception as exc:
                 logger.error(f'TTS generation failed for line {index+1} after retries: {exc}')
                 error_msg = f"[TTS Error: Failed audio for line {index+1}]"
                 results[index] = (transcript_line, error_msg) # Store error marker

    logger.info(f"TTS generation took {time.time() - tts_start_time:.2f} seconds.")
    logger.info(f"Total characters for TTS: {characters}")

    # --- Combine Results in Order ---
    gr.Info("Combining audio segments...")
    final_audio_chunks = []
    final_transcript_lines = []
    successful_lines = 0
    for i, result in enumerate(results):
        line_obj = llm_output.dialogue[i] # Get original line for context even if result is None/error
        if result is None:
            # This means the line was empty or skipped submission
            if line_obj.text.strip(): # Log error only if it should have been processed
                logger.error(f"Result missing for non-empty line {i+1}. Original text: '{line_obj.text[:50]}...' Skipping.")
                final_transcript_lines.append(f"[Internal Error: Audio result missing for line {i+1}] {line_obj.speaker}: {line_obj.text}")
            continue # Skip empty lines silently

        transcript_part, audio_part = result
        final_transcript_lines.append(transcript_part) # Append transcript regardless of audio success
        if isinstance(audio_part, bytes):
            final_audio_chunks.append(audio_part)
            successful_lines += 1
        # If audio_part is the error string, it's already included in the transcript_part for that line

    if not final_audio_chunks:
        if any("[TTS Error" in line for line in final_transcript_lines):
             raise gr.Error("Failed to generate audio for all lines. Please check the transcript for details and review API key/status.")
        else:
             raise gr.Error("Failed to generate any audio, although dialogue script was created. Check TTS service status or API key.")

    audio = b"".join(final_audio_chunks)
    transcript = "\n\n".join(final_transcript_lines)

    logger.info(f"Successfully generated audio for {successful_lines}/{total_lines} lines.")

    # --- Save and Clean Up ---
    temporary_directory = "./gradio_cached_files/tmp/" # Changed directory slightly
    os.makedirs(temporary_directory, exist_ok=True)

    # Use a more robust way to get a temporary file path
    try:
        # Changed NamedTemporaryFile to save with a context manager
        # but get the name for returning to Gradio *after* it's closed
        # (otherwise Gradio might not be able to access it on some OS)
        temp_file_path = None
        with NamedTemporaryFile(
            dir=temporary_directory,
            delete=False, # Keep file for Gradio
            suffix=".mp3",
            prefix="podcast_audio_"
        ) as temp_file:
             temp_file.write(audio)
             temp_file_path = temp_file.name # Get the name while file is open

        if temp_file_path:
             logger.info(f"Audio saved to temporary file: {temp_file_path}")
        else:
             raise IOError("Temporary file path was not obtained.")

    except Exception as e:
        logger.error(f"Failed to write temporary audio file: {e}")
        raise gr.Error("Failed to save the generated audio file.")


    # Clean old files
    try:
        for file in glob.glob(f"{temporary_directory}podcast_audio_*.mp3"):
            if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60: # Older than 1 day
                try:
                    os.remove(file)
                    logger.debug(f"Removed old temp file: {file}")
                except OSError as e_rem:
                     logger.warning(f"Could not remove old temp file {file}: {e_rem}") # Log specific file error
    except Exception as e: # Catch broader errors during glob/check
        logger.warning(f"Error during old temp file cleanup: {e}")

    total_duration = time.time() - start_time
    gr.Info(f"Podcast generation complete! Total time: {total_duration:.2f} seconds.")
    return temp_file_path, transcript


# --- Gradio UI Definition ---

allowed_extensions = [
    ".txt", ".pdf", ".jpg", ".jpeg", ".png"
]

examples_dir = Path("examples")
examples = [
    [
        "Upload Files", [str(examples_dir / "Intangible cultural heritage item.pdf")], None, "English", None
    ],
    [
        "Upload Files", [str(examples_dir / "JUPAS_Guide.jpg")], None, "Chinese", None
    ],
    [
        "Upload Files", [str(examples_dir / "AI_To_Replace_Doctors_Teachers.txt")], None, "English", None
    ],
    [
        "Enter Text", None, "Artificial intelligence (AI) refers to the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction.", "English", None
    ],
]

# Ensure description/footer/head files exist or handle absence gracefully
def read_file_content(filepath: str, default: str = "") -> str:
    try:
        return Path(filepath).read_text(encoding='utf-8') # Specify encoding
    except FileNotFoundError:
        logger.warning(f"{filepath} not found, using default content.")
        return default
    except Exception as e:
         logger.error(f"Error reading file {filepath}: {e}. Using default.")
         return default


description_md = read_file_content("description.md", "Generate a podcast from text or documents.")
footer_md = read_file_content("footer.md", "")
head_html = read_file_content("head.html", "")


with gr.Blocks(theme="ocean", title="Mr.🆖 PodcastAI 🎙️🎧") as demo:
    gr.Markdown(description_md)

    with gr.Row():
        input_method_radio = gr.Radio(
            ["Upload Files", "Enter Text"],
            label="Choose Input Method",
            value="Upload Files"
        )

    # --- **REVISED UI STRUCTURE** ---
    # Group UI elements for better conditional visibility control
    # Set initial visibility directly on the group
    with gr.Group(visible=True) as file_upload_group:
        file_input = gr.Files(
            label="Upload TXT, PDF, or Image Files",
            file_types=allowed_extensions,
            file_count="multiple",
            # type="filepath" # Commented out: default usually returns temp file objects
        )

    with gr.Group(visible=False) as text_input_group: # Start hidden
        text_input = gr.Textbox(
            label="Enter Text",
            lines=10,
            placeholder="Paste or type your text here..."
        )
    # --- **END REVISED UI STRUCTURE** ---

    lang_input = gr.Radio(
            label="Podcast Language",
            choices=["English", "Chinese"],
            value="English",
        )

    # Use an Accordion for optional settings like API key
    # Define the URL for the API key page
    API_KEY_URL = "https://api.mr5ai.com"
    with gr.Accordion("Advanced Settings", open=False):
        gr.Markdown(
            f"💡 Get your Mr.🆖 AI Hub API Key [here]({API_KEY_URL})"
        )
        api_key_input = gr.Textbox(
                label="Mr.🆖 AI Hub API Key",
                type="password",
                placeholder="Enter your API Key obtained from Mr.🆖 AI Hub, in the format: sk-xxx",
        )

    submit_button = gr.Button("✨ Generate Podcast", variant="primary")

    with gr.Column(): # Outputs vertically
        audio_output = gr.Audio(label="Podcast Audio", type="filepath") # Use filepath for consistency
        transcript_output = gr.Textbox(label="Transcript", lines=15, show_copy_button=True)

    # --- **REVISED DYNAMIC UI LOGIC** ---
    def switch_input_method(choice):
        """Updates visibility and clears the inactive input field."""
        is_upload = choice == "Upload Files"
        # Determine updates for visibility based on the choice
        file_vis = is_upload
        text_vis = not is_upload
        # Determine updates for values: clear the field being hidden
        # Use gr.update() for no change, specific value for clearing
        text_val_update = gr.update(value="") if is_upload else gr.update()
        file_val_update = gr.update(value=None) if not is_upload else gr.update()

        # Return dictionary mapping components to their updates
        return {
            file_upload_group: gr.update(visible=file_vis),
            text_input_group: gr.update(visible=text_vis),
            text_input: text_val_update,
            file_input: file_val_update
        }

    input_method_radio.change(
        fn=switch_input_method,
        inputs=input_method_radio,
        # Outputs list includes the groups for visibility and the inputs for clearing
        outputs=[file_upload_group, text_input_group, text_input, file_input]
    )
    # --- **END REVISED DYNAMIC UI LOGIC** ---

    # Connect button click
    submit_button.click(
        fn=generate_audio,
        inputs=[
            input_method_radio,
            file_input,
            text_input,
            lang_input,
            api_key_input
        ],
        outputs=[audio_output, transcript_output],
        api_name="generate_podcast" # Assign API name for potential external calls
    )

    # Configure Examples
    gr.Examples(
        examples=examples,
        inputs=[input_method_radio, file_input, text_input, lang_input, api_key_input],
        outputs=[audio_output, transcript_output],
        fn=generate_audio,
        cache_examples=True, # Use lazy caching or True/False
        run_on_click=True,
        label="Examples (Click for Demo)"
    )

    gr.Markdown(footer_md)
    # Combine env var and file content for head (handle potential None env var)
    demo.head = (os.getenv("HEAD", "") or "") + head_html

# --- App Setup & Launch ---

# Queue and Mount
demo = demo.queue(
    max_size=20,
    default_concurrency_limit=5, # Limit concurrent TTS/LLM calls - Consider adjusting based on resources
)

# Mount the Gradio app to the FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Ensure examples directory exists
    examples_dir.mkdir(exist_ok=True)
    # Check/create example files (optional, prevents errors if gitignored/missing)
    example_files = [
        "Intangible cultural heritage item.pdf",
        "JUPAS_Guide.jpg",
        "AI_To_Replace_Doctors_Teachers.txt"
    ]
    for fname in example_files:
        fpath = examples_dir / fname
        if not fpath.is_file():
            logger.warning(f"Example file {fpath} not found. Creating empty placeholder.")
            try:
                fpath.touch()
            except OSError as e:
                logger.error(f"Failed to create placeholder file {fpath}: {e}")


    # Ensure temp dir exists
    os.makedirs("./gradio_cached_files/tmp/", exist_ok=True)

    logger.info("Starting Gradio application via Uvicorn...")
    # Launch using Uvicorn for better control if needed, or use demo.launch()
    # Note: demo.launch() is simpler for basic cases. Uvicorn is needed if embedding in a larger FastAPI app structure.
    # Since we already have FastAPI app (`app`), using uvicorn is appropriate here.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # Or use 127.0.0.1 for local only

    # Alternatively, for simple launch:
    # logger.info("Starting Gradio application...")
    # demo.launch(show_api=False, server_name="0.0.0.0") # Use 0.0.0.0 to be accessible on network

