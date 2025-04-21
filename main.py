import concurrent.futures as cf
import glob
import io
import os
import time
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal, Optional

import gradio as gr
import sentry_sdk
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type
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
            "female-1": "onyx",
            "male-1": "alloy",
            "female-2": "fable",
            "male-2": "echo",
        }[self.speaker]


class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]


def get_mp3(text: str, voice: str, api_key: str = None) -> bytes:
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()


def is_pdf(filename):
    # Added check for None filename
    if not filename: return False
    t, _ = guess_type(filename)
    return filename.lower().endswith(".pdf") or (t or "").endswith("pdf")

def is_image(filename):
    # Added check for None filename
    if not filename: return False
    t, _ = guess_type(filename)
    image_exts = (".jpg", ".jpeg", ".png")
    return filename.lower().endswith(image_exts) or (t or "").startswith("image")

def is_text(filename):
    # Added check for None filename
    if not filename: return False
    t, _ = guess_type(filename)
    return filename.lower().endswith(".txt") or (t or "") == "text/plain"


def extract_text_from_image_via_vision(image_file, openai_api_key=None):
    client = OpenAI(
        api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
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
                    "image_url": {
                        "url": image_url,
                        "detail": "auto"
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
        model="gpt-4.1-mini", # Using a Vision-capable model
        messages=messages,
        max_tokens=4096, # Adjusted max_tokens slightly lower than 8192
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def generate_audio(
    input_method: str,
    files: Optional[List[str]],
    input_text: Optional[str],
    language: str = "English",
    openai_api_key: str = None,
) -> (str, str):
    """Generates podcast audio from either uploaded files or direct text input."""
    if not (os.getenv("OPENAI_API_KEY") or openai_api_key):
        raise gr.Error("OpenAI API key is required")
    # Base URL check removed as it might not always be needed if default is used

    full_text = ""

    if input_method == "Upload Files":
        if not files:
            raise gr.Error("Please upload at least one file or switch to text input.")
        # No need for isinstance check, gr.Files(file_count='multiple') always returns a list
        texts = []
        for file_path in files:
            # Add explicit check for file_path being None or empty string
            if not file_path:
                logger.warning("Received an empty file path in the list, skipping.")
                continue
            file_path_obj = Path(file_path) # Work with Path object for clarity
            if is_pdf(str(file_path_obj)):
                try:
                    with file_path_obj.open("rb") as f:
                        reader = PdfReader(f)
                        # Added check for encrypted PDFs
                        if reader.is_encrypted:
                             logger.warning(f"Skipping encrypted PDF: {file_path_obj.name}")
                             raise gr.Error(f"Cannot process password-protected PDF: {file_path_obj.name}")
                        page_texts = [page.extract_text() for page in reader.pages if page.extract_text()]
                        text = "\n\n".join(page_texts) if page_texts else ""
                except Exception as e:
                    logger.error(f"Error reading PDF {file_path_obj.name}: {e}")
                    # Provide more specific error for common issues
                    if "PdfReadError" in str(type(e)):
                         raise gr.Error(f"Error reading PDF file: {file_path_obj.name}. It might be corrupted or improperly formatted.")
                    else:
                         raise gr.Error(f"Error processing PDF file: {file_path_obj.name}.")
            elif is_image(str(file_path_obj)):
                try:
                    text = extract_text_from_image_via_vision(str(file_path_obj), openai_api_key)
                except Exception as e:
                    logger.error(f"Error processing image {file_path_obj.name} with Vision API: {e}")
                    raise gr.Error(f"Error extracting text from image: {file_path_obj.name}. Check API key, file format, and OpenAI status.")
            elif is_text(str(file_path_obj)):
                try:
                    with open(file_path_obj, "r", encoding="utf-8") as f:
                        text = f.read()
                except Exception as e:
                    logger.error(f"Error reading text file {file_path_obj.name}: {e}")
                    raise gr.Error(f"Error reading text file: {file_path_obj.name}.")
            else:
                # Check if it's an empty placeholder or truly unsupported
                if file_path_obj.stat().st_size > 0 :
                    raise gr.Error(f"Unsupported file type: {file_path_obj.name}. Please upload TXT, PDF, or image file (JPG, JPEG, PNG).")
                else:
                     logger.warning(f"Skipping empty or placeholder file: {file_path_obj.name}")
                     text = "" # Treat as empty text
            texts.append(text)
        full_text = "\n\n".join(filter(None, texts)) # Filter out empty strings before joining
        if not full_text.strip():
             raise gr.Error("Could not extract any text from the uploaded file(s). Please check the files or try different ones.")

    elif input_method == "Enter Text":
        if not input_text or not input_text.strip():
            raise gr.Error("Please enter text or switch to file upload.")
        full_text = input_text
    else:
        raise gr.Error("Invalid input method selected.")


    @retry(retry=retry_if_exception_type(ValidationError))
    @llm(
        model="gpt-4.1-mini", # Use the newer model for better performance potentially
        api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    def generate_dialogue(text: str, language: str) -> Dialogue:
        """
        Your task is to take the input text provided and turn it into an engaging, informative podcast dialogue. The input text may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points and interesting facts that could be discussed in a podcast.

        Important: The ENTIRE podcast dialogue (including brainstorming, scratchpad, and actual dialogue) should be written in {language}. If 'Traditional Chinese', use correct idiomatic Traditional Chinese (繁體中文) suitable for a Taiwanese audience.

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
        Write your engaging, informative podcast dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience. Use made-up names for the hosts and guests to create a more engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio. Assign appropriate speakers (female-1, male-1, female-2, male-2) to each line, varying them for a natural conversation.

        Make the dialogue as long and detailed as possible, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

        At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.
        </podcast_dialogue>
        """

    try:
        # Add progress update before calling LLM
        gr.Info("Generating dialogue script with AI...")
        llm_output = generate_dialogue(full_text, language)
    except ValidationError as e:
        logger.error(f"LLM output validation failed: {e}")
        raise gr.Error(f"The AI model returned an unexpected format. Please try again or rephrase your input. Details: {e}")
    except Exception as e:
        logger.error(f"Error during dialogue generation: {e}")
        # Check for common API errors
        error_str = str(e).lower()
        if "authentication" in error_str:
             raise gr.Error("Authentication error with OpenAI API. Please check your API key.")
        elif "rate limit" in error_str:
             raise gr.Error("OpenAI API rate limit exceeded. Please wait and try again.")
        elif "base_url" in error_str or "connection" in error_str:
             base_url = os.getenv('OPENAI_BASE_URL', 'the configured OpenAI endpoint')
             raise gr.Error(f"Could not connect to {base_url}. Please check the URL and network connection.")
        elif "invalid request" in error_str and "image" in error_str:
             raise gr.Error("Error processing image with Vision API. The image might be invalid, unsupported, or the model doesn't support image input.")
        else:
            raise gr.Error(f"An error occurred during dialogue generation: {e}")


    audio = b""
    transcript = ""
    characters = 0
    total_lines = len(llm_output.dialogue)

    # Add progress update before starting TTS
    gr.Info(f"Generating audio for {total_lines} dialogue lines...")

    with cf.ThreadPoolExecutor() as executor:
        futures_map = {} # Store future -> transcript_line mapping
        for i, line in enumerate(llm_output.dialogue):
            transcript_line = f"{line.speaker}: {line.text}"
            try:
                future = executor.submit(get_mp3, line.text, line.voice, openai_api_key)
                futures_map[future] = transcript_line
                characters += len(line.text)
            except Exception as e:
                logger.error(f"Error submitting TTS task for line {i+1}/{total_lines} '{line.text[:50]}...': {e}")
                # Add error note directly to transcript
                transcript += f"[Error submitting audio task for: {transcript_line}]\n\n"

        # Process completed futures as they finish
        processed_lines = 0
        for future in cf.as_completed(futures_map):
            transcript_line = futures_map[future]
            try:
                audio_chunk = future.result()
                audio += audio_chunk
                transcript += transcript_line + "\n\n"
                processed_lines += 1
                # Optional: Update progress more frequently
                # if processed_lines % 5 == 0: # Update every 5 lines
                #     gr.Info(f"Generated audio for {processed_lines}/{total_lines} lines...")
            except Exception as e:
                 logger.error(f"Error retrieving TTS result for line '{transcript_line[:50]}...': {e}")
                 # Add error note to transcript if retrieval fails
                 transcript += f"[Error generating audio for: {transcript_line}]\n\n"

    logger.info(f"Generated {characters} characters of audio for {processed_lines}/{total_lines} lines.")

    if not audio:
        # Check if transcript has error messages
        if "[Error" in transcript:
             raise gr.Error("Failed to generate audio for some/all lines. Please check the transcript for details and review OpenAI API key/status.")
        else:
             raise gr.Error("Failed to generate any audio, even though dialogue script was created. Check OpenAI TTS service status or API key.")


    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False, # Keep file until Gradio cleans it up
        suffix=".mp3",
        prefix="podcast_audio_" # Add prefix for easier identification
    )
    try:
        temporary_file.write(audio)
        temp_file_path = temporary_file.name
    finally:
        temporary_file.close()


    # Clean old files (optional - Gradio might handle temp files, but good practice)
    for file in glob.glob(f"{temporary_directory}*.mp3"):
        try:
            if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60: # Older than 1 day
                os.remove(file)
        except OSError as e:
            logger.warning(f"Could not remove old temp file {file}: {e}")

    gr.Info("Podcast generation complete!")
    return temp_file_path, transcript


allowed_extensions = [
    ".txt", ".pdf", ".jpg", ".jpeg", ".png"
]

# ----
# EXAMPLES SECTION: Format: [input_method, files, input_text, language, openai_api_key]
# Use None or "" for unused fields (files when input_method is text, input_text when method is files).
# Leave openai_api_key as None to use environment variable or prompt user.
# Ensure example files exist in the "examples" directory.
examples_dir = Path("examples")
examples = [
    [
        "Upload Files", [str(examples_dir / "Intangible cultural heritage item.pdf")], None, "English", None
    ],
    [
        "Upload Files", [str(examples_dir / "JUPAS_Guide.jpg")], None, "Traditional Chinese", None
    ],
    [
        "Upload Files", [str(examples_dir / "AI_To_Replace_Doctors_Teachers.txt")], None, "English", None
    ],
    [
        "Enter Text", None, "Artificial intelligence (AI) refers to the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction.", "English", None
    ],
]
# ----

# Gradio Interface using Blocks for dynamic UI
# Added title argument here
with gr.Blocks(theme="ocean", title="Mr.🆖 PodcastAI 🎙️🎧") as demo:
    gr.Markdown(Path("description.md").read_text())

    with gr.Row():
        input_method_radio = gr.Radio(
            ["Upload Files", "Enter Text"],
            label="Choose Input Method",
            value="Upload Files" # Default selection
        )

    with gr.Column(visible=True) as file_upload_ui: # Initially visible
         file_input = gr.Files(
            label="Upload TXT, PDF, or Image Files",
            file_types=allowed_extensions,
            file_count="multiple",
        )

    with gr.Column(visible=False) as text_input_ui: # Initially hidden
        text_input = gr.Textbox(
            label="Enter Text",
            lines=10,
            placeholder="Paste or type your text here..."
        )

    lang_input = gr.Radio(
            label="Podcast Language",
            choices=["English", "Traditional Chinese"],
            value="English",
        )

    api_key_input = gr.Textbox(
            label="OpenAI API Key",
            type="password",
            placeholder="Enter your OpenAI API key here",
            visible=not os.getenv("OPENAI_API_KEY"), # Only show if not in env
        )

    # Added variant="primary" for button color
    submit_button = gr.Button("✨ Generate Podcast", variant="primary")

    # Outputs arranged vertically (removed gr.Row wrapper)
    audio_output = gr.Audio(label="Podcast Audio", format="mp3")
    transcript_output = gr.Textbox(label="Transcript", lines=15, show_copy_button=True) # Added copy button

    # Dynamic UI Logic
    def switch_input_method(choice):
        if choice == "Upload Files":
            return {
                file_upload_ui: gr.update(visible=True),
                text_input_ui: gr.update(visible=False),
                # Clear other input type when switching
                text_input: gr.update(value="")
            }
        else: # Enter Text
            return {
                file_upload_ui: gr.update(visible=False),
                text_input_ui: gr.update(visible=True),
                # Clear other input type when switching
                file_input: gr.update(value=None)
            }

    input_method_radio.change(
        fn=switch_input_method,
        inputs=input_method_radio,
        outputs=[file_upload_ui, text_input_ui, file_input, text_input] # Added file/text inputs to clear
    )

    # Connect button click to the main function
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
        api_name=False # Keep API disabled if not needed
    )

    gr.Examples(
        examples=examples,
        inputs=[input_method_radio, file_input, text_input, lang_input, api_key_input], # Ensure order matches function signature
        outputs=[audio_output, transcript_output],
        fn=generate_audio, # Make examples clickable
        cache_examples=True, # Cache example results for speed
        # Prevent examples running automatically on load if inputs change slightly:
        run_on_click=True,
    )

    gr.Markdown(Path("footer.md").read_text())
    demo.head = os.getenv("HEAD", "") + Path("head.html").read_text()
    # Removed flagging_mode='never' as it's default, remove clear_btn=None as it's also default

# Queue and Mount
demo = demo.queue(
    max_size=20,
    default_concurrency_limit=5, # Adjusted concurrency limit for stability
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Ensure examples directory exists for file uploads
    examples_dir.mkdir(exist_ok=True)
    # Add dummy files if they don't exist, to prevent errors on startup if examples are missing
    # Check and create files only if they absolutely don't exist
    if not (examples_dir / "Intangible cultural heritage item.pdf").is_file(): (examples_dir / "Intangible cultural heritage item.pdf").touch()
    if not (examples_dir / "JUPAS_Guide.jpg").is_file(): (examples_dir / "JUPAS_Guide.jpg").touch()
    if not (examples_dir / "AI_To_Replace_Doctors_Teachers.txt").is_file(): (examples_dir / "AI_To_Replace_Doctors_Teachers.txt").touch()

    # Ensure temp dir exists before launch
    os.makedirs("./gradio_cached_examples/tmp/", exist_ok=True)

    logger.info("Starting Gradio application...")
    demo.launch(show_api=False)
