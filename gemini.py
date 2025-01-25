import os
import time
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file


def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()


# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}


def run_gemini(file_path):
    model = genai.GenerativeModel(
        model_name="gemini-exp-1206",
        generation_config=generation_config,
        system_instruction="You are now my research assistant and our research topic is video emotion recognition, you need to determine the emotion of a task by the expression and demeanor of the task in the video I send you, the emotion is one of the following categories: angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted. ",
    )

    # TODO Make these files available on the local file system
    # You may need to update the file paths
    files = [
        upload_to_gemini(f"{file_path}", mime_type="video/mp4"),
    ]

    # Some files have a processing delay. Wait for them to be ready.
    wait_for_files_active(files)

    input = [files[0], "\n\n", "Directly tell me what emotion the person in the video is feeling right now through face. And explain why you made that judgment after the first sentence of the emotion you provide."]
    response = chat_session.send_message("INSERT_INPUT_HERE")

    print(response.text)
