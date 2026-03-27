from openai import OpenAI
from pathlib import Path
import base64

from PIL import Image
from io import BytesIO

class Colours:
    USER = "\033[94m"       
    ASSISTANT = "\033[92m"  
    RESET = "\033[0m"
    BOLD = "\033[1m"

def load_config(path: Path) -> dict:
    """
    Loads model configurations from a text file into a dictionary.

    Parameters
    ----------
    path : str
        Path to the configuration file, relative to the root folder.

    Returns
    -------
    config : dict
        A dictionary containing the loaded model configurations.
    """

    config = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config



def main(host: str, context_length: int):
    """
    Main to run the chat application

    Parameters
    ----------
    host : str
        Host name of the model as defined in model_config.txt (consult your LM studio)
    context_length : int 
        Length of the context window (used to give warning about exceeting it)

    """
    client = OpenAI(
        base_url=host,
        api_key=""
    )

    conversation = []
    tokens_used = 0
    limit =2000
    print(f":{Colours.BOLD} You have launched a chat. To leave it, print 'exit' or 'quit'. {Colours.RESET}")

    while True:
        user_input = input(f"{Colours.BOLD}{Colours.USER}You: {Colours.RESET}")
        
        if user_input in {"exit", "quit"}:
            save_input = input("Do you want to save the conversation? (y/n): ").strip().lower()
            if save_input == "y":
                save_path = input("Enter the file path to save the conversation: ").strip()
                with open(save_path, "w") as f:
                    for message in conversation:
                        role = message["role"]
                        content = message["content"]
                        f.write(f"{role.capitalize()}: {content}\n")
                print(f"Conversation saved to {save_path}"
            )
            break

        conversation.append({"role": "user", "content": user_input})

        response = client.responses.create(
            model="local-model",
            input=conversation
        )

        reply = response.output_text
        print(f"{Colours.BOLD}{Colours.ASSISTANT}Assistant:{Colours.RESET} {reply}")

        conversation.append({"role": "assistant", "content": reply})
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens

        tokens_used += in_tokens + out_tokens

        if tokens_used >=limit:
            limit += 2000
            print(f"{Colours.BOLD}{Colours.ASSISTANT}Assistant:{Colours.RESET} Used {tokens_used}/{context_length} tokens. Responses may get slower.")
            
            
def run():
    """"
    Run an interactive chat session with the local LLM.

    The chat keeps track of the conversation history and token usage.
    Users can exit the chat and choose to save the conversation to a file.

    """
    config_path = Path(__file__).parent / "model_config.txt"
    config = load_config(config_path)
    main(
        host=config["base_url"],
        context_length=int(config["context_length"])
    )

def single_run(prompt:str)->str:
    """
    Run a single prompt through the local LLM.

    Parameters
    ----------
    prompt : str
        The prompt to send to the LLM.

    Returns
    -------
    response : str
        The response from the LLM.
    """
    
    config_path = Path(__file__).parent / "model_config.txt"
    config = load_config(config_path)
    host=config["base_url"]
    model = config["model"]
            
    client = OpenAI(
        base_url=host,
        api_key=""
    )

    response = client.responses.create(
        input=prompt,
        model=model
    )
    return response.output_text.replace("\n", "")


def encode_image(file_path: str, target_width: int = 270) -> str:
    """
    Reduce an image and turn it into a unicode compatible with a local LLM

    Parameters
    ----------
    file : str 
        Full path to the image file 
    target_width: int
        The width to which the picture needs to be reduced (in pixels). Submitting images in their original size can be resource-consuming.

    Returns
    -------
    out : str
        A Python Unicode string corresponding to the resized image
    """

    # Load the image
    with Image.open(file_path) as img:
        # Maintain aspect ratio
        w_percent = target_width / float(img.size[0])
        target_height = int(float(img.size[1]) * w_percent)

        img = img.resize((target_width, target_height), Image.LANCZOS)

        # Save the image in buffer
        buffer = BytesIO()
        img.save(buffer, format="JPEG")  
        buffer.seek(0)

        out = base64.b64encode(buffer.read()).decode("utf-8")

        return out

def single_run_multimodal(file:str, prompt:str, target_width:int=270)->str:
    """
    Run a single prompt through the local multimodal LLM. Supports image attachments

    Parameters
    ----------
    file : str 
        Full path to the image file 
    prompt : str
        The prompt to send to the LLM.
    target_width: int
        The width to which the picture needs to be reduced (in pixels). Submitting images in their original size can be resource-consuming.

    Returns
    -------
    response : str
        The response from the LLM.
    """

    config_path = Path(__file__).parent / "multimodal_model_config.txt"
    config = load_config(config_path)
    host = config["base_url"]
    model = config["model"]

    client = OpenAI(
        base_url=host,
        api_key=""
    )

    content = []

    if file is not None:
        image_base64 = encode_image(file)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })

    # Text comes AFTER the image
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return response.choices[0].message.content.strip()

def tokeniser_single_run(input_text:str)->list:
    """
    Tokenise a single prompt using the model as specified in embedding_model_config.txt. Requires an enabled tokeniser model in LM studio

    Parameters
    ----------
    input_text : str
        The prompt to send to the tokeniser.

    Returns
    -------
    out : list
        A list of vectorised entries (length specified by the selected model)
    """
    config_path = Path(__file__).parent / "embedding_model_config.txt"
    config = load_config(config_path)
    host=config["base_url"]
    model = config["model"]
            
    client = OpenAI(
        base_url=host,
        api_key=""
    )

    response = client.embeddings.create(
        input=input_text,
        model=model
    )
    out = response.data[0].embedding
    return out



if __name__ == "__main__":
    run()