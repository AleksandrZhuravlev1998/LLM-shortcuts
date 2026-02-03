from openai import OpenAI
from pathlib import Path

class Colours:
    USER = "\033[94m"       
    ASSISTANT = "\033[92m"  
    RESET = "\033[0m"
    BOLD = "\033[1m"

def load_config(path: Path) -> dict:
    """
    Loads model configurations from a text file into a dictionary.

    :param path: picked from the root folder
    :return: Description
    :rtype: dict
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

# Get the 
def main(host, context_length):
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

def single_run(prompt):
    """"
    Run a single prompt through the local LLM.

    :param prompt: The prompt to send to the LLM.
    :type prompt: str
    :return: The response from the LLM.
    :rtype: str
    """
    
    config_path = Path(__file__).parent / "model_config.txt"
    config = load_config(config_path)
    host=config["base_url"]
            
    client = OpenAI(
        base_url=host,
        api_key=""
    )

    response = client.responses.create(
        model="local-model",
        input=prompt
    )
    return response.output_text.replace("\n", "")



if __name__ == "__main__":
    run()