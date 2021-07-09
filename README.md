## gpt-2-cobalt

A fork of OpenAI's gpt-2, including a Python server running on local port `7001` for receiving completion tasks from the [Cobalt bot](https://github.com/the-sink/Cobalt). The server communicates entirely in plaintext (but this may change in the future if more features are needed).

First, you need to download the model you'd like to use (124M, 345M, 774M, or 1.5B):

```
python download_model.py <model name>
```

To start the server after the model is downloaded, run the following command in this repo's working directory:

```
python src/server.py --model-name <name>
```

Replace `<name>` with the name of the model you're working with.

Applications trying to communicate with the Python server must make a POST request of type ``text/plain`` containing the prompt to be completed, and after processing the server will return the resulting completion text (including the original prompt) in plaintext as well (or an HTTP 500 if something went wrong).

For details on how to train GPT-2 on custom data, visit [this](https://medium.com/ai-innovation/beginners-guide-to-retrain-gpt-2-117m-to-generate-custom-text-content-8bb5363d8b7f) guide.
