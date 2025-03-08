# Perplexity Conversation
This is a custom component of Home Assistant.

Forked from [Grok Conversation](https://github.com/braytonstafford/grok_conversation)

Derived from [OpenAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/) but uses the Perplexity API URL [(https://api.perplexity.ai)](https://api.perplexity.ai).

## How it works
Perplexity Conversation uses OpenAI's python package to call to call the Perplexity API URL to interact with the Perplexity's online models; see Perplexity [documentation]([text](https://docs.perplexity.ai/home)).

## Installation
1. Install via registering as a custom repository of HACS or by copying `perplexity_conversation` folder into `<config directory>/custom_components`
2. Restart Home Assistant
3. Go to Settings > Devices & Services.
4. In the bottom right corner, select the Add Integration button.
5. Follow the instructions on screen to complete the setup (API Key is required).
    - [Generating an API Key](https://www.perplexity.ai/settings/api)
6. Go to Settings > [Voice Assistants](https://my.home-assistant.io/redirect/voice_assistants/).
7. Click to edit Assistant (named "Home Assistant" by default).
8. Select "Perplexity" from "Conversation agent" tab.

## Preparation
After installed, you need to expose entities from "https://{your-home-assistant}/config/voice-assistants/expose" for the entities/devices to be controlled by your voice assistant.
