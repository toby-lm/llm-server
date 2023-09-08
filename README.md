# LLM Capability

*Toby Duffy - toby@learningmachines.au*

This was spun up to provide a local LLM capability for chat, summarisation, document Q&A, hypothesis generation, and many more future use cases. This capability allows for learning and experimentation.

Currently there are 3 models configured:
* Llama 2 (13B Parameters)
* Llama 2 Chat (13B Parameters)
* AstroLlama (7B Parameters)

These models each have their own API, detailed in the swagger  page ([http://msogpib1:8800/]()).

There is also a Chat app ([http://msogpib1:3000/]()) that wraps around these APIs for a more familiar ChatGPT-like interface.

These are all deployed as Docker containers, managed by a single Docker Compose file (`docker-compose.yml`).

To start the whole service, use:
```
$ docker compose up -d
```

To stop the whole service, use:
```
$ docker compose down
```

To start up specific containers/LLMs, simply list the services you want to start:
```
$ docker compose up -d chatui llm-llama2 astrollama
```