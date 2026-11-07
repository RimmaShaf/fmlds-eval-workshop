FROM ghcr.io/berriai/litellm:main-stable

COPY infra/litellm_config.yaml /app/config.yaml

CMD ["--config", "/app/config.yaml", "--port", "8080"]
