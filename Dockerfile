FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"
COPY . .
# --forwarded-allow-ips='*': Render's edge proxy connects to this container
# over the network, not from 127.0.0.1 (uvicorn's default trusted peer), so
# --proxy-headers alone silently discarded X-Forwarded-For and every
# request's client_ip (used by app/core/rate_limit.py) resolved to Render's
# own proxy address — collapsing per-IP rate limiting into one shared bucket
# for every visitor. Trusting '*' here is safe specifically because Render
# services are not directly reachable from the public internet; the only
# path in is through Render's own routing layer, which is what sets this
# header. Do not set this if the app is ever deployed somewhere a client
# could connect directly to the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
