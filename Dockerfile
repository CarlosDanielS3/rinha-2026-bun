# --- Stage: compile native AVX2 search module ---
FROM alpine:3.20 AS native
RUN apk add --no-cache gcc musl-dev
WORKDIR /build
COPY src/native/search.c .
RUN gcc -O3 -march=haswell -shared -fPIC -o libsearch.so search.c

# --- Stage: builder ---
FROM oven/bun:1-alpine AS builder

WORKDIR /app

COPY package.json ./
COPY tsconfig.json ./
COPY src/ ./src/

RUN bun install --frozen-lockfile || bun install

# --- Stage: api (loads index in-process) ---
FROM oven/bun:1-alpine AS api

WORKDIR /app

COPY --from=native /build/libsearch.so /app/libsearch.so
COPY index.bin /data/index.bin
COPY src/api/ ./src/api/
COPY src/index-service/ivf-index.ts ./src/index-service/ivf-index.ts
COPY resources/mcc_risk.json ./resources/mcc_risk.json

ENV INDEX_PATH=/data/index.bin
ENV API_PORT=9999

CMD ["bun", "run", "src/api/server.ts"]
