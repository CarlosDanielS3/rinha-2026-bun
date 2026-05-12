# rinha-2026-bun

Fraud detection API for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

k-nearest neighbors (k=5) search over 3M 14-dimensional reference vectors to classify transactions as fraud or legitimate.

## Architecture

```
nginx (0.10 CPU / 15MB)
  ├── api-1 (0.45 CPU / 167MB)  ─┐
  └── api-2 (0.45 CPU / 167MB)  ─┤── in-process IVF index
                                   └── native AVX2 search (libsearch.so)
```

- **Runtime:** Bun
- **Search engine:** C with AVX2 SIMD via Bun FFI (JS fallback for local dev)
- **Index:** IVF with int16 quantization (scale=10000), nprobe=1 + bounding-box repair
- **Transport:** Unix domain sockets between nginx and API instances

## How it works

1. `POST /fraud-score` receives a transaction payload
2. The transaction is vectorized into 14 normalized dimensions
3. The nearest centroid is found among 4096 k-means clusters
4. All vectors in that cluster are scanned for the 5 nearest neighbors
5. Bounding-box repair checks remaining clusters — any cluster whose bbox lower-bound distance could beat the current 5th-best is also scanned (guarantees exact kNN accuracy)
6. The fraud score is the count of fraud labels among the 5 nearest neighbors (0–5), mapped to `{approved, fraud_score}`

## Quantization

Vectors are quantized to `int16` with a fixed scale of 10,000. This gives sufficient precision for exact-match accuracy while halving memory vs float32. Distance computations use integer arithmetic — no per-dimension weight correction needed.

## Project structure

```
src/
├── api/
│   ├── server.ts        # HTTP server (Bun.serve + FFI)
│   └── vectorize.ts     # Transaction → 14D float vector
├── index-service/
│   └── ivf-index.ts     # JS fallback IVF search
├── native/
│   └── search.c         # AVX2 SIMD search (nprobe=1 + bbox repair)
└── build-index.ts       # Build-time: references.json.gz → index.bin
```

## Running locally

```bash
# Build the index (~5 min, needs references.json.gz)
RESOURCES_DIR=../rinha-de-backend-2026/resources OUTPUT_PATH=./test-index.bin bun run src/build-index.ts

# Start the API (uses JS fallback on non-x86)
INDEX_PATH=./test-index.bin API_PORT=9999 MCC_RISK_PATH=./resources/mcc_risk.json bun run src/api/server.ts

# Test
curl -s -X POST http://localhost:9999/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{"id":"1","transaction":{"amount":500,"installments":3,"requested_at":"2025-01-15T14:30:00Z"},"customer":{"avg_amount":200,"tx_count_24h":5,"known_merchants":["m1"]},"merchant":{"id":"m2","mcc":"5411","avg_amount":300},"terminal":{"is_online":true,"card_present":false,"km_from_home":50},"last_transaction":{"timestamp":"2025-01-15T13:00:00Z","km_from_current":100}}'
```

## Running with Docker (competition mode)

```bash
docker compose up --build -d
# API available at http://localhost:9999
```

Resource limits: 1.0 CPU total, 349MB RAM total.

## Running k6 tests

```bash
cd ../rinha-de-backend-2026
k6 run test/test.js
cat test/results.json
```
