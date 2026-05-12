import { vectorize, type TransactionPayload } from "./vectorize";

const PORT = parseInt(process.env.API_PORT ?? "9999");
const INDEX_PATH = process.env.INDEX_PATH ?? "/data/index.bin";
const SOCKET_PATH = process.env.SOCKET_PATH;
const LIB_PATH = process.env.LIB_SEARCH_PATH ?? "/app/libsearch.so";

const FRAUD_BODIES = [
  '{"approved":true,"fraud_score":0.0}',
  '{"approved":true,"fraud_score":0.2}',
  '{"approved":true,"fraud_score":0.4}',
  '{"approved":false,"fraud_score":0.6}',
  '{"approved":false,"fraud_score":0.8}',
  '{"approved":false,"fraud_score":1.0}',
];
const RESP_HEADERS = { "Content-Type": "application/json" };

console.log(`Loading index from ${INDEX_PATH}...`);
const startLoad = performance.now();

let searchFn: (query: Float32Array) => number;

try {
  // Native AVX2 path via Bun FFI
  const { dlopen, FFIType, ptr } = await import("bun:ffi");
  const indexBuf = new Uint8Array(await Bun.file(INDEX_PATH).arrayBuffer());

  const lib = dlopen(LIB_PATH, {
    ivf_init: { args: [FFIType.ptr, FFIType.u32], returns: FFIType.void },
    ivf_search: { args: [FFIType.ptr], returns: FFIType.i32 },
  });

  lib.symbols.ivf_init(ptr(indexBuf), indexBuf.length);
  // Pin buffer so GC doesn't collect it while C holds pointers into it
  (globalThis as any).__indexBuf = indexBuf;

  searchFn = (query: Float32Array) =>
    lib.symbols.ivf_search(ptr(query)) as number;
  console.log("  Native AVX2 search engine loaded");
} catch {
  // JS fallback (local dev, non-x86, missing .so)
  const { IVFIndex } = await import("../index-service/ivf-index");
  const buf = await Bun.file(INDEX_PATH).arrayBuffer();
  const index = IVFIndex.fromBuffer(buf);
  searchFn = (query: Float32Array) => index.search(query);
  console.log("  JS search engine (native unavailable)");
}

console.log(`Index loaded in ${(performance.now() - startLoad).toFixed(0)}ms`);

// Warmup — prime CPU caches and branch predictor
const dummy = new Float32Array(14);
for (let w = 0; w < 200; w++) {
  for (let d = 0; d < 14; d++) dummy[d] = Math.random();
  searchFn(dummy);
}
console.log("  Warmup: 200 queries done");

const serverOpts: any = {
  fetch(req: Request) {
    const url = req.url;
    // Works for both http://host:port/path and unix socket URLs
    let pi = url.indexOf("//");
    if (pi !== -1) {
      pi = url.indexOf("/", pi + 2);
    } else {
      pi = url.indexOf("/");
    }
    const path = pi !== -1 ? url.substring(pi) : "/";

    if (path === "/fraud-score") {
      return handleFraudScore(req);
    }

    if (path === "/ready") {
      return new Response("OK");
    }

    return new Response("Not Found", { status: 404 });
  },
};

if (SOCKET_PATH) {
  try {
    const { unlinkSync } = await import("fs");
    unlinkSync(SOCKET_PATH);
  } catch {}
  serverOpts.unix = SOCKET_PATH;
  console.log(`API server listening on unix:${SOCKET_PATH}`);
} else {
  serverOpts.port = PORT;
  console.log(`API server listening on port ${PORT}`);
}

Bun.serve(serverOpts);

// Make socket writable by nginx worker (runs as 'nginx' user)
if (SOCKET_PATH) {
  const { chmodSync } = await import("fs");
  chmodSync(SOCKET_PATH, 0o777);
}

async function handleFraudScore(req: Request): Promise<Response> {
  const payload: TransactionPayload = await req.json();
  const vector = vectorize(payload);
  const fraudCount = searchFn(vector);
  return new Response(FRAUD_BODIES[fraudCount], {
    headers: RESP_HEADERS,
  });
}
