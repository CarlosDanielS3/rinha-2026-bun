/**
 * Build-time script: processes references.json.gz into an IVF (Inverted File Index).
 *
 * Uses k-means to cluster vectors, then stores them grouped by cluster
 * with per-cluster bounding boxes for bbox_repair search.
 *
 * Quantization: int16, fixed scale = 10000 (matching top-1 approach).
 */

import { gunzipSync } from "zlib";

const DIMS = 14;
const NUM_CLUSTERS = 256;
const QUANT_SCALE = 10000;
const KMEANS_ITERATIONS = 15;
const RESOURCES_DIR = process.env.RESOURCES_DIR ?? "/resources";
const OUTPUT_PATH = process.env.OUTPUT_PATH ?? "/data/index.bin";

console.log("=== IVF Index Builder ===");
console.log(`Resources: ${RESOURCES_DIR}`);
console.log(`Output: ${OUTPUT_PATH}`);
console.log(
  `Clusters: ${NUM_CLUSTERS}, Scale: ${QUANT_SCALE}, K-means iterations: ${KMEANS_ITERATIONS}`,
);

// Step 1: Decompress and parse references
console.log("\nStep 1: Decompressing references.json.gz...");
const gzFile = Bun.file(`${RESOURCES_DIR}/references.json.gz`);
const gzBuffer = await gzFile.arrayBuffer();
const jsonBuffer = gunzipSync(Buffer.from(gzBuffer));
console.log(
  `  Decompressed: ${(jsonBuffer.length / 1024 / 1024).toFixed(1)} MB`,
);

console.log("Step 2: Parsing JSON...");
const references: Array<{ vector: number[]; label: string }> = JSON.parse(
  jsonBuffer.toString(),
);
const N = references.length;
console.log(`  Parsed ${N} reference vectors`);

// Step 3: Extract vectors and labels
console.log("Step 3: Extracting vectors and labels...");
const vectors = new Float32Array(N * DIMS);
const labels = new Uint8Array(N);

for (let i = 0; i < N; i++) {
  const ref = references[i];
  for (let d = 0; d < DIMS; d++) {
    vectors[i * DIMS + d] = ref.vector[d];
  }
  labels[i] = ref.label === "fraud" ? 1 : 0;
}
console.log(`  Vectors: ${(vectors.byteLength / 1024 / 1024).toFixed(1)} MB`);

// Step 4: K-means clustering
console.log(
  `\nStep 4: K-means clustering (${NUM_CLUSTERS} clusters, ${KMEANS_ITERATIONS} iterations)...`,
);
const startKmeans = performance.now();

// Initialize centroids via random sampling (fast, sufficient for large N)
const centroids = new Float32Array(NUM_CLUSTERS * DIMS);
const assignments = new Uint16Array(N);

console.log("  Initializing centroids (random sample)...");
{
  // Fisher-Yates partial shuffle to pick NUM_CLUSTERS unique indices
  const indices = new Uint32Array(N);
  for (let i = 0; i < N; i++) indices[i] = i;
  for (let c = 0; c < NUM_CLUSTERS; c++) {
    const j = c + Math.floor(Math.random() * (N - c));
    const tmp = indices[c];
    indices[c] = indices[j];
    indices[j] = tmp;
    const src = indices[c] * DIMS;
    const dst = c * DIMS;
    for (let d = 0; d < DIMS; d++) {
      centroids[dst + d] = vectors[src + d];
    }
  }
  console.log(`  ${NUM_CLUSTERS} centroids initialized`);
}

// K-means iterations
console.log("  Running iterations...");
const clusterSizes = new Uint32Array(NUM_CLUSTERS);
const clusterSums = new Float64Array(NUM_CLUSTERS * DIMS);

for (let iter = 0; iter < KMEANS_ITERATIONS; iter++) {
  const iterStart = performance.now();

  // Assignment step — partial distance early exit (8/14 dims first)
  let changed = 0;
  for (let i = 0; i < N; i++) {
    const iBase = i * DIMS;
    const v0 = vectors[iBase],
      v1 = vectors[iBase + 1],
      v2 = vectors[iBase + 2],
      v3 = vectors[iBase + 3];
    const v4 = vectors[iBase + 4],
      v5 = vectors[iBase + 5],
      v6 = vectors[iBase + 6],
      v7 = vectors[iBase + 7];
    const v8 = vectors[iBase + 8],
      v9 = vectors[iBase + 9],
      v10 = vectors[iBase + 10],
      v11 = vectors[iBase + 11];
    const v12 = vectors[iBase + 12],
      v13 = vectors[iBase + 13];
    let bestDist = Infinity;
    let bestCluster = 0;

    for (let c = 0; c < NUM_CLUSTERS; c++) {
      const cBase = c * DIMS;
      const d0 = v0 - centroids[cBase],
        d1 = v1 - centroids[cBase + 1];
      const d2 = v2 - centroids[cBase + 2],
        d3 = v3 - centroids[cBase + 3];
      const d4 = v4 - centroids[cBase + 4],
        d5 = v5 - centroids[cBase + 5];
      const d6 = v6 - centroids[cBase + 6],
        d7 = v7 - centroids[cBase + 7];
      const partial =
        d0 * d0 +
        d1 * d1 +
        d2 * d2 +
        d3 * d3 +
        d4 * d4 +
        d5 * d5 +
        d6 * d6 +
        d7 * d7;
      if (partial >= bestDist) continue;
      const d8 = v8 - centroids[cBase + 8],
        d9 = v9 - centroids[cBase + 9];
      const d10 = v10 - centroids[cBase + 10],
        d11 = v11 - centroids[cBase + 11];
      const d12 = v12 - centroids[cBase + 12],
        d13 = v13 - centroids[cBase + 13];
      const dist =
        partial +
        d8 * d8 +
        d9 * d9 +
        d10 * d10 +
        d11 * d11 +
        d12 * d12 +
        d13 * d13;
      if (dist < bestDist) {
        bestDist = dist;
        bestCluster = c;
      }
    }

    if (assignments[i] !== bestCluster) {
      assignments[i] = bestCluster;
      changed++;
    }
  }

  // Update step
  clusterSizes.fill(0);
  clusterSums.fill(0);

  for (let i = 0; i < N; i++) {
    const cluster = assignments[i];
    clusterSizes[cluster]++;
    const iBase = i * DIMS;
    const cBase = cluster * DIMS;
    for (let d = 0; d < DIMS; d++) {
      clusterSums[cBase + d] += vectors[iBase + d];
    }
  }

  for (let c = 0; c < NUM_CLUSTERS; c++) {
    const size = clusterSizes[c];
    if (size === 0) continue;
    const cBase = c * DIMS;
    for (let d = 0; d < DIMS; d++) {
      centroids[cBase + d] = clusterSums[cBase + d] / size;
    }
  }

  const iterTime = performance.now() - iterStart;
  const pctChanged = ((changed / N) * 100).toFixed(2);
  console.log(
    `  Iteration ${iter + 1}/${KMEANS_ITERATIONS}: ${changed} changes (${pctChanged}%) (${(iterTime / 1000).toFixed(1)}s)`,
  );

  if (changed === 0 || changed < N * 0.0005) {
    console.log("  Converged!");
    break;
  }
}

const kmeansTime = performance.now() - startKmeans;
console.log(`  K-means completed in ${(kmeansTime / 1000).toFixed(1)}s`);

// Log cluster size stats
let minSize = Infinity,
  maxSize = 0;
for (let c = 0; c < NUM_CLUSTERS; c++) {
  if (clusterSizes[c] < minSize) minSize = clusterSizes[c];
  if (clusterSizes[c] > maxSize) maxSize = clusterSizes[c];
}
console.log(
  `  Cluster sizes: min=${minSize}, max=${maxSize}, avg=${(N / NUM_CLUSTERS).toFixed(0)}`,
);

// Step 5: Reorder vectors by cluster
console.log("\nStep 5: Building inverted lists...");

const clusterOffsets = new Uint32Array(NUM_CLUSTERS + 1);
clusterOffsets[0] = 0;
for (let c = 0; c < NUM_CLUSTERS; c++) {
  clusterOffsets[c + 1] = clusterOffsets[c] + clusterSizes[c];
}

const orderedVectors = new Float32Array(N * DIMS);
const orderedLabels = new Uint8Array(N);
const writePos = new Uint32Array(NUM_CLUSTERS);
writePos.set(clusterOffsets.subarray(0, NUM_CLUSTERS));

for (let i = 0; i < N; i++) {
  const cluster = assignments[i];
  const pos = writePos[cluster]++;
  const srcBase = i * DIMS;
  const dstBase = pos * DIMS;
  for (let d = 0; d < DIMS; d++) {
    orderedVectors[dstBase + d] = vectors[srcBase + d];
  }
  orderedLabels[pos] = labels[i];
}

// Step 6: Quantize vectors to int16 with fixed scale and compute per-cluster bounding boxes
console.log(
  `\nStep 6: Quantizing vectors to int16 (scale=${QUANT_SCALE}) + computing bounding boxes...`,
);

function quantize(value: number): number {
  const scaled = Math.round(value * QUANT_SCALE);
  if (scaled < -32768) return -32768;
  if (scaled > 32767) return 32767;
  return scaled;
}

const quantizedVectors = new Int16Array(N * DIMS);
for (let i = 0; i < N; i++) {
  const base = i * DIMS;
  for (let d = 0; d < DIMS; d++) {
    quantizedVectors[base + d] = quantize(orderedVectors[base + d]);
  }
}

// Compute per-cluster bounding boxes (min/max per dim per cluster)
const bboxMin = new Int16Array(NUM_CLUSTERS * DIMS);
const bboxMax = new Int16Array(NUM_CLUSTERS * DIMS);
bboxMin.fill(32767); // will be min'd down
bboxMax.fill(-32768); // will be max'd up

for (let c = 0; c < NUM_CLUSTERS; c++) {
  const start = clusterOffsets[c];
  const end = clusterOffsets[c + 1];
  if (start === end) {
    // empty cluster
    const cb = c * DIMS;
    for (let d = 0; d < DIMS; d++) {
      bboxMin[cb + d] = 0;
      bboxMax[cb + d] = 0;
    }
    continue;
  }
  for (let i = start; i < end; i++) {
    const vb = i * DIMS;
    const cb = c * DIMS;
    for (let d = 0; d < DIMS; d++) {
      const v = quantizedVectors[vb + d];
      if (v < bboxMin[cb + d]) bboxMin[cb + d] = v;
      if (v > bboxMax[cb + d]) bboxMax[cb + d] = v;
    }
  }
}

// Log some stats
let totalBboxVolume = 0;
for (let c = 0; c < NUM_CLUSTERS; c++) {
  let vol = 1;
  const cb = c * DIMS;
  for (let d = 0; d < DIMS; d++) {
    vol *= Math.max(1, bboxMax[cb + d] - bboxMin[cb + d]);
  }
  totalBboxVolume += Math.log(vol);
}
console.log(
  `  Avg log bbox volume: ${(totalBboxVolume / NUM_CLUSTERS).toFixed(1)}`,
);

// Step 7: Serialize
// Format: header(16B) + centroids + bboxMin + bboxMax + offsets + vectors(int16) + labels
console.log("\nStep 7: Writing index.bin...");

const headerSize = 4 * 4; // N, NC, DIMS, SCALE
const centroidsSize = NUM_CLUSTERS * DIMS * 4; // float32
const bboxSize = NUM_CLUSTERS * DIMS * 2; // int16 per bbox
const offsetsSize = (NUM_CLUSTERS + 1) * 4;
const vectorsSize = N * DIMS * 2; // int16
const labelsSize = N;
const totalFileSize =
  headerSize +
  centroidsSize +
  bboxSize * 2 +
  offsetsSize +
  vectorsSize +
  labelsSize;

console.log(
  `  Total file size: ${(totalFileSize / 1024 / 1024).toFixed(1)} MB`,
);

const outputBuffer = Buffer.alloc(totalFileSize);
let offset = 0;

outputBuffer.writeUInt32LE(N, offset);
offset += 4;
outputBuffer.writeUInt32LE(NUM_CLUSTERS, offset);
offset += 4;
outputBuffer.writeUInt32LE(DIMS, offset);
offset += 4;
outputBuffer.writeUInt32LE(QUANT_SCALE, offset);
offset += 4;

Buffer.from(centroids.buffer).copy(outputBuffer, offset);
offset += centroidsSize;
Buffer.from(bboxMin.buffer).copy(outputBuffer, offset);
offset += bboxSize;
Buffer.from(bboxMax.buffer).copy(outputBuffer, offset);
offset += bboxSize;
Buffer.from(clusterOffsets.buffer).copy(outputBuffer, offset);
offset += offsetsSize;
Buffer.from(quantizedVectors.buffer).copy(outputBuffer, offset);
offset += vectorsSize;
Buffer.from(orderedLabels.buffer).copy(outputBuffer, offset);

await Bun.write(OUTPUT_PATH, outputBuffer);
console.log(`  Written to ${OUTPUT_PATH}`);
console.log("\n=== Done! ===");
