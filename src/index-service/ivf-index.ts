/**
 * IVF (Inverted File Index) with int16 quantized vectors.
 * nprobe=1 + bounding-box repair for exact-quality kNN search.
 * JS fallback for local dev (native AVX2 module used in Docker).
 */

const QUANT_SCALE = 10000;

export class IVFIndex {
  private centroids: Float32Array;
  private bboxMin: Int16Array;
  private bboxMax: Int16Array;
  private clusterOffsets: Uint32Array;
  private vectors: Int16Array;
  private labels: Uint8Array;
  private numClusters: number;
  private numVectors: number;

  constructor(
    centroids: Float32Array,
    bboxMin: Int16Array,
    bboxMax: Int16Array,
    clusterOffsets: Uint32Array,
    vectors: Int16Array,
    labels: Uint8Array,
    numClusters: number,
    numVectors: number,
  ) {
    this.centroids = centroids;
    this.bboxMin = bboxMin;
    this.bboxMax = bboxMax;
    this.clusterOffsets = clusterOffsets;
    this.vectors = vectors;
    this.labels = labels;
    this.numClusters = numClusters;
    this.numVectors = numVectors;
  }

  static fromBuffer(buffer: ArrayBuffer): IVFIndex {
    const view = new DataView(buffer);
    let offset = 0;

    const numVectors = view.getUint32(offset, true);
    offset += 4;
    const numClusters = view.getUint32(offset, true);
    offset += 4;
    const dims = view.getUint32(offset, true);
    offset += 4;
    const scale = view.getUint32(offset, true);
    offset += 4;

    const centroids = new Float32Array(buffer, offset, numClusters * dims);
    offset += numClusters * dims * 4;

    const bboxMin = new Int16Array(buffer, offset, numClusters * dims);
    offset += numClusters * dims * 2;

    const bboxMax = new Int16Array(buffer, offset, numClusters * dims);
    offset += numClusters * dims * 2;

    const clusterOffsets = new Uint32Array(buffer, offset, numClusters + 1);
    offset += (numClusters + 1) * 4;

    const vectors = new Int16Array(buffer, offset, numVectors * dims);
    offset += numVectors * dims * 2;

    const labels = new Uint8Array(buffer, offset, numVectors);

    const indexMB = (buffer.byteLength / 1024 / 1024).toFixed(1);
    console.log(
      `  IVF: ${numVectors} vectors, ${numClusters} clusters, ${indexMB}MB (int16, scale=${scale})`,
    );
    console.log(`  Strategy: nprobe=1 + bbox_repair`);

    return new IVFIndex(
      centroids,
      bboxMin,
      bboxMax,
      clusterOffsets,
      vectors,
      labels,
      numClusters,
      numVectors,
    );
  }

  search(query: Float32Array): number {
    const numClusters = this.numClusters;
    const centroids = this.centroids;
    const vectors = this.vectors;
    const labels = this.labels;
    const clusterOffsets = this.clusterOffsets;
    const bboxMin = this.bboxMin;
    const bboxMax = this.bboxMax;

    const qf0 = query[0],
      qf1 = query[1],
      qf2 = query[2],
      qf3 = query[3];
    const qf4 = query[4],
      qf5 = query[5],
      qf6 = query[6],
      qf7 = query[7];
    const qf8 = query[8],
      qf9 = query[9],
      qf10 = query[10],
      qf11 = query[11];
    const qf12 = query[12],
      qf13 = query[13];

    // Step 1: Find nearest centroid
    let bestDist = Infinity;
    let bestCluster = 0;
    for (let c = 0; c < numClusters; c++) {
      const b = c * 14;
      const d0 = qf0 - centroids[b],
        d1 = qf1 - centroids[b + 1];
      const d2 = qf2 - centroids[b + 2],
        d3 = qf3 - centroids[b + 3];
      const d4 = qf4 - centroids[b + 4],
        d5 = qf5 - centroids[b + 5];
      const d6 = qf6 - centroids[b + 6],
        d7 = qf7 - centroids[b + 7];
      const d8 = qf8 - centroids[b + 8],
        d9 = qf9 - centroids[b + 9];
      const d10 = qf10 - centroids[b + 10],
        d11 = qf11 - centroids[b + 11];
      const d12 = qf12 - centroids[b + 12],
        d13 = qf13 - centroids[b + 13];
      const dist =
        d0 * d0 +
        d1 * d1 +
        d2 * d2 +
        d3 * d3 +
        d4 * d4 +
        d5 * d5 +
        d6 * d6 +
        d7 * d7 +
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

    // Step 2: Quantize query to int16
    const q0 = Math.round(qf0 * QUANT_SCALE);
    const q1 = Math.round(qf1 * QUANT_SCALE);
    const q2 = Math.round(qf2 * QUANT_SCALE);
    const q3 = Math.round(qf3 * QUANT_SCALE);
    const q4 = Math.round(qf4 * QUANT_SCALE);
    const q5 = Math.round(qf5 * QUANT_SCALE);
    const q6 = Math.round(qf6 * QUANT_SCALE);
    const q7 = Math.round(qf7 * QUANT_SCALE);
    const q8 = Math.round(qf8 * QUANT_SCALE);
    const q9 = Math.round(qf9 * QUANT_SCALE);
    const q10 = Math.round(qf10 * QUANT_SCALE);
    const q11 = Math.round(qf11 * QUANT_SCALE);
    const q12 = Math.round(qf12 * QUANT_SCALE);
    const q13 = Math.round(qf13 * QUANT_SCALE);

    // Step 3: Scan nearest cluster
    const heapDist = [Infinity, Infinity, Infinity, Infinity, Infinity];
    const heapLabel = [0, 0, 0, 0, 0];
    let heapSize = 0;
    let tau = Infinity;

    const start0 = clusterOffsets[bestCluster];
    const end0 = clusterOffsets[bestCluster + 1];
    for (let i = start0; i < end0; i++) {
      const b = i * 14;
      const d0 = q0 - vectors[b],
        d1 = q1 - vectors[b + 1];
      const d2 = q2 - vectors[b + 2],
        d3 = q3 - vectors[b + 3];
      const d4 = q4 - vectors[b + 4],
        d5 = q5 - vectors[b + 5];
      const d6 = q6 - vectors[b + 6],
        d7 = q7 - vectors[b + 7];

      const partial =
        d0 * d0 +
        d1 * d1 +
        d2 * d2 +
        d3 * d3 +
        d4 * d4 +
        d5 * d5 +
        d6 * d6 +
        d7 * d7;
      if (heapSize === 5 && partial > tau) continue;

      const d8 = q8 - vectors[b + 8],
        d9 = q9 - vectors[b + 9];
      const d10 = q10 - vectors[b + 10],
        d11 = q11 - vectors[b + 11];
      const d12 = q12 - vectors[b + 12],
        d13 = q13 - vectors[b + 13];
      const dist =
        partial +
        d8 * d8 +
        d9 * d9 +
        d10 * d10 +
        d11 * d11 +
        d12 * d12 +
        d13 * d13;

      if (heapSize < 5) {
        heapDist[heapSize] = dist;
        heapLabel[heapSize] = labels[i];
        heapSize++;
        if (heapSize === 5) {
          this.buildHeap(heapDist, heapLabel);
          tau = heapDist[0];
        }
      } else if (dist < tau) {
        heapDist[0] = dist;
        heapLabel[0] = labels[i];
        this.siftDown(heapDist, heapLabel);
        tau = heapDist[0];
      }
    }

    // Step 4: bbox repair
    for (let c = 0; c < numClusters; c++) {
      if (c === bestCluster) continue;
      if (clusterOffsets[c] === clusterOffsets[c + 1]) continue;

      // Compute bbox lower bound with early exit
      const bb = c * 14;
      let lb = 0;
      for (let d = 0; d < 14; d++) {
        const qd =
          d === 0
            ? q0
            : d === 1
              ? q1
              : d === 2
                ? q2
                : d === 3
                  ? q3
                  : d === 4
                    ? q4
                    : d === 5
                      ? q5
                      : d === 6
                        ? q6
                        : d === 7
                          ? q7
                          : d === 8
                            ? q8
                            : d === 9
                              ? q9
                              : d === 10
                                ? q10
                                : d === 11
                                  ? q11
                                  : d === 12
                                    ? q12
                                    : q13;
        const mn = bboxMin[bb + d],
          mx = bboxMax[bb + d];
        if (qd < mn) {
          const delta = qd - mn;
          lb += delta * delta;
        } else if (qd > mx) {
          const delta = qd - mx;
          lb += delta * delta;
        }
        if (lb > tau) break;
      }
      if (lb > tau) continue;

      // Scan this cluster
      const start = clusterOffsets[c];
      const end = clusterOffsets[c + 1];
      for (let i = start; i < end; i++) {
        const b = i * 14;
        const d0 = q0 - vectors[b],
          d1 = q1 - vectors[b + 1];
        const d2 = q2 - vectors[b + 2],
          d3 = q3 - vectors[b + 3];
        const d4 = q4 - vectors[b + 4],
          d5 = q5 - vectors[b + 5];
        const d6 = q6 - vectors[b + 6],
          d7 = q7 - vectors[b + 7];

        const partial =
          d0 * d0 +
          d1 * d1 +
          d2 * d2 +
          d3 * d3 +
          d4 * d4 +
          d5 * d5 +
          d6 * d6 +
          d7 * d7;
        if (partial > tau) continue;

        const d8 = q8 - vectors[b + 8],
          d9 = q9 - vectors[b + 9];
        const d10 = q10 - vectors[b + 10],
          d11 = q11 - vectors[b + 11];
        const d12 = q12 - vectors[b + 12],
          d13 = q13 - vectors[b + 13];
        const dist =
          partial +
          d8 * d8 +
          d9 * d9 +
          d10 * d10 +
          d11 * d11 +
          d12 * d12 +
          d13 * d13;

        if (dist < tau) {
          heapDist[0] = dist;
          heapLabel[0] = labels[i];
          this.siftDown(heapDist, heapLabel);
          tau = heapDist[0];
        }
      }
    }

    return (
      heapLabel[0] + heapLabel[1] + heapLabel[2] + heapLabel[3] + heapLabel[4]
    );
  }

  private buildHeap(dist: number[], label: number[]): void {
    let lg = 1;
    if (dist[3] > dist[lg]) lg = 3;
    if (dist[4] > dist[lg]) lg = 4;
    if (lg !== 1) {
      let td = dist[1];
      dist[1] = dist[lg];
      dist[lg] = td;
      let tl = label[1];
      label[1] = label[lg];
      label[lg] = tl;
    }
    lg = 0;
    if (dist[1] > dist[lg]) lg = 1;
    if (dist[2] > dist[lg]) lg = 2;
    if (lg !== 0) {
      let td = dist[0];
      dist[0] = dist[lg];
      dist[lg] = td;
      let tl = label[0];
      label[0] = label[lg];
      label[lg] = tl;
      if (lg === 1) {
        let lg2 = 1;
        if (dist[3] > dist[lg2]) lg2 = 3;
        if (dist[4] > dist[lg2]) lg2 = 4;
        if (lg2 !== 1) {
          td = dist[1];
          dist[1] = dist[lg2];
          dist[lg2] = td;
          tl = label[1];
          label[1] = label[lg2];
          label[lg2] = tl;
        }
      }
    }
  }

  private siftDown(dist: number[], label: number[]): void {
    let idx = 0;
    while (true) {
      let largest = idx;
      const left = 2 * idx + 1;
      const right = 2 * idx + 2;
      if (left < 5 && dist[left] > dist[largest]) largest = left;
      if (right < 5 && dist[right] > dist[largest]) largest = right;
      if (largest === idx) break;
      const td = dist[idx];
      dist[idx] = dist[largest];
      dist[largest] = td;
      const tl = label[idx];
      label[idx] = label[largest];
      label[largest] = tl;
      idx = largest;
    }
  }

  warmup(): void {
    const dummy = new Float32Array(14);
    for (let w = 0; w < 200; w++) {
      for (let d = 0; d < 14; d++) dummy[d] = Math.random();
      this.search(dummy);
    }
    console.log("  Warmup: 200 queries completed");
  }
}
