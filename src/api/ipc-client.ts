import { Socket } from "bun";

const INDEX_HOST = process.env.INDEX_HOST ?? "index-service";
const INDEX_PORT = parseInt(process.env.INDEX_PORT ?? "3000");

const QUERY_SIZE = 14 * 4; // 14 floats × 4 bytes
const RESPONSE_SIZE = 1; // 1 byte: fraud count among 5 neighbors

// Connection pool
const POOL_SIZE = 16;

interface PooledConnection {
  socket: Socket<undefined> | null;
  busy: boolean;
  buffer: Buffer;
  resolve: ((value: number) => void) | null;
  received: number;
}

const pool: PooledConnection[] = [];
const waitQueue: Array<{
  data: Buffer;
  resolve: (value: number) => void;
  reject: (err: Error) => void;
}> = [];

async function createConnection(index: number): Promise<void> {
  const conn = pool[index];
  conn.socket = await Bun.connect({
    hostname: INDEX_HOST,
    port: INDEX_PORT,
    socket: {
      data(socket, data) {
        const conn = pool[index];
        conn.buffer[0] = data[0];
        conn.received = 1;
        if (conn.resolve) {
          conn.resolve(data[0]);
          conn.resolve = null;
          conn.busy = false;
          processQueue();
        }
      },
      close() {
        const conn = pool[index];
        conn.socket = null;
        conn.busy = false;
        // Reconnect
        setTimeout(() => createConnection(index), 100);
      },
      error(socket, error) {
        const conn = pool[index];
        if (conn.resolve) {
          conn.resolve(0); // fallback
          conn.resolve = null;
        }
        conn.busy = false;
        conn.socket = null;
        setTimeout(() => createConnection(index), 100);
      },
      connectError(socket, error) {
        const conn = pool[index];
        conn.socket = null;
        setTimeout(() => createConnection(index), 500);
      },
    },
  });
}

function processQueue() {
  while (waitQueue.length > 0) {
    const freeIdx = pool.findIndex((c) => !c.busy && c.socket !== null);
    if (freeIdx === -1) break;

    const item = waitQueue.shift()!;
    const conn = pool[freeIdx];
    conn.busy = true;
    conn.received = 0;
    conn.resolve = item.resolve;
    conn.socket!.write(item.data);
  }
}

export async function initPool(): Promise<void> {
  for (let i = 0; i < POOL_SIZE; i++) {
    pool.push({
      socket: null,
      busy: false,
      buffer: Buffer.alloc(RESPONSE_SIZE),
      resolve: null,
      received: 0,
    });
  }

  // Connect all with retries
  for (let i = 0; i < POOL_SIZE; i++) {
    await createConnection(i);
  }
}

const queryBuffer = Buffer.alloc(QUERY_SIZE);

export function queryIndex(vector: Float32Array): Promise<number> {
  // Write vector into query buffer
  for (let i = 0; i < 14; i++) {
    queryBuffer.writeFloatLE(vector[i], i * 4);
  }

  const dataCopy = Buffer.from(queryBuffer);

  return new Promise<number>((resolve, reject) => {
    const freeIdx = pool.findIndex((c) => !c.busy && c.socket !== null);
    if (freeIdx !== -1) {
      const conn = pool[freeIdx];
      conn.busy = true;
      conn.received = 0;
      conn.resolve = resolve;
      conn.socket!.write(dataCopy);
    } else {
      waitQueue.push({ data: dataCopy, resolve, reject });
    }
  });
}

export async function checkReady(): Promise<boolean> {
  return pool.some((c) => c.socket !== null);
}
