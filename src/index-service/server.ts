import { IVFIndex } from "./ivf-index";

const PORT = parseInt(process.env.INDEX_PORT ?? "3000");
const INDEX_PATH = process.env.INDEX_PATH ?? "/data/index.bin";
const NPROBE = process.env.NPROBE ? parseInt(process.env.NPROBE) : undefined;

console.log(`Loading index from ${INDEX_PATH}...`);
const startLoad = performance.now();

const file = Bun.file(INDEX_PATH);
const buffer = await file.arrayBuffer();
const index = IVFIndex.fromBuffer(buffer, NPROBE);

console.log(`Index loaded in ${(performance.now() - startLoad).toFixed(0)}ms`);

const QUERY_SIZE = 14 * 4; // 56 bytes

// Per-connection state to handle partial reads
const socketBuffers = new WeakMap<
  object,
  { buf: Buffer; len: number; query: Float32Array }
>();

Bun.listen({
  hostname: "0.0.0.0",
  port: PORT,
  socket: {
    open(socket) {
      socketBuffers.set(socket, {
        buf: Buffer.alloc(QUERY_SIZE),
        len: 0,
        query: new Float32Array(14),
      });
    },
    data(socket, data) {
      const state = socketBuffers.get(socket)!;
      let dataOffset = 0;

      while (dataOffset < data.length) {
        // How much we need to complete a query
        const needed = QUERY_SIZE - state.len;
        const available = data.length - dataOffset;
        const toCopy = Math.min(needed, available);

        // Copy into accumulation buffer
        data.copy(state.buf, state.len, dataOffset, dataOffset + toCopy);
        state.len += toCopy;
        dataOffset += toCopy;

        // If we have a complete query, process it
        if (state.len === QUERY_SIZE) {
          for (let i = 0; i < 14; i++) {
            state.query[i] = state.buf.readFloatLE(i * 4);
          }

          const fraudCount = index.search(state.query);
          socket.write(new Uint8Array([fraudCount]));

          state.len = 0; // Reset for next query
        }
      }
    },
    close(socket) {
      socketBuffers.delete(socket);
    },
    error(socket, error) {
      console.error("Socket error:", error);
    },
  },
});

console.log(`Index service listening on port ${PORT}`);
