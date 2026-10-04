// Pure TypeScript forward pass of the notebook's CNN (no ML library needed).
export type Layer = { w: number[]; b: number[] };
export type Weights = { conv1: Layer; conv2: Layer; fc1: Layer; out: Layer };

let cache: Promise<Weights> | null = null;
export const loadWeights = (): Promise<Weights> =>
  (cache ??= fetch("/model.json").then((r) => {
    if (!r.ok) throw new Error("model.json missing");
    return r.json();
  }));

// 5x5 convolution, stride 1, padding 1 (PyTorch layout: [out][in][kh][kw])
function conv(x: Float32Array, cin: number, h: number, w: number, L: Layer, cout: number) {
  const k = 5, pad = 1, oh = h + 2 * pad - k + 1, ow = w + 2 * pad - k + 1;
  const y = new Float32Array(cout * oh * ow);
  for (let o = 0; o < cout; o++)
    for (let i = 0; i < oh; i++)
      for (let j = 0; j < ow; j++) {
        let s = L.b[o];
        for (let c = 0; c < cin; c++)
          for (let a = 0; a < k; a++) {
            const yy = i + a - pad;
            if (yy < 0 || yy >= h) continue;
            for (let b = 0; b < k; b++) {
              const xx = j + b - pad;
              if (xx < 0 || xx >= w) continue;
              s += x[(c * h + yy) * w + xx] * L.w[((o * cin + c) * k + a) * k + b];
            }
          }
        y[(o * oh + i) * ow + j] = s;
      }
  return { y, h: oh, w: ow };
}

// 2x2 max pool followed by ReLU
function poolRelu(x: Float32Array, c: number, h: number, w: number) {
  const oh = Math.floor(h / 2), ow = Math.floor(w / 2), y = new Float32Array(c * oh * ow);
  for (let k = 0; k < c; k++)
    for (let i = 0; i < oh; i++)
      for (let j = 0; j < ow; j++) {
        const p = (r: number, q: number) => x[(k * h + 2 * i + r) * w + 2 * j + q];
        y[(k * oh + i) * ow + j] = Math.max(0, p(0, 0), p(0, 1), p(1, 0), p(1, 1));
      }
  return { y, h: oh, w: ow };
}

function dense(x: Float32Array, L: Layer, nOut: number, relu: boolean) {
  const nIn = x.length, y = new Float32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let s = L.b[o];
    for (let i = 0; i < nIn; i++) s += x[i] * L.w[o * nIn + i];
    y[o] = relu ? Math.max(0, s) : s;
  }
  return y;
}

/** pixels: 784 numbers in [0,1], row-major. Returns 10 probabilities. */
export function predict(m: Weights, pixels: number[]): number[] {
  const a = conv(Float32Array.from(pixels), 1, 28, 28, m.conv1, 10);
  const b = poolRelu(a.y, 10, a.h, a.w);
  const c = conv(b.y, 10, b.h, b.w, m.conv2, 20);
  const d = poolRelu(c.y, 20, c.h, c.w);
  const logits = dense(dense(d.y, m.fc1, 50, true), m.out, 10, false);
  const mx = Math.max(...logits), e = Array.from(logits, (v) => Math.exp(v - mx));
  const sum = e.reduce((p, q) => p + q, 0);
  return e.map((v) => v / sum);
}
