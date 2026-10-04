"use client";
import { useEffect, useRef } from "react";

type Props = { onChange: (pixels: number[] | null) => void; clearSignal: number };

export default function DrawCanvas({ onChange, clearSignal }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const last = useRef<[number, number] | null>(null);

  const clear = () => {
    const c = ref.current!, ctx = c.getContext("2d")!;
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, c.width, c.height);
  };
  useEffect(() => { clear(); onChange(null); }, [clearSignal]); // eslint-disable-line

  const pos = (e: React.PointerEvent): [number, number] => {
    const c = ref.current!, r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height];
  };
  const stroke = (e: React.PointerEvent) => {
    const ctx = ref.current!.getContext("2d")!, p = pos(e), from = last.current ?? p;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 18; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
    last.current = p;
  };

  // MNIST-style preprocessing: crop -> fit in 20x20 -> center by mass in 28x28
  const export28 = () => {
    const c = ref.current!, d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height).data;
    let x0 = c.width, y0 = c.height, x1 = -1, y1 = -1;
    for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++)
      if (d[(y * c.width + x) * 4] > 20) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    if (x1 < 0) { onChange(null); return; }
    const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h);
    const small = document.createElement("canvas"); small.width = small.height = 28;
    const sctx = small.getContext("2d")!; sctx.imageSmoothingQuality = "high";
    const draw = (dx: number, dy: number) => {
      sctx.fillStyle = "#000"; sctx.fillRect(0, 0, 28, 28);
      sctx.drawImage(c, x0, y0, w, h, (28 - w * s) / 2 + dx, (28 - h * s) / 2 + dy, w * s, h * s);
      return sctx.getImageData(0, 0, 28, 28).data;
    };
    let px = draw(0, 0), m = 0, cx = 0, cy = 0;
    for (let i = 0; i < 784; i++) { const v = px[i * 4]; m += v; cx += v * (i % 28); cy += v * Math.floor(i / 28); }
    px = draw(14 - cx / m, 14 - cy / m);
    const out: number[] = [];
    for (let i = 0; i < 784; i++) out.push(px[i * 4] / 255);
    onChange(out);
  };

  return (
    <canvas ref={ref} width={280} height={280} aria-label="Draw a digit here"
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); last.current = null; stroke(e); }}
      onPointerMove={(e) => last.current && stroke(e)}
      onPointerUp={() => { last.current = null; export28(); }} />
  );
}
