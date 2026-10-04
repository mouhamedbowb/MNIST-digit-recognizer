"use client";
import { useEffect, useState } from "react";
import DrawCanvas from "../components/DrawCanvas";
import { loadWeights, predict, Weights } from "../lib/cnn";

export default function Page() {
  const [wt, setWt] = useState<Weights | null>(null);
  const [pixels, setPixels] = useState<number[] | null>(null);
  const [probs, setProbs] = useState<number[] | null>(null);
  const [error, setError] = useState("");
  const [clearSignal, setClearSignal] = useState(0);

  useEffect(() => {
    loadWeights().then(setWt).catch(() => setError("model.json not found. Run backend/export_weights.py first."));
  }, []);

  useEffect(() => {
    setProbs(pixels && wt ? predict(wt, pixels) : null);
  }, [pixels, wt]);

  const best = probs ? probs.indexOf(Math.max(...probs)) : -1;

  return (
    <main>
      <h1>Draw a digit</h1>
      <p className="sub">A small CNN trained on MNIST guesses what you wrote. It runs entirely in your browser.</p>
      <div className="grid">
        <div className="panel">
          <DrawCanvas onChange={setPixels} clearSignal={clearSignal} />
          <div className="row"><button onClick={() => setClearSignal((n) => n + 1)}>Clear</button></div>
        </div>
        <div className="panel">
          <p className="big">{best >= 0 ? best : "?"}</p>
          {Array.from({ length: 10 }, (_, i) => (
            <div key={i} className={`bar ${i === best ? "hit" : ""}`}>
              <span>{i}</span>
              <div className="track"><div className="fill" style={{ width: `${(probs?.[i] ?? 0) * 100}%` }} /></div>
              <span>{probs ? (probs[i] * 100).toFixed(1) + "%" : "–"}</span>
            </div>
          ))}
          {error && <p className="err">{error}</p>}
        </div>
      </div>
    </main>
  );
}
