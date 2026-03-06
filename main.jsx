import { useState, useEffect } from "react";

const CONFIG = {
  seq_len: 6,
  d_model: 512,
  num_heads: 8,
  d_k: 64,
  d_ff: 2048,
};

const LAYERS = [
  {
    id: "input",
    label: "Input Embeddings",
    sublabel: "Token + Positional",
    color: "#38bdf8",
    glow: "#38bdf880",
    inputShape: null,
    outputShape: ["seq", "d_model"],
    outputDim: [CONFIG.seq_len, CONFIG.d_model],
    detail: {
      title: "Input Representation",
      desc: "Each token is embedded into a d_model-dimensional vector. Positional encodings are added to inject sequence order information.",
      ops: [
        { name: "Token Embedding", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "vocab → d_model" },
        { name: "+ Positional Encoding", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "sin/cos patterns" },
      ],
      formula: "x = Embed(token) + PE(position)",
    },
  },
  {
    id: "qkv",
    label: "Q, K, V Projections",
    sublabel: "Linear Transforms × 3",
    color: "#a78bfa",
    glow: "#a78bfa80",
    inputShape: ["seq", "d_model"],
    outputShape: ["heads", "seq", "d_k"],
    outputDim: [CONFIG.num_heads, CONFIG.seq_len, CONFIG.d_k],
    detail: {
      title: "Query, Key, Value Projections",
      desc: "Three separate linear projections transform input into Q, K, V matrices. Then reshaped and split into multiple heads.",
      ops: [
        { name: "W_Q projection", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "W_Q: d_model×d_model" },
        { name: "W_K projection", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "W_K: d_model×d_model" },
        { name: "W_V projection", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "W_V: d_model×d_model" },
        { name: "Reshape + Split heads", shape: `(${CONFIG.num_heads}, ${CONFIG.seq_len}, ${CONFIG.d_k})`, note: `${CONFIG.num_heads} heads × d_k=${CONFIG.d_k}` },
      ],
      formula: "Q=xW_Q, K=xW_K, V=xW_V  →  reshape to (h, seq, d_k)",
    },
  },
  {
    id: "attention",
    label: "Scaled Dot-Product Attention",
    sublabel: "Per Head",
    color: "#f472b6",
    glow: "#f472b680",
    inputShape: ["heads", "seq", "d_k"],
    outputShape: ["heads", "seq", "d_k"],
    outputDim: [CONFIG.num_heads, CONFIG.seq_len, CONFIG.d_k],
    detail: {
      title: "Scaled Dot-Product Attention",
      desc: "Each head independently computes attention scores between all token pairs, then attends over values.",
      ops: [
        { name: "QKᵀ (dot product)", shape: `(${CONFIG.num_heads}, ${CONFIG.seq_len}, ${CONFIG.seq_len})`, note: "attention logits" },
        { name: "÷ √d_k scaling", shape: `(${CONFIG.num_heads}, ${CONFIG.seq_len}, ${CONFIG.seq_len})`, note: `÷ √${CONFIG.d_k} = ${Math.sqrt(CONFIG.d_k).toFixed(2)}` },
        { name: "Softmax", shape: `(${CONFIG.num_heads}, ${CONFIG.seq_len}, ${CONFIG.seq_len})`, note: "attention weights" },
        { name: "× V (weighted sum)", shape: `(${CONFIG.num_heads}, ${CONFIG.seq_len}, ${CONFIG.d_k})`, note: "context vectors" },
      ],
      formula: "Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V",
    },
  },
  {
    id: "concat",
    label: "Concat + W_O Projection",
    sublabel: "Multi-Head Merge",
    color: "#fb923c",
    glow: "#fb923c80",
    inputShape: ["heads", "seq", "d_k"],
    outputShape: ["seq", "d_model"],
    outputDim: [CONFIG.seq_len, CONFIG.d_model],
    detail: {
      title: "Concatenate Heads & Output Projection",
      desc: "All 8 head outputs are concatenated along the last dim (8×64=512), then projected back to d_model.",
      ops: [
        { name: "Concat all heads", shape: `(${CONFIG.seq_len}, ${CONFIG.num_heads * CONFIG.d_k})`, note: `${CONFIG.num_heads}×${CONFIG.d_k}=${CONFIG.d_model}` },
        { name: "W_O projection", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "W_O: d_model×d_model" },
      ],
      formula: "MultiHead = Concat(head₁,...,headₕ) · W_O",
    },
  },
  {
    id: "addnorm1",
    label: "Add & Layer Norm",
    sublabel: "Residual Connection #1",
    color: "#34d399",
    glow: "#34d39980",
    inputShape: ["seq", "d_model"],
    outputShape: ["seq", "d_model"],
    outputDim: [CONFIG.seq_len, CONFIG.d_model],
    detail: {
      title: "Residual Connection + Layer Normalization",
      desc: "The original input (before attention) is added back (skip connection), then normalized across the d_model dimension.",
      ops: [
        { name: "x + MHA(x)", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "residual add" },
        { name: "LayerNorm", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "normalize per token" },
      ],
      formula: "LayerNorm(x + MultiHead(x, x, x))",
    },
  },
  {
    id: "ffn",
    label: "Feed-Forward Network",
    sublabel: "Position-wise FFN",
    color: "#facc15",
    glow: "#facc1580",
    inputShape: ["seq", "d_model"],
    outputShape: ["seq", "d_model"],
    outputDim: [CONFIG.seq_len, CONFIG.d_model],
    detail: {
      title: "Position-wise Feed-Forward Network",
      desc: "Applied independently to each position. Expands to d_ff=2048 (4× d_model), applies ReLU, then projects back.",
      ops: [
        { name: "Linear W₁ + bias", shape: `(${CONFIG.seq_len}, ${CONFIG.d_ff})`, note: `expand: ${CONFIG.d_model}→${CONFIG.d_ff}` },
        { name: "ReLU activation", shape: `(${CONFIG.seq_len}, ${CONFIG.d_ff})`, note: "non-linearity" },
        { name: "Linear W₂ + bias", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: `contract: ${CONFIG.d_ff}→${CONFIG.d_model}` },
      ],
      formula: "FFN(x) = max(0, xW₁+b₁)W₂+b₂",
    },
  },
  {
    id: "addnorm2",
    label: "Add & Layer Norm",
    sublabel: "Residual Connection #2",
    color: "#34d399",
    glow: "#34d39980",
    inputShape: ["seq", "d_model"],
    outputShape: ["seq", "d_model"],
    outputDim: [CONFIG.seq_len, CONFIG.d_model],
    detail: {
      title: "Second Residual + Layer Norm",
      desc: "The FFN input is added back via another skip connection, then normalized. Output is the final encoder block representation.",
      ops: [
        { name: "x + FFN(x)", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "residual add" },
        { name: "LayerNorm", shape: `(${CONFIG.seq_len}, ${CONFIG.d_model})`, note: "normalize per token" },
      ],
      formula: "LayerNorm(x + FFN(x))",
    },
  },
];

const DIM_COLORS = {
  seq: "#38bdf8",
  d_model: "#a78bfa",
  heads: "#fb923c",
  d_k: "#f472b6",
  d_ff: "#facc15",
};

function ShapeTag({ dims, dimLabels }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 2, flexWrap: "wrap", justifyContent: "center" }}>
      <span style={{ color: "#64748b", fontSize: 12 }}>(</span>
      {dimLabels.map((lbl, i) => (
        <span key={i} style={{ display: "flex", alignItems: "center", gap: 2 }}>
          <span style={{
            background: DIM_COLORS[lbl] + "22",
            border: `1px solid ${DIM_COLORS[lbl]}55`,
            color: DIM_COLORS[lbl],
            borderRadius: 4,
            padding: "1px 6px",
            fontSize: 11,
            fontFamily: "monospace",
            fontWeight: 700,
          }}>
            {dims[i]}<span style={{ color: DIM_COLORS[lbl] + "88", fontSize: 9, marginLeft: 2 }}>{lbl}</span>
          </span>
          {i < dimLabels.length - 1 && <span style={{ color: "#64748b", fontSize: 12 }}>×</span>}
        </span>
      ))}
      <span style={{ color: "#64748b", fontSize: 12 }}>)</span>
    </div>
  );
}

function AttentionMap() {
  const n = CONFIG.seq_len;
  const tokens = ["The", "cat", "sat", "on", "the", "mat"];
  const weights = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const base = Math.random() * 0.3;
      const diag = i === j ? 0.5 : 0;
      const nearby = Math.abs(i - j) === 1 ? 0.2 : 0;
      return base + diag + nearby;
    })
  );
  // normalize rows
  weights.forEach(row => {
    const sum = row.reduce((a, b) => a + b, 0);
    row.forEach((_, i) => row[i] /= sum);
  });

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 10, color: "#64748b", marginBottom: 6, textAlign: "center" }}>
        Attention Weight Matrix (Head 1) — (seq × seq) = ({n} × {n})
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `40px repeat(${n}, 1fr)`, gap: 2, fontSize: 9 }}>
        <div />
        {tokens.map((t, j) => (
          <div key={j} style={{ color: "#94a3b8", textAlign: "center", fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis" }}>{t}</div>
        ))}
        {tokens.map((t, i) => (
          <>
            <div key={`row-${i}`} style={{ color: "#94a3b8", fontFamily: "monospace", display: "flex", alignItems: "center" }}>{t}</div>
            {weights[i].map((w, j) => (
              <div key={j} style={{
                background: `rgba(244, 114, 182, ${w})`,
                border: "1px solid #f472b622",
                borderRadius: 3,
                height: 20,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: w > 0.4 ? "#fff" : "#f472b6",
                fontWeight: 600,
                fontSize: 8,
              }}>
                {w.toFixed(2)}
              </div>
            ))}
          </>
        ))}
      </div>
    </div>
  );
}

function TensorViz({ shape, color }) {
  if (shape.length === 2) {
    const [rows, cols] = shape;
    const displayRows = Math.min(rows, 4);
    const displayCols = Math.min(cols, 6);
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "center" }}>
        {Array.from({ length: displayRows }).map((_, i) => (
          <div key={i} style={{ display: "flex", gap: 2 }}>
            {Array.from({ length: displayCols }).map((_, j) => (
              <div key={j} style={{
                width: 10, height: 10,
                background: color + Math.floor(Math.random() * 40 + 20).toString(16),
                border: `1px solid ${color}44`,
                borderRadius: 2,
              }} />
            ))}
            {cols > displayCols && <span style={{ color: "#475569", fontSize: 9, alignSelf: "center" }}>…</span>}
          </div>
        ))}
        {rows > displayRows && <span style={{ color: "#475569", fontSize: 9 }}>…</span>}
        <div style={{ fontSize: 9, color: color + "cc", marginTop: 2 }}>{rows}×{cols}</div>
      </div>
    );
  }
  if (shape.length === 3) {
    const [depth, rows, cols] = shape;
    return (
      <div style={{ position: "relative", width: 70, height: 60 }}>
        {[2, 1, 0].map(d => (
          <div key={d} style={{
            position: "absolute",
            top: d * 4,
            left: d * 4,
            display: "grid",
            gridTemplateColumns: `repeat(${Math.min(cols, 4)}, 8px)`,
            gap: 1,
            background: "#0f172a",
            padding: 2,
            border: `1px solid ${color}44`,
            borderRadius: 3,
          }}>
            {Array.from({ length: Math.min(rows, 3) * Math.min(cols, 4) }).map((_, i) => (
              <div key={i} style={{
                width: 8, height: 8,
                background: color + Math.floor(Math.random() * 40 + 20).toString(16),
                borderRadius: 1,
              }} />
            ))}
          </div>
        ))}
        <div style={{ position: "absolute", bottom: -14, left: 0, right: 0, textAlign: "center", fontSize: 9, color: color + "cc" }}>
          {depth}×{rows}×{cols}
        </div>
      </div>
    );
  }
  return null;
}

export default function TransformerViz() {
  const [selected, setSelected] = useState(null);
  const [animStep, setAnimStep] = useState(0);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let interval;
    if (running) {
      interval = setInterval(() => {
        setAnimStep(s => {
          if (s >= LAYERS.length - 1) { setRunning(false); return s; }
          return s + 1;
        });
      }, 700);
    }
    return () => clearInterval(interval);
  }, [running]);

  const selectedLayer = selected !== null ? LAYERS[selected] : null;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#020817",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      color: "#e2e8f0",
      padding: "24px 16px",
      display: "flex",
      gap: 24,
      flexDirection: "column",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 11, letterSpacing: 6, color: "#38bdf8", textTransform: "uppercase", marginBottom: 8 }}>
          Architecture Deep Dive
        </div>
        <h1 style={{
          fontSize: 28,
          fontWeight: 800,
          background: "linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          margin: 0,
        }}>
          Transformer Encoder Block
        </h1>
        <div style={{ color: "#475569", fontSize: 12, marginTop: 6 }}>
          seq_len={CONFIG.seq_len} · d_model={CONFIG.d_model} · heads={CONFIG.num_heads} · d_k={CONFIG.d_k} · d_ff={CONFIG.d_ff}
        </div>
        <button
          onClick={() => { setAnimStep(0); setSelected(null); setRunning(true); }}
          style={{
            marginTop: 12,
            padding: "8px 24px",
            background: "linear-gradient(90deg, #38bdf822, #a78bfa22)",
            border: "1px solid #38bdf844",
            borderRadius: 8,
            color: "#38bdf8",
            cursor: "pointer",
            fontSize: 11,
            letterSpacing: 2,
            textTransform: "uppercase",
          }}>
          ▶ Animate Forward Pass
        </button>
      </div>

      <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
        {/* Left: Architecture flow */}
        <div style={{ flex: "0 0 320px", display: "flex", flexDirection: "column", gap: 0 }}>
          {/* Residual bracket labels */}
          <div style={{ position: "relative" }}>
            {/* MHA Residual indicator */}
            <div style={{
              position: "absolute", left: -20, top: 80, width: 14, height: 240,
              borderLeft: "2px dashed #34d39944",
              borderTop: "2px dashed #34d39944",
              borderBottom: "2px dashed #34d39944",
              borderRadius: "6px 0 0 6px",
              zIndex: 1,
            }} />
            <div style={{
              position: "absolute", left: -60, top: 185, fontSize: 9,
              color: "#34d399", letterSpacing: 1, textTransform: "uppercase",
              transform: "rotate(-90deg)", whiteSpace: "nowrap",
            }}>Residual</div>

            {/* FFN Residual indicator */}
            <div style={{
              position: "absolute", left: -20, top: 340, width: 14, height: 210,
              borderLeft: "2px dashed #34d39944",
              borderTop: "2px dashed #34d39944",
              borderBottom: "2px dashed #34d39944",
              borderRadius: "6px 0 0 6px",
              zIndex: 1,
            }} />
            <div style={{
              position: "absolute", left: -60, top: 435, fontSize: 9,
              color: "#34d399", letterSpacing: 1, textTransform: "uppercase",
              transform: "rotate(-90deg)", whiteSpace: "nowrap",
            }}>Residual</div>

            {LAYERS.map((layer, i) => (
              <div key={layer.id}>
                {/* Connector arrow */}
                {i > 0 && (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "2px 0" }}>
                    <div style={{
                      width: 1,
                      height: 16,
                      background: `linear-gradient(180deg, ${LAYERS[i - 1].color}88, ${layer.color}88)`,
                    }} />
                    <div style={{ color: layer.color + "88", fontSize: 10 }}>▼</div>
                    {/* Shape between layers */}
                    {layer.inputShape && (
                      <ShapeTag dims={LAYERS[i-1].outputDim} dimLabels={LAYERS[i-1].outputShape} />
                    )}
                    <div style={{ height: 4 }} />
                  </div>
                )}

                {/* Layer block */}
                <div
                  onClick={() => setSelected(selected === i ? null : i)}
                  style={{
                    border: `1px solid ${selected === i ? layer.color : layer.color + "44"}`,
                    borderRadius: 10,
                    padding: "12px 16px",
                    cursor: "pointer",
                    background: selected === i
                      ? `radial-gradient(ellipse at 50% 0%, ${layer.glow}, transparent 70%), #0f172a`
                      : animStep >= i && running ? `${layer.color}11` : "#0f172a",
                    boxShadow: selected === i ? `0 0 20px ${layer.glow}` : animStep === i && running ? `0 0 30px ${layer.glow}` : "none",
                    transition: "all 0.3s ease",
                    position: "relative",
                    overflow: "hidden",
                  }}>
                  {animStep === i && running && (
                    <div style={{
                      position: "absolute", inset: 0,
                      background: `linear-gradient(90deg, transparent, ${layer.color}22, transparent)`,
                      animation: "pulse 1s ease-in-out",
                    }} />
                  )}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div>
                      <div style={{
                        fontSize: 13, fontWeight: 700, color: layer.color,
                        textShadow: selected === i ? `0 0 10px ${layer.color}` : "none",
                      }}>
                        {layer.label}
                      </div>
                      <div style={{ fontSize: 10, color: "#475569", marginTop: 2 }}>{layer.sublabel}</div>
                    </div>
                    <div style={{
                      fontSize: 10, color: layer.color + "88",
                      background: layer.color + "11",
                      border: `1px solid ${layer.color}22`,
                      borderRadius: 4, padding: "2px 6px",
                    }}>
                      {selected === i ? "▲" : "▼"} detail
                    </div>
                  </div>

                  {/* Output shape pill */}
                  <div style={{ marginTop: 8 }}>
                    <ShapeTag dims={layer.outputDim} dimLabels={layer.outputShape} />
                  </div>
                </div>
              </div>
            ))}

            {/* Final output label */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "4px 0" }}>
              <div style={{ width: 1, height: 16, background: "#34d39988" }} />
              <div style={{ color: "#34d399", fontSize: 10 }}>▼</div>
              <div style={{
                border: "1px dashed #34d39966",
                borderRadius: 8,
                padding: "8px 16px",
                color: "#34d399",
                fontSize: 11,
                textAlign: "center",
                background: "#34d39911",
              }}>
                Encoder Output<br />
                <span style={{ fontSize: 10, opacity: 0.7 }}>(passed to next block or decoder)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Detail panel */}
        <div style={{ flex: 1, minWidth: 300 }}>
          {!selectedLayer ? (
            <div style={{
              border: "1px solid #1e293b",
              borderRadius: 12,
              padding: 24,
              background: "#0f172a",
              height: "100%",
            }}>
              <div style={{ fontSize: 12, color: "#475569", marginBottom: 20 }}>← Click any layer to inspect</div>

              {/* Legend */}
              <div style={{ marginBottom: 24 }}>
                <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>
                  Dimension Legend
                </div>
                {Object.entries(DIM_COLORS).map(([k, v]) => (
                  <div key={k} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: v }} />
                    <span style={{ color: v, fontFamily: "monospace", fontSize: 12, width: 80 }}>{k}</span>
                    <span style={{ color: "#64748b", fontSize: 11 }}>
                      {{
                        seq: `sequence length = ${CONFIG.seq_len} tokens`,
                        d_model: `model dimension = ${CONFIG.d_model}`,
                        heads: `attention heads = ${CONFIG.num_heads}`,
                        d_k: `key/query dim = ${CONFIG.d_k} (d_model/heads)`,
                        d_ff: `feed-forward dim = ${CONFIG.d_ff} (4×d_model)`,
                      }[k]}
                    </span>
                  </div>
                ))}
              </div>

              {/* Parameter count */}
              <div style={{ border: "1px solid #1e293b", borderRadius: 8, padding: 16, background: "#020817" }}>
                <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>
                  Parameter Count (1 Block)
                </div>
                {[
                  ["W_Q, W_K, W_V", `3 × (${CONFIG.d_model}×${CONFIG.d_model})`, 3 * CONFIG.d_model * CONFIG.d_model],
                  ["W_O", `${CONFIG.d_model}×${CONFIG.d_model}`, CONFIG.d_model * CONFIG.d_model],
                  ["FFN W₁", `${CONFIG.d_model}×${CONFIG.d_ff}`, CONFIG.d_model * CONFIG.d_ff],
                  ["FFN W₂", `${CONFIG.d_ff}×${CONFIG.d_model}`, CONFIG.d_ff * CONFIG.d_model],
                  ["LayerNorm (×2)", `2×2×${CONFIG.d_model}`, 4 * CONFIG.d_model],
                ].map(([name, shape, count]) => (
                  <div key={name} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 11 }}>
                    <span style={{ color: "#94a3b8" }}>{name}</span>
                    <span style={{ color: "#475569", fontFamily: "monospace", fontSize: 10 }}>{shape}</span>
                    <span style={{ color: "#38bdf8" }}>{(count / 1000).toFixed(0)}K</span>
                  </div>
                ))}
                <div style={{ borderTop: "1px solid #1e293b", paddingTop: 8, marginTop: 8, display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                  <span style={{ color: "#e2e8f0", fontWeight: 700 }}>Total</span>
                  <span style={{ color: "#a78bfa", fontWeight: 700 }}>
                    {((3 * CONFIG.d_model ** 2 + CONFIG.d_model ** 2 + 2 * CONFIG.d_model * CONFIG.d_ff + 4 * CONFIG.d_model) / 1e6).toFixed(2)}M params
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div style={{
              border: `1px solid ${selectedLayer.color}44`,
              borderRadius: 12,
              padding: 24,
              background: "#0f172a",
              boxShadow: `0 0 30px ${selectedLayer.glow}`,
            }}>
              <div style={{ fontSize: 11, color: selectedLayer.color + "88", letterSpacing: 3, textTransform: "uppercase", marginBottom: 4 }}>
                Layer Detail
              </div>
              <h2 style={{ margin: "0 0 8px", fontSize: 18, color: selectedLayer.color }}>
                {selectedLayer.detail.title}
              </h2>
              <p style={{ color: "#94a3b8", fontSize: 12, lineHeight: 1.6, margin: "0 0 20px" }}>
                {selectedLayer.detail.desc}
              </p>

              {/* Formula */}
              <div style={{
                background: "#020817",
                border: `1px solid ${selectedLayer.color}33`,
                borderRadius: 8,
                padding: "10px 16px",
                marginBottom: 20,
                fontFamily: "monospace",
                fontSize: 13,
                color: selectedLayer.color,
                textAlign: "center",
              }}>
                {selectedLayer.detail.formula}
              </div>

              {/* Tensor Viz */}
              <div style={{ display: "flex", gap: 20, alignItems: "center", marginBottom: 20, justifyContent: "center" }}>
                {selectedLayer.inputShape && (
                  <>
                    <div style={{ textAlign: "center" }}>
                      <div style={{ fontSize: 10, color: "#475569", marginBottom: 8 }}>Input</div>
                      <TensorViz
                        shape={LAYERS[LAYERS.findIndex(l => l.id === selectedLayer.id) - 1]?.outputDim || selectedLayer.outputDim}
                        color="#64748b"
                      />
                    </div>
                    <div style={{ color: selectedLayer.color, fontSize: 20 }}>→</div>
                  </>
                )}
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 10, color: "#475569", marginBottom: 8 }}>Output</div>
                  <TensorViz shape={selectedLayer.outputDim} color={selectedLayer.color} />
                </div>
              </div>

              {/* Step-by-step ops */}
              <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 2, textTransform: "uppercase", marginBottom: 10 }}>
                Operations
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {selectedLayer.detail.ops.map((op, i) => (
                  <div key={i} style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    background: "#020817",
                    border: `1px solid ${selectedLayer.color}22`,
                    borderRadius: 8,
                    padding: "8px 12px",
                  }}>
                    <div style={{
                      width: 20, height: 20,
                      borderRadius: "50%",
                      background: selectedLayer.color + "22",
                      border: `1px solid ${selectedLayer.color}66`,
                      color: selectedLayer.color,
                      fontSize: 10,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      flexShrink: 0,
                      fontWeight: 700,
                    }}>{i + 1}</div>
                    <div style={{ flex: 1 }}>
                      <div style={{ color: "#e2e8f0", fontSize: 12 }}>{op.name}</div>
                      <div style={{ color: "#475569", fontSize: 10 }}>{op.note}</div>
                    </div>
                    <div style={{
                      fontFamily: "monospace",
                      fontSize: 11,
                      color: selectedLayer.color,
                      background: selectedLayer.color + "11",
                      border: `1px solid ${selectedLayer.color}33`,
                      borderRadius: 4,
                      padding: "2px 8px",
                      whiteSpace: "nowrap",
                    }}>
                      {op.shape}
                    </div>
                  </div>
                ))}
              </div>

              {/* Attention heatmap for attention layer */}
              {selectedLayer.id === "attention" && <AttentionMap />}
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #020817; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
      `}</style>
    </div>
  );
}
