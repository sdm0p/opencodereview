import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_REVIEW_API_URL || "";

function App() {
  const [repo, setRepo] = useState("");
  const [pr, setPr] = useState("");
  const [endpoint, setEndpoint] = useState("auto");
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("");

  async function startReview(event) {
    event.preventDefault();
    if (!/^[-\w.]+\/[-\w.]+$/.test(repo.trim()) || !Number(pr)) {
      setStatus("error");
      setMessage("Enter a repository as owner/name and a valid PR number.");
      return;
    }
    setStatus("loading");
    setMessage("Review queued. The API worker will report progress here.");
    try {
      const response = await fetch(`${API_BASE}/api/reviews`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo: repo.trim(), pr_number: Number(pr), endpoint }),
      });
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      setStatus("queued");
      setMessage("Review queued successfully. Approval will be required before posting.");
    } catch (error) {
      setStatus("error");
      setMessage(API_BASE ? error.message : "Set VITE_REVIEW_API_URL to connect the review worker.");
    }
  }

  return <div className="shell">
    <header className="topbar"><div className="brand"><span className="brand-mark">OC</span><span>OpenCodeReview</span></div><span className="status-dot">Human approval required</span></header>
    <main className="content">
      <section className="hero"><p className="eyebrow">Pull request intelligence</p><h1>Review changes with<br /><em>evidence before action.</em></h1><p className="lede">Parallel analysis for correctness, security, and test coverage. Nothing is posted to GitHub until you approve it.</p></section>
      <section className="workspace">
        <form className="panel form-panel" onSubmit={startReview}>
          <div className="panel-kicker">New review</div><h2>Inspect a pull request</h2>
          <label>Repository<input value={repo} onChange={e => setRepo(e.target.value)} placeholder="owner/repository" autoComplete="off" /></label>
          <label>Pull request<input value={pr} onChange={e => setPr(e.target.value)} placeholder="42" inputMode="numeric" /></label>
          <label>LLM endpoint<select value={endpoint} onChange={e => setEndpoint(e.target.value)}><option value="auto">Automatic fallback</option><option value="gemini">Gemini</option><option value="groq">Groq</option></select></label>
          <button type="submit" disabled={status === "loading"}>{status === "loading" ? "Queueing review…" : "Start review"}<span>↗</span></button>
          {message && <p className={`message ${status}`}>{message}</p>}
        </form>
        <aside className="panel signal-panel"><div className="panel-kicker">What you receive</div><div className="signal"><strong>01</strong><span><b>Evidence-grounded findings</b><small>Code locations, severity, and confidence</small></span></div><div className="signal"><strong>02</strong><span><b>Human checkpoint</b><small>Approve or reject before GitHub comments</small></span></div><div className="signal"><strong>03</strong><span><b>Evaluation trail</b><small>RAGAS and cost telemetry when configured</small></span></div></aside>
      </section>
    </main>
    <footer>OpenCodeReview <span>·</span> React migration preview <span>·</span> API keys stay server-side</footer>
  </div>;
}

createRoot(document.getElementById("root")).render(<App />);
