import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_REVIEW_API_URL || "";

function App() {
  const [repo, setRepo] = useState("");
  const [pr, setPr] = useState("");
  const [endpoint, setEndpoint] = useState("auto");
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("");
  const [job, setJob] = useState(null);
  const [decisionBusy, setDecisionBusy] = useState(false);
  useEffect(() => {
    if (!job?.id || ["awaiting_approval","rejected","posted","partially_posted","post_failed","failed"].includes(job.status)) return;
    const timer = setInterval(async () => { try { const response = await fetch(`${API_BASE}/api/reviews/${job.id}`); if (!response.ok) throw new Error(`Status check failed (${response.status})`); const next = await response.json(); setJob(next); setStatus(next.status); setMessage(next.status === "awaiting_approval" ? "Review complete. Inspect the findings before posting." : `Review status: ${next.status}`); } catch (error) { setStatus("error"); setMessage(error.message); } }, 3000);
    return () => clearInterval(timer);
  }, [job?.id, job?.status]);

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
      const created = await response.json(); setJob(created); setStatus(created.status); setMessage("Review queued successfully. Approval will be required before posting.");
    } catch (error) {
      setStatus("error");
      setMessage(API_BASE ? error.message : "Set VITE_REVIEW_API_URL to connect the review worker.");
    }
  }
  async function decide(action) { if (!job?.id) return; setDecisionBusy(true); try { const response = await fetch(`${API_BASE}/api/reviews/${job.id}/decision`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }) }); const next = await response.json(); if (!response.ok) throw new Error(next.detail || `Decision failed (${response.status})`); setJob(next); setStatus(next.status); setMessage(action === "approve" ? "Posting completed. See the delivery receipt below." : "Review rejected. Nothing was posted to GitHub."); } catch (error) { setStatus("error"); setMessage(error.message); } finally { setDecisionBusy(false); } }

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
          {job && <div className="job-card"><strong>Job {job.id.slice(0, 8)}</strong><span className={`pill ${job.status}`}>{job.status.replaceAll("_", " ")}</span></div>}
          {job?.verdict && <div className="result-card"><div className="result-title"><strong>{job.verdict.recommendation.replaceAll("_", " ").toUpperCase()}</strong><b>{job.verdict.overall_score}/10</b></div><p>{job.verdict.summary}</p><small>{job.findings?.length || 0} findings · {job.cost || "Cost unavailable"}</small></div>}
          {job?.findings?.length > 0 && <div className="findings">{job.findings.map((finding, index) => <article className="finding" key={`${finding.file_path}-${finding.line_start}-${index}`}><div><span className={`severity ${finding.severity}`}>{finding.severity}</span><span className="category">{finding.category}</span></div><strong>{finding.file_path}:{finding.line_start}–{finding.line_end}</strong><p>{finding.comment}</p><small>Confidence {Math.round((finding.confidence || 0) * 100)}%</small></article>)}</div>}
          {job?.status === "awaiting_approval" && <div className="actions"><button type="button" onClick={() => decide("approve")} disabled={decisionBusy}>Approve & post</button><button type="button" className="secondary" onClick={() => decide("reject")} disabled={decisionBusy}>Reject</button></div>}
          {job?.posting_result && <div className="result-card"><strong>Delivery receipt</strong><p>{job.posting_result.status.replaceAll("_", " ")}</p><small>Inline: {job.posting_result.inline_posted || 0} · Fallback: {job.posting_result.fallback_posted || 0} · Failed: {job.posting_result.failed || 0}</small></div>}
        </form>
        <aside className="panel signal-panel"><div className="panel-kicker">What you receive</div><div className="signal"><strong>01</strong><span><b>Evidence-grounded findings</b><small>Code locations, severity, and confidence</small></span></div><div className="signal"><strong>02</strong><span><b>Human checkpoint</b><small>Approve or reject before GitHub comments</small></span></div><div className="signal"><strong>03</strong><span><b>Evaluation trail</b><small>RAGAS and cost telemetry when configured</small></span></div></aside>
      </section>
    </main>
    <footer>OpenCodeReview <span>·</span> React migration preview <span>·</span> API keys stay server-side</footer>
  </div>;
}

createRoot(document.getElementById("root")).render(<App />);
