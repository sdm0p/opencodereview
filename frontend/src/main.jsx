import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = (import.meta.env.VITE_REVIEW_API_URL || "").replace(/\/$/, "");
const TERMINAL = new Set(["awaiting_approval", "rejected", "posted", "partially_posted", "post_failed", "failed"]);
const apiUrl = (path) => `${API_BASE}${path}`;
const pretty = (value) => String(value || "").replaceAll("_", " ");

function Verdict({ job }) {
  if (!job?.verdict) return null;
  const v = job.verdict;
  const blocking = ["block", "request_changes"].includes(v.recommendation);
  const score = typeof v.overall_score === "number" ? v.overall_score : null;
  const scoreLabel = score == null ? "Not scored" : score < 4 ? "High risk" : score < 7 ? "Review carefully" : "Low risk";
  const scoreHelp = score == null ? "The review did not return a numeric score." : score < 4 ? "Resolve the high-severity findings before merging." : score < 7 ? "Read the findings and confirm the risk before merging." : "No major risks were detected by the review.";
  return <section className={`result-card verdict ${blocking ? "blocking" : "approved"}`} aria-labelledby="verdict-title">
    <div className="result-title"><div><span className="panel-kicker">Review decision</span><h3 id="verdict-title">{pretty(v.recommendation).toUpperCase()}</h3></div><div className="score-block"><strong className="score">{score == null ? "â€”" : `${score.toFixed(1)}/10`}</strong><span>{scoreLabel}</span></div></div>
    <p className="verdict-summary">{v.summary || "The review is ready for your decision."}</p><div className="score-explanation"><strong>What this means</strong><span>{scoreHelp}</span></div><div className="metrics"><span>{job.findings?.length || 0} findings</span>{job.cost && <span>Cost {typeof job.cost === "string" ? job.cost : JSON.stringify(job.cost)}</span>}</div>
  </section>;
}

function Findings({ findings = [] }) {
  return <section aria-labelledby="findings-title"><div className="section-heading"><h3 id="findings-title">Findings</h3><span>{findings.length}</span></div>{findings.length ? <div className="findings">{findings.map((f, i) => <article className="finding" key={`${f.file_path || "finding"}-${f.line_start || 0}-${i}`}><div className="finding-meta"><span className={`severity ${String(f.severity || "info").toLowerCase()}`}>{f.severity || "info"}</span>{f.category && <span className="category">{f.category}</span>}<span className="finding-location">{f.file_path || "Unknown file"}{f.line_start ? `:${f.line_start}${f.line_end && f.line_end !== f.line_start ? `â€“${f.line_end}` : ""}` : ""}</span></div><p>{f.comment || f.description || "No explanation provided."}</p>{f.confidence != null && <small>Confidence {Math.round(Number(f.confidence) * 100)}%</small>}</article>)}</div> : <div className="empty-result"><strong>No findings</strong><span>The reviewers did not identify actionable issues.</span></div>}</section>;
}

function Receipt({ result }) {
  if (!result) return null;
  return <section className="result-card receipt" aria-labelledby="receipt-title"><span className="panel-kicker">Delivery receipt</span><h3 id="receipt-title">{pretty(result.status || "completed")}</h3><div className="metrics"><span>Inline: {result.inline_posted || 0}</span><span>Fallback: {result.fallback_posted || 0}</span><span>Finding failures: {result.finding_failed || 0}</span><span>Verdict: {result.verdict_posted ? "posted" : "failed"}</span></div></section>;
}

function Ragas({ ragas }) { if (!ragas) return null; const text = ragas.status === "pending" ? "Computing scores in the backgroundâ€¦" : ragas.status === "failed" ? `Scoring failed: ${ragas.error || "provider error"}` : ragas.status === "skipped" ? `Skipped: ${ragas.reason || "retrieval context was unavailable"}` : ragas.status === "unavailable" ? `Unavailable: ${ragas.error || "the scoring task was lost"}` : ragas.status === "completed_no_scores" ? (ragas.error || "Completed without metric values") : Object.entries(ragas.scores || {}).map(([key, value]) => `${key}: ${typeof value === "number" ? value.toFixed(3) : value}`).join(" Â· ") || "Completed without scores"; return <section className="result-card receipt" aria-label="RAGAS retrieval quality"><span className="panel-kicker">Retrieval quality (RAGAS)</span><p>{text}</p></section>; }

function App() {
  const [activeTab, setActiveTab] = useState("review");
  const [repo, setRepo] = useState(""); const [pr, setPr] = useState(""); const [endpoint, setEndpoint] = useState("auto"); const [provider, setProvider] = useState("openai"); const [model, setModel] = useState(""); const [apiKey, setApiKey] = useState(""); const [baseUrl, setBaseUrl] = useState("");
  const [savedProviders, setSavedProviders] = useState(() => { try { return JSON.parse(localStorage.getItem("ocr-providers") || "[]"); } catch { return []; } });
  const [theme, setTheme] = useState(() => localStorage.getItem("ocr-theme") || "light"); const [job, setJob] = useState(null); const [message, setMessage] = useState(""); const [error, setError] = useState(""); const [busy, setBusy] = useState(false); const [user, setUser] = useState(null);
  useEffect(() => { fetch(apiUrl("/api/auth/me"), { credentials: "include" }).then(r => r.ok ? r.json() : null).then(setUser).catch(() => setUser(null)); }, []);
  async function logout() { await fetch(apiUrl("/api/auth/logout"), { method: "POST", credentials: "include" }); setUser(null); }
  async function logout() { await fetch(apiUrl("/api/auth/logout"), { method: "POST", credentials: "include" }); setUser(null); }
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("ocr-theme", theme); }, [theme]);
  const isTerminal = useMemo(() => TERMINAL.has(job?.status), [job?.status]);
  useEffect(() => {
    if (!job?.id || isTerminal) return undefined; let stopped = false;
    const poll = async () => { try { const r = await fetch(apiUrl(`/api/reviews/${job.id}`)); if (!r.ok) throw new Error(`Status check failed (${r.status})`); const next = await r.json(); if (!stopped) { setJob(next); setMessage(next.status === "awaiting_approval" ? "Review complete. Inspect the findings before posting." : `Review ${pretty(next.status)}â€¦`); } } catch (e) { if (!stopped) setError(e.message); } };
    poll(); const timer = window.setInterval(poll, 3000); return () => { stopped = true; window.clearInterval(timer); };
  }, [job?.id, isTerminal]);
  useEffect(() => { localStorage.setItem("ocr-providers", JSON.stringify(savedProviders)); }, [savedProviders]);
  const customProvider = endpoint === "custom" || endpoint.startsWith("saved:");
  async function testProvider() {
    if (!model.trim() || !apiKey.trim()) { setError("Enter a model and API key before testing."); return; }
    setError(""); setMessage("Testing model connection…");
    try {
      const response = await fetch("/api/provider-test", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: provider || "openai", model: model.trim(), apiKey: apiKey.trim(), baseUrl: baseUrl.trim() }) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Provider test failed.");
      setMessage(`Connection verified: ${model.trim()} responded successfully.`);
    } catch (err) { setMessage(""); setError(err.message); }
  }
  function addProvider() {
    if (!model.trim() || !apiKey.trim()) { setError("Enter a model and API key first."); return; }
    const type = provider || "openai";
    const name = `${type} / ${model.trim()}`;
    const item = { name, provider: type, model: model.trim(), apiKey: apiKey.trim(), baseUrl: baseUrl.trim() };
    try {
      const next = [...savedProviders.filter(entry => entry.name !== name), item];
      localStorage.setItem("ocr-providers", JSON.stringify(next));
      setSavedProviders(next); setEndpoint(`saved:${name}`); setError(""); setMessage(`${name} saved. Select it in Review.`);
    } catch { setMessage(""); setError("Browser storage is unavailable. Provider could not be saved."); }
  }
  async function startReview(e) { e.preventDefault(); setError(""); if (apiKey && (!provider || !model)) { setError("Choose a provider and model when using your own API key."); return; } if (!/^[-\w.]+\/[-\w.]+$/.test(repo.trim()) || !Number.isInteger(Number(pr)) || Number(pr) < 1) { setError("Enter a repository as owner/name and a valid PR number."); return; } setBusy(true); setJob(null); setMessage("Queueing reviewâ€¦"); try { const r = await fetch(apiUrl("/api/reviews"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ repo: repo.trim(), pr_number: Number(pr), endpoint: endpoint.startsWith("saved:") ? "auto" : endpoint, ...(endpoint.startsWith("saved:") ? (() => { const selected = savedProviders.find(item => `saved:${item.name}` === endpoint); if (!selected) throw new Error("Select a saved provider."); return { provider: selected.provider, model: selected.model, api_key: selected.apiKey, base_url: selected.baseUrl || undefined }; })() : {}) }) }); const data = await r.json().catch(() => ({})); if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`); setJob(data); setMessage("Review queued. This page will update as analysis progresses."); } catch (e) { setError(API_BASE ? e.message : "Set VITE_REVIEW_API_URL to connect the review worker."); } finally { setBusy(false); } }
  async function decide(action) { if (!job?.id) return; setBusy(true); setError(""); setMessage(action === "approve" ? "Posting approved reviewâ€¦" : "Rejecting reviewâ€¦"); try { const r = await fetch(apiUrl(`/api/reviews/${job.id}/decision`), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }) }); const data = await r.json().catch(() => ({})); if (!r.ok) throw new Error(data.detail || `Decision failed (${r.status})`); setJob(data); setMessage(action === "approve" ? "Review delivery finished." : "Review rejected. Nothing was posted to GitHub."); } catch (e) { setError(e.message); } finally { setBusy(false); } }
  return <div className="shell"><header className="topbar"><div className="brand"><span className="brand-mark">OC</span><span>OpenCodeReview</span></div><div>{user ? <><span className="status-dot">Signed in as {user.login}</span><button className="auth-button" onClick={logout}>Sign out</button></> : <a className="status-dot" href={`${API_BASE}/api/auth/github/start`}>Sign in with GitHub</a>}<span className="status-dot">Human approval required</span><button className="theme-toggle" type="button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Toggle color theme">{theme === "dark" ? "Light" : "Dark"}</button></div></header><main className="content"><section className="hero"><p className="eyebrow">Pull request intelligence</p><h1>Review changes with<br /><em>evidence before action.</em></h1><p className="lede">Parallel analysis for correctness, security, and test coverage. Nothing is posted to GitHub until you approve it.</p></section><section className="intro-band"><div className="intro-copy"><p className="eyebrow">AI code review with a human checkpoint</p><h2>Know what changed<br /><em>before it ships.</em></h2><p>OpenCodeReview traces pull requests back to the diff, checks correctness and security, then gives you the evidence to decide what reaches GitHub.</p><div className="intro-actions"><a className="primary-link" href={user ? "#new-review" : `${API_BASE}/api/auth/github/start`}>Start a review <span>&rarr;</span></a><a className="quiet-link" href="#how-it-works">See how it works</a></div></div><div className="review-preview" aria-label="Review workflow preview"><div className="preview-top"><span className="live-dot"></span><span>REVIEW PIPELINE</span><span className="preview-time">LIVE</span></div><div className="preview-row"><strong>01</strong><span><b>Read the diff</b><small>Files, lines, and context</small></span><i>done</i></div><div className="preview-row"><strong>02</strong><span><b>Cross-check findings</b><small>Evidence, severity, confidence</small></span><i>done</i></div><div className="preview-row active"><strong>03</strong><span><b>Your decision</b><small>Approve, reject, or inspect</small></span><i>waiting</i></div></div></section><section className="trust-strip" id="how-it-works"><div><strong>Evidence first</strong><span>Every finding points to code.</span></div><div><strong>Human controlled</strong><span>Nothing posts without approval.</span></div><div><strong>Measurable quality</strong><span>RAGAS scores show retrieval quality.</span></div></section>{user ? <section className="workspace" id="new-review"><nav className="workspace-tabs" aria-label="Workspace"><button type="button" className={activeTab === "review" ? "active" : ""} onClick={() => setActiveTab("review")}>Review</button><button type="button" className={activeTab === "providers" ? "active" : ""} onClick={() => setActiveTab("providers")}>Providers</button></nav>{activeTab === "providers" && <section className="panel provider-manager" id="providers"><div className="panel-kicker">Provider library</div><h2>Saved providers</h2><p className="provider-hint">Keep your model connections ready for future reviews.</p>{savedProviders.length ? <div className="saved-provider-list">{savedProviders.map(item => <div className="saved-provider" key={item.name}><strong>{item.name}</strong><small>{item.baseUrl || "Default API endpoint"}</small></div>)}</div> : <p className="provider-hint">No providers saved yet.</p>}<label htmlFor="provider-manager-type">Provider type<select id="provider-manager-type" value={provider || "openai"} onChange={e => setProvider(e.target.value)}><option value="openai">OpenAI-compatible</option><option value="anthropic">Anthropic</option><option value="google">Google</option></select></label><label htmlFor="provider-manager-model">Model<input id="provider-manager-model" value={model} onChange={e => setModel(e.target.value)} placeholder="model-name" /></label><label htmlFor="provider-manager-key">API key<input id="provider-manager-key" type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="Used for reviews" /></label><label htmlFor="provider-manager-base">Base URL <span>(optional)</span><input id="provider-manager-base" value={baseUrl} onChange={e => setBaseUrl(e.target.value)} placeholder="https://api.example.com/v1" /></label><div className="provider-actions"><button type="button" onClick={testProvider}>Test provider</button><button type="button" onClick={addProvider}>Add provider <span>ï¼‹</span></button></div>{(error || message) && <div className={`message ${error ? "error" : "success"}`} role="status" aria-live="polite">{error || message}</div>}</section>}<form className={`panel form-panel ${activeTab === "providers" ? "is-hidden" : ""}`} onSubmit={startReview} aria-busy={busy}><div className="panel-kicker">New review</div><h2>Inspect a pull request</h2><label htmlFor="repo">Repository<input id="repo" value={repo} onChange={e => setRepo(e.target.value)} placeholder="owner/repository" autoComplete="off" /></label><label htmlFor="pr">Pull request<input id="pr" value={pr} onChange={e => setPr(e.target.value)} placeholder="42" inputMode="numeric" /></label><label htmlFor="endpoint">Provider<select id="endpoint" value={endpoint} onChange={e => setEndpoint(e.target.value)}><option value="auto">Automatic fallback</option><option value="gemini">Gemini</option><option value="groq">Groq</option>{savedProviders.map(item => <option key={item.name} value={`saved:${item.name}`}>{item.name}</option>)}</select></label><button type="submit" disabled={busy}>{busy && !job ? "Queueing reviewâ€¦" : "Start review"}<span aria-hidden="true">â†—</span></button>{(error || message) && <div className={`message ${error ? "error" : "success"}`} role="status" aria-live="polite">{error || message}</div>}{job && <div className="job-card"><div><strong>Review {job.id?.slice(0, 8)}</strong><small>{job.repo} Â· PR #{job.pr_number}</small></div><span className={`pill ${job.status}`}>{pretty(job.status)}</span></div>}{job && !isTerminal && <div className="progress" role="progressbar" aria-label={`Review ${pretty(job.status)}`}><span /><small>Analysis is running. This may take a few minutes.</small></div>}<div className="results"><Verdict job={job} /><Ragas ragas={job?.ragas} />{job?.findings && <Findings findings={job.findings} />}{job?.status === "awaiting_approval" && <div className="actions"><button type="button" onClick={() => decide("approve")} disabled={busy}>Approve &amp; post</button><button type="button" className="secondary" onClick={() => decide("reject")} disabled={busy}>Reject</button></div>}<Receipt result={job?.posting_result} />{job?.error && <div className="error detail">{job.error}</div>}</div></form><aside className={`panel signal-panel ${activeTab === "providers" ? "is-hidden" : ""}`}><div className="panel-kicker">What you receive</div><div className="signal"><strong>01</strong><span><b>Evidence-grounded findings</b><small>Code locations, severity, and confidence</small></span></div><div className="signal"><strong>02</strong><span><b>Human checkpoint</b><small>Approve or reject before GitHub comments</small></span></div><div className="signal"><strong>03</strong><span><b>Delivery receipt</b><small>See exactly what reached GitHub</small></span></div></aside></section> : <section className="auth-wall"><p className="eyebrow">Private review workspace</p><h2>Sign in to inspect a pull request.</h2><p>Your GitHub identity keeps reviews tied to your account and protects the review results.</p><a className="primary-link" href={`${API_BASE}/api/auth/github/start`}>Continue with GitHub <span>&rarr;</span></a></section>}</main><footer>OpenCodeReview <span>Â·</span> React review workspace <span>Â·</span> API keys stay server-side</footer></div>;
}
createRoot(document.getElementById("root")).render(<App />);

















