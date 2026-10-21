import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { reviewPayload, readProviders, initialEndpoint, groupFindings, formatCost, upsertProvider } from "./review-client.js";

const API_BASE = (import.meta.env.VITE_REVIEW_API_URL || "").replace(/\/$/, "");
const TERMINAL = new Set(["awaiting_approval", "rejected", "posted", "partially_posted", "post_failed", "failed", "incomplete", "cancelled"]);
const apiUrl = (path) => `${API_BASE}${path}`;
const pretty = (value) => String(value || "").replaceAll("_", " ");

function Verdict({ job }) {
  if (!job?.verdict) return null;
  const v = job.verdict;
  const score = typeof v.overall_score === "number" ? v.overall_score : null;
  const scoreLabel = score == null ? "Not scored" : score < 4 ? "High risk" : score < 7 ? "Review carefully" : "Low risk";
  const cost = formatCost(job.cost_details || job.cost);
  return <section className={`result-card verdict ${["block", "request_changes", "incomplete"].includes(v.recommendation) ? "blocking" : "approved"}`} aria-labelledby="verdict-title">
    <div className="result-title"><div><span className="panel-kicker">Review decision</span><h3 id="verdict-title">{pretty(v.recommendation).toUpperCase()}</h3></div><div className="score-block"><strong className="score">{score == null ? "—" : `${score.toFixed(1)}/10`}</strong><span>{scoreLabel}</span></div></div>
    <p>{v.summary || "The review is ready for your decision."}</p>
    <div className="score-explanation"><strong>Risk estimate, not answer accuracy</strong><span>{v.score_explanation || (score == null ? "A reliable score is unavailable because review stages did not finish." : "The score reflects reported code risks. Test suggestions and unverified claims should be assessed separately.")}</span></div>
    {v.score_breakdown && Object.keys(v.score_breakdown).length > 0 && <details><summary>How the score was calculated</summary><ul>{Object.entries(v.score_breakdown).map(([key,value]) => <li key={key}>{pretty(key)}: {typeof value === "object" ? JSON.stringify(value) : String(value)}</li>)}</ul></details>}
    <div className="metrics"><span>{job.findings?.length || 0} findings and suggestions</span>{cost && <span>{cost}</span>}</div>
  </section>;
}

function Findings({ findings = [], feedback = {}, onFeedback, feedbackBusy }) {
  const groups = groupFindings(findings);
  const titles = {bug: "Bugs", security_concern: "Security concerns", test_suggestion: "Suggested tests"};
  return <section aria-labelledby="findings-title"><div className="section-heading"><h3 id="findings-title">Review findings</h3><span>{findings.length}</span></div>{findings.length ? Object.entries(groups).map(([kind, entries]) => entries.length > 0 && <section className="finding-group" key={kind} aria-label={titles[kind]}><h4>{titles[kind]} <span>{entries.length}</span></h4>{entries.map(({finding:f,index}) => <article className="finding" key={`${f.file_path}-${index}`}>
    <div className="finding-meta"><span className={`severity ${String(f.severity || "info").toLowerCase()}`}>{f.severity || "info"}</span><span className="finding-location">{f.file_path || "Unknown file"}{f.line_start ? `:${f.line_start}${f.line_end && f.line_end !== f.line_start ? `–${f.line_end}` : ""}` : ""}</span></div>
    <p>{f.comment || f.description || "No explanation provided."}</p>
    <span className={`evidence-label ${f.evidence_status === "source_grounded" ? "grounded" : "uncertain"}`}>{f.evidence_status === "source_grounded" ? "Source location checked · claim needs your review" : f.evidence_status === "outside_diff" ? "Outside reviewed changes · needs verification" : "Needs verification"}</span>
    {f.calibration_note && <p className="calibration-note">{f.calibration_note}</p>}
    {f.evidence_excerpt && <details><summary>Supporting source excerpt</summary><pre className="evidence-excerpt">{f.evidence_excerpt}</pre></details>}
    {f.suggested_fix && <details><summary>Suggested fix</summary><p>{f.suggested_fix}</p></details>}
    {f.confidence != null && <small>Model confidence (uncalibrated) {Math.round(Number(f.confidence) * 100)}% · not independently measured</small>}
    {onFeedback && <div className="finding-feedback" aria-label={`Feedback for finding ${index + 1}`}><span>Was this helpful?</span>{["correct","incorrect","useful"].map(label => <button type="button" key={label} disabled={feedbackBusy === index} aria-pressed={feedback[String(index)] === label} onClick={() => onFeedback(index,label)}>{label === "useful" ? "Useful suggestion" : pretty(label)}</button>)}{feedback[String(index)] && <small>Feedback saved</small>}</div>}
  </article>)}</section>) : <div className="empty-result"><strong>No findings returned</strong><span>This does not prove the changes are free of issues. Check the review coverage and reviewer statuses.</span></div>}</section>;
}

function Coverage({ coverage }) {
  if (!coverage) return null;
  return <section className="result-card review-coverage" aria-label="Review coverage"><span className="panel-kicker">What was reviewed</span><p>{coverage.files_available ?? coverage.files_changed ?? "Unknown"} files available · {coverage.incremental ? "incremental preview · " : ""}{pretty(coverage.mode || "review")} mode · {coverage.diff_chars ?? "Unknown"} diff characters</p>{coverage.changed_file_paths?.length > 0 && <details><summary>Files in scope</summary><ul>{coverage.changed_file_paths.map(path => <li key={path}>{path}</li>)}</ul></details>}{coverage.limitations?.length > 0 && <ul>{coverage.limitations.map((note,index) => <li key={index}>{note}</li>)}</ul>}{coverage.reviewers?.length > 0 && <details><summary>Reviewer coverage</summary><ul>{coverage.reviewers.map((reviewer,index) => <li key={index}>{pretty(reviewer.reviewer || reviewer.name)}: {pretty(reviewer.status)}{reviewer.diff_chars_included != null ? ` · ${reviewer.diff_chars_included} diff characters included` : ""}{reviewer.limitations?.map((note,i) => <p key={i}>{note}</p>)}</li>)}</ul></details>}</section>;
}

function Receipt({ result }) {
  if (!result) return null;
  return <section className="result-card receipt" aria-labelledby="receipt-title"><span className="panel-kicker">Delivery receipt</span><h3 id="receipt-title">{pretty(result.status || "completed")}</h3><div className="metrics"><span>Inline: {result.inline_posted || 0}</span><span>Fallback: {result.fallback_posted || 0}</span><span>Finding failures: {result.finding_failed || 0}</span><span>Unverified findings withheld: {result.withheld_unverified || 0}</span><span>Verdict: {result.verdict_posted ? "posted" : "failed"}</span></div></section>;
}

function Ragas({ ragas }) {
  if (!ragas) return null;
  const scores = ragas.scores || {};
  const metrics = ragas.metrics || {};
  const text = ragas.status === "pending" ? "Evaluating in the background…" : ragas.status === "failed" ? `Evaluation failed: ${ragas.error || "provider error"}` : ragas.status === "skipped" ? `Skipped: ${ragas.reason || "evaluation unavailable"}` : ragas.status === "unavailable" ? `Unavailable: ${ragas.error || "the scoring task was interrupted"}` : ragas.error || pretty(ragas.status);
  return <section className="result-card receipt" aria-label="RAGAS evaluation"><span className="panel-kicker">Review evaluation (RAGAS)</span><p>{text}</p><p className="provider-hint">These metrics assess specific aspects of the response; they do not establish whether every code finding is correct.</p>{Object.keys(metrics).length > 0 ? <ul className="evaluation-metrics">{Object.entries(metrics).map(([key,metric]) => <li key={key}><strong>{pretty(key)}</strong><span>{pretty(metric.status)}{typeof metric.value === "number" ? ` · ${metric.value.toFixed(3)}` : ""}{metric.duration_seconds != null ? ` · ${Number(metric.duration_seconds).toFixed(1)}s` : ""}</span>{metric.error && <small>{metric.error}</small>}</li>)}</ul> : Object.keys(scores).length > 0 && <ul>{Object.entries(scores).map(([key,value]) => <li key={key}>{pretty(key)}: {typeof value === "number" ? value.toFixed(3) : String(value)}</li>)}</ul>}{ragas.evaluation && <details><summary>Evaluation configuration</summary><pre className="evidence-excerpt">{JSON.stringify(ragas.evaluation,null,2)}</pre></details>}</section>;
}

function ReviewProgress({ job }) {
  if (!job) return null;
  const stages = job.review_stages || [];
  const timings = job.stage_timings || {};
  return <section className="review-progress" aria-label="Review progress" aria-live="polite">
    {job.status === "incomplete" && <p className="message error">Review incomplete. Failed reviewers do not mean the code is free of issues.</p>}
    <div className="metrics">{job.elapsed_seconds != null && <span>Elapsed: {Math.round(job.elapsed_seconds)}s</span>}{job.review_mode && <span>{pretty(job.review_mode)} review</span>}{job.progress?.completed_stages?.length > 0 && <span>{job.progress.completed_stages.length} stages finished</span>}</div>
    {stages.length > 0 && <ul className="reviewer-stages">{stages.map((stage, index) => <li key={`${stage.reviewer}-${index}`}><strong>{pretty(stage.reviewer)}</strong><span>{pretty(stage.status)}{stage.duration_seconds != null ? ` · ${stage.duration_seconds.toFixed(1)}s` : ""}</span>{stage.model && <small>{stage.model}</small>}{stage.reason && <small>{stage.reason}</small>}{stage.error && <small className="reviewer-error">{stage.error}</small>}</li>)}</ul>}
    {Object.keys(timings).length > 0 && <details><summary>Stage completion times (since review start)</summary><ul>{Object.entries(timings).map(([name, seconds]) => <li key={name}>{pretty(name)}: {Number(seconds).toFixed(1)}s</li>)}</ul></details>}
  </section>;
}

function App() {
  const [activeTab, setActiveTab] = useState("review");
  const [reviewMode, setReviewMode] = useState("fast");
  const [baselineSha, setBaselineSha] = useState("");
  const [testingProvider, setTestingProvider] = useState(false);
  const [editingProvider, setEditingProvider] = useState(null);
  const [lastTest, setLastTest] = useState(null);
  const [defaultProvider, setDefaultProvider] = useState(() => { try { return localStorage.getItem("ocr-default-provider") || ""; } catch { return ""; } });
  const [feedbackBusy, setFeedbackBusy] = useState(null);

  const [repo, setRepo] = useState(""); const [pr, setPr] = useState(""); const [endpoint, setEndpoint] = useState(() => initialEndpoint(localStorage)); const [provider, setProvider] = useState("openai"); const [model, setModel] = useState(""); const [apiKey, setApiKey] = useState(""); const [baseUrl, setBaseUrl] = useState("");
  const [savedProviders, setSavedProviders] = useState(() => readProviders(localStorage));
  const [theme, setTheme] = useState(() => localStorage.getItem("ocr-theme") || "light"); const [job, setJob] = useState(null); const [message, setMessage] = useState(""); const [error, setError] = useState(""); const [busy, setBusy] = useState(false); const [user, setUser] = useState(null);
  useEffect(() => { fetch(apiUrl("/api/auth/me"), { credentials: "include" }).then(r => r.ok ? r.json() : null).then(setUser).catch(() => setUser(null)); }, []);
  async function logout() { await fetch(apiUrl("/api/auth/logout"), { method: "POST", credentials: "include" }); setUser(null); }
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("ocr-theme", theme); }, [theme]);
  const isTerminal = useMemo(() => TERMINAL.has(job?.status), [job?.status]);
  useEffect(() => {
    if (!job?.id || (isTerminal && job?.ragas?.status !== "pending")) return undefined; let stopped = false;
    const poll = async () => { try { const r = await fetch(apiUrl(`/api/reviews/${job.id}`)); if (!r.ok) throw new Error(`Status check failed (${r.status})`); const next = await r.json(); if (!stopped) { setJob(next); setMessage(next.status === "awaiting_approval" ? "Review complete. Inspect the findings before posting." : next.status === "incomplete" ? "Review incomplete. Some reviewers could not finish; these findings are partial." : next.status === "cancelled" ? "Review cancelled." : `Review ${pretty(next.status)}…`); } } catch (e) { if (!stopped) setError(e.message); } };
    poll(); const timer = window.setInterval(poll, 3000); return () => { stopped = true; window.clearInterval(timer); };
  }, [job?.id, isTerminal, job?.ragas?.status]);

  function providerDraft() {
    return {name: `${provider || "openai"} / ${model.trim()}`, provider: provider || "openai", model: model.trim(), apiKey: apiKey.trim(), baseUrl: baseUrl.trim()};
  }
  function saveEntries(entries) {
    localStorage.setItem("ocr-providers", JSON.stringify(entries)); setSavedProviders(entries);
  }
  function editProvider(item) {
    setEditingProvider(item.name); setProvider(item.provider); setModel(item.model); setApiKey(item.apiKey); setBaseUrl(item.baseUrl || ""); setLastTest(item.lastTest || null); setError(""); setMessage(`Editing ${item.name}. Changes are saved when you click Save changes.`);
  }
  function removeProvider(item) {
    try {
      saveEntries(savedProviders.filter(entry => entry.name !== item.name));
      if (defaultProvider === item.name) { localStorage.removeItem("ocr-default-provider"); setDefaultProvider(""); }
      if (endpoint === `saved:${item.name}`) setEndpoint("auto");
      if (editingProvider === item.name) resetProviderDraft();
      setError(""); setMessage(`${item.name} removed from this browser.`);
    } catch { setError("Could not remove provider from browser storage."); }
  }
  function makeDefault(item) {
    try { localStorage.setItem("ocr-default-provider",item.name); setDefaultProvider(item.name); setEndpoint(`saved:${item.name}`); setError(""); setMessage(`${item.name} is the default for new reviews.`); }
    catch { setError("Could not save the default provider."); }
  }
  function resetProviderDraft() { setEditingProvider(null); setModel(""); setApiKey(""); setBaseUrl(""); setProvider("openai"); setLastTest(null); setError(""); setMessage(""); }
  async function testProvider() {
    if (!model.trim() || !apiKey.trim()) { setError("Enter a model and API key before testing."); return; }
    const draft = providerDraft();
    setError(""); setMessage("Testing response time and JSON output…"); setTestingProvider(true);
    try {
      const response = await fetch("/api/provider-test", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({provider:draft.provider, model:draft.model, apiKey:draft.apiKey, baseUrl:draft.baseUrl}) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Provider test failed.");
      const tested = {at: new Date().toISOString(), latency_ms:result.latency_ms, json_valid:result.json_object_valid, fingerprint:JSON.stringify(draft)};
      setLastTest(tested);
      const saved = savedProviders.find(item => item.name === draft.name && item.apiKey === draft.apiKey && item.baseUrl === draft.baseUrl);
      if (saved) saveEntries(savedProviders.map(item => item.name === saved.name ? {...item,lastTest:{at:tested.at,latency_ms:tested.latency_ms,json_valid:tested.json_valid}} : item));
      setMessage(`Connection verified in ${(result.latency_ms / 1000).toFixed(1)}s. JSON response: ${result.json_object_valid ? "valid" : "invalid — this model may need a different output format"}.`);
    } catch (err) { setLastTest(null); setMessage(""); setError(err.message); } finally { setTestingProvider(false); }
  }
  function addProvider() {
    if (!model.trim() || !apiKey.trim()) { setError("Enter a model and API key first."); return; }
    const item = providerDraft();
    const freshTest = lastTest?.fingerprint === JSON.stringify(item) ? lastTest : null;
    if (freshTest) item.lastTest = {at:freshTest.at,latency_ms:freshTest.latency_ms,json_valid:freshTest.json_valid};
    const previous = savedProviders.find(entry => entry.name === editingProvider);
    if (!freshTest && previous && previous.provider === item.provider && previous.model === item.model && previous.apiKey === item.apiKey && (previous.baseUrl || "") === item.baseUrl) item.lastTest = previous.lastTest;
    try {
      saveEntries(upsertProvider(savedProviders,item,editingProvider));
      if (editingProvider && defaultProvider === editingProvider) { localStorage.setItem("ocr-default-provider",item.name); setDefaultProvider(item.name); }
      setEditingProvider(item.name); setEndpoint(`saved:${item.name}`); setError(""); setMessage(`${item.name} saved. Select it in Review.`);
    } catch { setMessage(""); setError("Browser storage is unavailable. Provider could not be saved."); }
  }
  async function saveFeedback(index,label) {
    if (!job?.id) return;
    setFeedbackBusy(index); setError("");
    try {
      const response = await fetch(apiUrl(`/api/reviews/${job.id}/feedback`), {method:"POST",credentials:"include",headers:{"Content-Type":"application/json"},body:JSON.stringify({finding_index:index,label})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not save feedback.");
      setJob(current => current?.id === job.id ? {...current,feedback:result.feedback} : current);
    } catch(err) { setError(err.message); } finally { setFeedbackBusy(null); }
  }
  async function startReview(e) {
    e.preventDefault(); setError("");
    if (!/^[-\w.]+\/[-\w.]+$/.test(repo.trim()) || !Number.isInteger(Number(pr)) || Number(pr) < 1) { setError("Enter a repository as owner/name and a valid PR number."); return; }
    setBusy(true); setJob(null); setMessage("Queueing review…");
    try {
      const response = await fetch(apiUrl("/api/reviews"), { method: "POST", credentials: "include", headers: { "Content-Type": "application/json" }, body: JSON.stringify(reviewPayload(repo, pr, endpoint, savedProviders, reviewMode, baselineSha)) });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
      setJob(data); setMessage(data.cache_hit ? "Previous review reused for this exact commit and configuration." : data.reused ? "An existing review is already running. Following its progress." : "Review queued. Findings appear as reviewers finish.");
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  async function cancelReview() {
    if (!job?.id) return;
    setBusy(true); setError("");
    try {
      const response = await fetch(apiUrl(`/api/reviews/${job.id}/cancel`), { method: "POST", credentials: "include" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Could not cancel review.");
      setJob(data); setMessage("Cancellation requested. A model call already in progress may finish before the worker stops.");
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  async function decide(action) { if (!job?.id) return; setBusy(true); setError(""); setMessage(action === "approve" ? "Posting approved review…" : "Rejecting review…"); try { const r = await fetch(apiUrl(`/api/reviews/${job.id}/decision`), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }) }); const data = await r.json().catch(() => ({})); if (!r.ok) throw new Error(data.detail || `Decision failed (${r.status})`); setJob(data); setMessage(action === "approve" ? "Review delivery finished." : "Review rejected. Nothing was posted to GitHub."); } catch (e) { setError(e.message); } finally { setBusy(false); } }
  return <div className="shell"><header className="topbar"><div className="brand"><span className="brand-mark">OC</span><span>OpenCodeReview</span></div><div>{user ? <><span className="status-dot">Signed in as {user.login}</span><button className="auth-button" onClick={logout}>Sign out</button></> : <a className="status-dot" href={`${API_BASE}/api/auth/github/start`}>Sign in with GitHub</a>}<span className="status-dot">Human approval required</span><button className="theme-toggle" type="button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Toggle color theme">{theme === "dark" ? "Light" : "Dark"}</button></div></header><main className="content"><section className="hero"><p className="eyebrow">Pull request intelligence</p><h1>Review changes with<br /><em>evidence before action.</em></h1><p className="lede">Parallel analysis for correctness, security, and test coverage. Nothing is posted to GitHub until you approve it.</p></section><section className="intro-band"><div className="intro-copy"><p className="eyebrow">AI code review with a human checkpoint</p><h2>Know what changed<br /><em>before it ships.</em></h2><p>OpenCodeReview traces pull requests back to the diff, checks correctness and security, then gives you the evidence to decide what reaches GitHub.</p><div className="intro-actions"><a className="primary-link" href={user ? "#new-review" : `${API_BASE}/api/auth/github/start`}>Start a review <span>&rarr;</span></a><a className="quiet-link" href="#how-it-works">See how it works</a></div></div><div className="review-preview" aria-label="Review workflow preview"><div className="preview-top"><span className="live-dot"></span><span>REVIEW PIPELINE</span><span className="preview-time">LIVE</span></div><div className="preview-row"><strong>01</strong><span><b>Read the diff</b><small>Files, lines, and context</small></span><i>done</i></div><div className="preview-row"><strong>02</strong><span><b>Cross-check findings</b><small>Evidence, severity, confidence</small></span><i>done</i></div><div className="preview-row active"><strong>03</strong><span><b>Your decision</b><small>Approve, reject, or inspect</small></span><i>waiting</i></div></div></section><section className="trust-strip" id="how-it-works"><div><strong>Evidence first</strong><span>Every finding points to code.</span></div><div><strong>Human controlled</strong><span>Nothing posts without approval.</span></div><div><strong>Measurable quality</strong><span>RAGAS scores show retrieval quality.</span></div></section>{user ? <section className="workspace" id="new-review"><nav className="workspace-tabs" aria-label="Workspace"><button type="button" className={activeTab === "review" ? "active" : ""} onClick={() => { setActiveTab("review"); setError(""); setMessage(""); }}>Review</button><button type="button" className={activeTab === "providers" ? "active" : ""} onClick={() => { setActiveTab("providers"); setError(""); setMessage(""); }}>Providers</button></nav>{activeTab === "providers" && <section className="panel provider-manager" id="providers"><div className="panel-kicker">Provider library</div><h2>Saved providers</h2><p className="provider-hint">Keep your model connections ready for future reviews.</p>{savedProviders.length ? <div className="saved-provider-list">{savedProviders.map(item => <div className="saved-provider" key={item.name}><div className="provider-entry-title"><strong>{item.name}</strong>{defaultProvider === item.name && <span className="pill">Default</span>}</div><small>{item.baseUrl || "Default API endpoint"}</small><small>{item.lastTest ? `Last successful connection: ${new Date(item.lastTest.at).toLocaleString()} · ${(Number(item.lastTest.latency_ms) / 1000).toFixed(1)}s · JSON ${item.lastTest.json_valid ? "valid" : "invalid"}` : "Connection has not been tested"}</small><div className="provider-entry-actions"><button type="button" onClick={() => editProvider(item)} aria-label={`Edit ${item.name}`}>Edit</button><button type="button" onClick={() => makeDefault(item)} disabled={defaultProvider === item.name} aria-label={`Make ${item.name} default`}>Make default</button><button type="button" onClick={() => removeProvider(item)} aria-label={`Delete ${item.name}`}>Delete</button></div></div>)}</div> : <p className="provider-hint">No providers saved yet.</p>}<div className="provider-form-heading"><h3>{editingProvider ? "Edit provider" : "Add a provider"}</h3>{editingProvider && <button type="button" onClick={resetProviderDraft}>New provider</button>}</div><p className="provider-hint">Saved on this browser. Keys are hidden from the list and sent only when a provider is tested or used.</p><label htmlFor="provider-manager-type">Provider type<select id="provider-manager-type" value={provider || "openai"} onChange={e => setProvider(e.target.value)}><option value="openai">OpenAI-compatible</option><option value="anthropic">Anthropic</option><option value="google">Google</option></select></label><label htmlFor="provider-manager-model">Model<input id="provider-manager-model" value={model} onChange={e => setModel(e.target.value)} placeholder="model-name" /></label><label htmlFor="provider-manager-key">API key<input id="provider-manager-key" type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="Used for reviews" /></label><label htmlFor="provider-manager-base">Base URL <span>(optional)</span><input id="provider-manager-base" value={baseUrl} onChange={e => setBaseUrl(e.target.value)} placeholder="https://api.example.com/v1" /></label><div className="provider-actions"><button type="button" onClick={testProvider} disabled={testingProvider}>{testingProvider ? "Testing…" : "Test speed & JSON"}</button><button type="button" onClick={addProvider}>{editingProvider ? "Save changes" : "Add provider"}<span>＋</span></button></div>{(error || message) && <div className={`message ${error ? "error" : "success"}`} role="status" aria-live="polite">{error || message}</div>}</section>}<form className={`panel form-panel ${activeTab === "providers" ? "is-hidden" : ""}`} onSubmit={startReview} aria-busy={busy}><div className="panel-kicker">New review</div><h2>Inspect a pull request</h2><label htmlFor="repo">Repository<input id="repo" value={repo} onChange={e => setRepo(e.target.value)} placeholder="owner/repository" autoComplete="off" /></label><label htmlFor="pr">Pull request<input id="pr" value={pr} onChange={e => setPr(e.target.value)} placeholder="42" inputMode="numeric" /></label><label htmlFor="endpoint">Provider<select id="endpoint" value={endpoint} onChange={e => setEndpoint(e.target.value)}><option value="auto">Automatic fallback</option><option value="gemini">Gemini</option><option value="groq">Groq</option>{savedProviders.map(item => <option key={item.name} value={`saved:${item.name}`}>{item.name}</option>)}</select></label><fieldset className="review-mode"><legend>Review depth</legend><label><input type="radio" name="review-mode" value="fast" checked={reviewMode === "fast"} onChange={() => setReviewMode("fast")} />Fast<span>Smaller prompts and response budget</span></label><label><input type="radio" name="review-mode" value="thorough" checked={reviewMode === "thorough"} onChange={() => setReviewMode("thorough")} />Thorough<span>More context and analysis</span></label></fieldset><details className="incremental-options"><summary>Review only changes since a commit</summary><label htmlFor="baseline-sha">Previously reviewed commit SHA<input id="baseline-sha" value={baselineSha} onChange={e => setBaselineSha(e.target.value)} placeholder="40-character commit SHA (optional)" pattern="[a-fA-F0-9]{40}" autoComplete="off" /></label><p className="provider-hint">When set, this reviews only changes since that commit, not the whole pull request.</p></details><button type="submit" disabled={busy}>{busy && !job ? "Queueing review…" : "Start review"}<span aria-hidden="true">↗</span></button>{(error || message) && <div className={`message ${error ? "error" : "success"}`} role="status" aria-live="polite">{error || message}</div>}{job && <div className="job-card"><div><strong>Review {job.id?.slice(0, 8)}</strong><small>{job.repo} · PR #{job.pr_number}</small></div><span className={`pill ${job.status}`}>{pretty(job.status)}</span></div>}{job && !isTerminal && <div className="progress" role="progressbar" aria-label={`Review ${pretty(job.status)}`}><span /><small>{job.queue_position ? `Queue position: ${job.queue_position}` : `Current stage: ${pretty(job.progress?.stage || job.status)}`}</small><button type="button" className="secondary" onClick={cancelReview} disabled={busy}>Cancel review</button></div>}<div className="results"><ReviewProgress job={job} /><Verdict job={job} /><Coverage coverage={job?.review_coverage || job?.coverage} /><Ragas ragas={job?.ragas} />{job?.findings?.length > 0 && <Findings findings={job.findings} feedback={job.feedback} onFeedback={isTerminal ? saveFeedback : undefined} feedbackBusy={feedbackBusy} />}{job?.status === "awaiting_approval" && job?.findings?.length === 0 && <Findings findings={[]} />}{job?.status === "awaiting_approval" && <div className="actions"><button type="button" onClick={() => decide("approve")} disabled={busy}>Approve &amp; post</button><button type="button" className="secondary" onClick={() => decide("reject")} disabled={busy}>Reject</button></div>}<Receipt result={job?.posting_result} />{job?.error && <div className="error detail">{job.error}</div>}</div></form><aside className={`panel signal-panel ${activeTab === "providers" ? "is-hidden" : ""}`}><div className="panel-kicker">What you receive</div><div className="signal"><strong>01</strong><span><b>Evidence-grounded findings</b><small>Code locations, severity, and confidence</small></span></div><div className="signal"><strong>02</strong><span><b>Human checkpoint</b><small>Approve or reject before GitHub comments</small></span></div><div className="signal"><strong>03</strong><span><b>Delivery receipt</b><small>See exactly what reached GitHub</small></span></div></aside></section> : <section className="auth-wall"><p className="eyebrow">Private review workspace</p><h2>Sign in to inspect a pull request.</h2><p>Your GitHub identity keeps reviews tied to your account and protects the review results.</p><a className="primary-link" href={`${API_BASE}/api/auth/github/start`}>Continue with GitHub <span>&rarr;</span></a></section>}</main><footer>OpenCodeReview <span>·</span> React review workspace <span>·</span> API keys stay server-side</footer></div>;
}
createRoot(document.getElementById("root")).render(<App />);

















