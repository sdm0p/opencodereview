export function readProviders(storage) {
  try {
    const entries = JSON.parse(storage.getItem("ocr-providers") || "[]");
    return Array.isArray(entries) ? entries.filter(item => item && typeof item.name === "string" && typeof item.provider === "string" && typeof item.model === "string" && typeof item.apiKey === "string") : [];
  } catch { return []; }
}

export function initialEndpoint(storage) {
  try {
    const name = storage.getItem('ocr-default-provider');
    return readProviders(storage).some(item => item.name === name) ? `saved:${name}` : 'auto';
  } catch { return 'auto'; }
}

export function groupFindings(findings) {
  const groups = {bug: [], security_concern: [], test_suggestion: []};
  findings.forEach((finding, index) => {
    const fallback = finding.category === 'test_coverage' ? 'test_suggestion' : finding.category === 'security' ? 'security_concern' : 'bug';
    groups[Object.hasOwn(groups, finding.kind) ? finding.kind : fallback].push({finding, index});
  });
  return groups;
}

export function formatCost(cost) {
  if (!cost) return null;
  if (typeof cost === 'string') return cost.replace(/^Cost:\s*/i, '');
  if (typeof cost === 'object') {
    const source = cost.source || cost.cost_source || 'unknown';
    const amount = cost.total_cost ?? cost.amount;
    const labels = {provider_reported:'Provider-reported',provider:'Provider-reported',measured:'Provider-reported',estimated:'Estimated, not a confirmed charge',free_model_estimate:'Free-model estimate, not a confirmed charge',mixed:'Mixed reported and estimated cost',unknown:'Cost information'};
    return `${labels[source] || 'Cost information'}${typeof amount === 'number' ? `: $${amount.toFixed(4)}` : ': unavailable'}${cost.complete === false ? ' · partial usage data' : ''}`;
  }
  return 'Cost unavailable';
}

export function upsertProvider(entries, item, previousName = null) {
  return [...entries.filter(entry => entry.name !== item.name && entry.name !== previousName), item];
}

export function reviewPayload(repo, pr, endpoint, savedProviders, reviewMode, baselineSha = "") {
  const builtinEndpoint = {gemini: 'Gemini', groq: 'Grok'}[endpoint] || endpoint;
  const payload = { repo: repo.trim(), pr_number: Number(pr), endpoint: builtinEndpoint, review_mode: reviewMode };
  if (baselineSha.trim()) {
    if (!/^[a-fA-F0-9]{40}$/.test(baselineSha.trim())) throw new Error("Enter the full 40-character baseline commit SHA.");
    payload.baseline_sha = baselineSha.trim().toLowerCase();
  }
  if (endpoint.startsWith("saved:")) {
    const selected = savedProviders.find(item => `saved:${item.name}` === endpoint);
    if (!selected) throw new Error("Select a saved provider before starting the review.");
    if (!selected.apiKey.trim() || !selected.model.trim()) throw new Error("This saved provider needs a model and API key. Update it in Providers.");
    Object.assign(payload, { endpoint: "auto", provider: selected.provider, model: selected.model.trim(), api_key: selected.apiKey.trim(), base_url: selected.baseUrl?.trim() || undefined });
  }
  return payload;
}
