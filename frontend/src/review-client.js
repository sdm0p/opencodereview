export function readProviders(storage) {
  try {
    const entries = JSON.parse(storage.getItem("ocr-providers") || "[]");
    return Array.isArray(entries) ? entries.filter(item => item && typeof item.name === "string" && typeof item.provider === "string" && typeof item.model === "string" && typeof item.apiKey === "string") : [];
  } catch { return []; }
}

export function reviewPayload(repo, pr, endpoint, savedProviders, reviewMode, baselineSha = "") {
  const payload = { repo: repo.trim(), pr_number: Number(pr), endpoint, review_mode: reviewMode };
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
