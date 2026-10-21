export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') return res.status(405).json({error:'Use POST.'});
  const {provider = 'openai', model, apiKey, baseUrl} = req.body || {};
  if (!model?.trim() || !apiKey?.trim()) return res.status(400).json({error:'Model and API key are required.'});
  const started = Date.now();
  const prompt = 'Return exactly one JSON object with this shape: {"findings": []}. Do not include markdown.';
  try {
    let url, headers, body;
    if (provider === 'google') {
      url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
      headers = {'x-goog-api-key':apiKey};
      body = {contents:[{parts:[{text:prompt}]}],generationConfig:{maxOutputTokens:256}};
    } else if (provider === 'anthropic') {
      url = 'https://api.anthropic.com/v1/messages';
      headers = {'x-api-key':apiKey,'anthropic-version':'2023-06-01'};
      body = {model,max_tokens:256,messages:[{role:'user',content:prompt}]};
    } else {
      const base = new URL(baseUrl || 'https://api.openai.com/v1');
      if (base.protocol !== 'https:' || !['openrouter.ai','api.openai.com','api.groq.com','api.deepseek.com'].includes(base.hostname) || base.port || base.username || base.password) return res.status(400).json({error:'Use an HTTPS API URL from OpenRouter, OpenAI, Groq, or DeepSeek.'});
      url = `${base.origin}${base.pathname.replace(/\/$/,'')}/chat/completions`;
      headers = {Authorization:`Bearer ${apiKey}`};
      body = {model,max_tokens:256,messages:[{role:'user',content:prompt}]};
      if (base.hostname === 'openrouter.ai' && model.toLowerCase().includes('minimax-m3.1')) {
        body.max_tokens = 8192;
        body.reasoning = {effort:'low'};
      }
    }
    const response = await fetch(url,{method:'POST',redirect:'error',signal:AbortSignal.timeout(20000),headers:{'Content-Type':'application/json',...headers},body:JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) return res.status(400).json({error:`Provider returned ${response.status}: ${String(data.error?.message || 'Request rejected').replaceAll(apiKey,'[redacted]').slice(0,400)}`});
    if (!data.choices?.length && !data.content?.length && !data.candidates?.length) return res.status(502).json({error:'Provider returned no model response.'});
    const content = data.choices?.[0]?.message?.content || data.content?.filter(item => item.type === 'text').map(item => item.text).join('') || data.candidates?.[0]?.content?.parts?.map(item => item.text || '').join('') || '';
    let jsonObjectValid = false;
    try {
      const value = JSON.parse(content.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, ''));
      jsonObjectValid = value !== null && !Array.isArray(value) && typeof value === 'object' && Array.isArray(value.findings);
    } catch { /* Report formatting failure separately from connectivity. */ }
    return res.status(200).json({ok:true,model,latency_ms:Date.now()-started,json_object_valid:jsonObjectValid});
  } catch { return res.status(502).json({error:'Connection failed or timed out. Check the provider URL and try again.'}); }
}
