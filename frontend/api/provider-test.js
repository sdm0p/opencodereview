export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') return res.status(405).json({error:'Use POST.'});
  const {provider = 'openai', model, apiKey, baseUrl} = req.body || {};
  if (!model?.trim() || !apiKey?.trim()) return res.status(400).json({error:'Model and API key are required.'});
  try {
    let url, headers, body;
    if (provider === 'google') {
      url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
      headers = {'x-goog-api-key':apiKey};
      body = {contents:[{parts:[{text:'Reply OK.'}]}],generationConfig:{maxOutputTokens:16}};
    } else if (provider === 'anthropic') {
      url = 'https://api.anthropic.com/v1/messages';
      headers = {'x-api-key':apiKey,'anthropic-version':'2023-06-01'};
      body = {model,max_tokens:16,messages:[{role:'user',content:'Reply OK.'}]};
    } else {
      const base = new URL(baseUrl || 'https://api.openai.com/v1');
      if (base.protocol !== 'https:' || !['openrouter.ai','api.openai.com','api.groq.com','api.deepseek.com'].includes(base.hostname) || base.port || base.username || base.password) return res.status(400).json({error:'Use an HTTPS API URL from OpenRouter, OpenAI, Groq, or DeepSeek.'});
      url = `${base.origin}${base.pathname.replace(/\/$/,'')}/chat/completions`;
      headers = {Authorization:`Bearer ${apiKey}`};
      body = {model,max_tokens:16,messages:[{role:'user',content:'Reply OK.'}]};
    }
    const response = await fetch(url,{method:'POST',redirect:'error',signal:AbortSignal.timeout(20000),headers:{'Content-Type':'application/json',...headers},body:JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) return res.status(400).json({error:`Provider returned ${response.status}: ${String(data.error?.message || 'Request rejected').replaceAll(apiKey,'[redacted]').slice(0,400)}`});
    if (!data.choices?.length && !data.content?.length && !data.candidates?.length) return res.status(502).json({error:'Provider returned no model response.'});
    return res.status(200).json({ok:true,model});
  } catch { return res.status(502).json({error:'Connection failed or timed out. Check the provider URL and try again.'}); }
}
