import test from 'node:test';
import assert from 'node:assert/strict';
import handler from './provider-test.js';

function response() {
  return { code:null,body:null,setHeader(){},status(code){this.code=code;return this;},json(body){this.body=body;return this;} };
}
const request = () => ({method:'POST',body:{provider:'openai',model:'test-model',apiKey:'test-secret',baseUrl:'https://openrouter.ai/api/v1'}});

test('probe sends selected model auth and records real JSON validation and latency', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, 'https://openrouter.ai/api/v1/chat/completions');
      assert.equal(options.headers.Authorization, 'Bearer test-secret');
      assert.equal(JSON.parse(options.body).model, 'test-model');
      return {ok:true,json:async()=>({choices:[{message:{content:'{"findings":[]}'}}]})};
    };
    const res = response(); await handler(request(),res);
    assert.equal(res.code,200); assert.equal(res.body.json_object_valid,true); assert.ok(res.body.latency_ms>=0);
    globalThis.fetch = async () => ({ok:true,json:async()=>({choices:[{message:{content:'[]'}}]})});
    await handler(request(),res); assert.equal(res.body.json_object_valid,false);
  } finally { globalThis.fetch = original; }
});

test('probe rejects private destinations and redacts provider error credentials', async () => {
  const original = globalThis.fetch;
  try {
    const req=request(),res=response(); req.body.baseUrl='https://127.0.0.1/v1';
    globalThis.fetch=async()=>{throw new Error('must not fetch private address');};
    await handler(req,res); assert.equal(res.code,400);
    req.body.baseUrl='https://openrouter.ai/api/v1';
    globalThis.fetch=async()=>({ok:false,status:401,json:async()=>({error:{message:'bad test-secret'}})});
    await handler(req,res); assert.equal(res.code,400); assert.ok(!res.body.error.includes('test-secret'));
  } finally { globalThis.fetch=original; }
});

test('MiniMax probe reserves reasoning budget consistently with review calls', async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async (url, options) => {
      const body = JSON.parse(options.body);
      assert.equal(body.max_tokens,8192);
      assert.deepEqual(body.reasoning,{effort:'low'});
      return {ok:true,json:async()=>({choices:[{message:{content:'{"findings":[]}'}}]})};
    };
    const req = request(), res = response(); req.body.model='minimax/minimax-m3.1-flash:free';
    await handler(req,res); assert.equal(res.code,200);
  } finally { globalThis.fetch=original; }
});
