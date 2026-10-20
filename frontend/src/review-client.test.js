import test from 'node:test';
import assert from 'node:assert/strict';
import { readProviders, reviewPayload } from './review-client.js';

test('saved model supplies its own credentials even when provider draft differs', () => {
  const entries = [{name:'openrouter/test',provider:'openai',model:' model-a ',apiKey:' saved-key ',baseUrl:' https://openrouter.ai/api/v1 '}];
  const payload = reviewPayload(' owner/repo ', 42, 'saved:openrouter/test', entries, 'fast');
  assert.deepEqual(payload, {repo:'owner/repo',pr_number:42,endpoint:'auto',review_mode:'fast',provider:'openai',model:'model-a',api_key:'saved-key',base_url:'https://openrouter.ai/api/v1'});
});

test('built-in selection does not accidentally send custom credentials', () => {
  assert.deepEqual(reviewPayload('owner/repo', 1, 'groq', [], 'thorough'), {repo:'owner/repo',pr_number:1,endpoint:'groq',review_mode:'thorough'});
});

test('missing saved selection fails rather than falling back to operator keys', () => {
  assert.throws(() => reviewPayload('owner/repo', 1, 'saved:deleted', [], 'fast'), /Select a saved provider/);
});

test('bad browser storage cannot crash the workspace', () => {
  for (const value of ['null', '{}', 'bad json', '[null,{},42]']) assert.deepEqual(readProviders({getItem:()=>value}), []);
});

test('incremental baseline is explicit and requires a full commit SHA', () => {
  assert.throws(() => reviewPayload('owner/repo',1,'groq',[],'fast','abc'), /40-character/);
  assert.equal(reviewPayload('owner/repo',1,'groq',[],'fast','A'.repeat(40)).baseline_sha,'a'.repeat(40));
});
