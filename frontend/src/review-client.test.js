import test from 'node:test';
import assert from 'node:assert/strict';
import { readProviders, reviewPayload, initialEndpoint, groupFindings, formatCost, upsertProvider } from './review-client.js';

test('saved model supplies its own credentials even when provider draft differs', () => {
  const entries = [{name:'openrouter/test',provider:'openai',model:' model-a ',apiKey:' saved-key ',baseUrl:' https://openrouter.ai/api/v1 '}];
  const payload = reviewPayload(' owner/repo ', 42, 'saved:openrouter/test', entries, 'fast');
  assert.deepEqual(payload, {repo:'owner/repo',pr_number:42,endpoint:'auto',review_mode:'fast',provider:'openai',model:'model-a',api_key:'saved-key',base_url:'https://openrouter.ai/api/v1'});
});

test('built-in selection does not accidentally send custom credentials', () => {
  assert.deepEqual(reviewPayload('owner/repo', 1, 'groq', [], 'thorough'), {repo:'owner/repo',pr_number:1,endpoint:'Grok',review_mode:'thorough'});
  assert.equal(reviewPayload('owner/repo', 1, 'gemini', [], 'thorough').endpoint, 'Gemini');
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

test('finding groups preserve original feedback indices while separating suggestions', () => {
  const input = [{category:'test_coverage'}, {kind:'security_concern'}, {kind:'bug'}];
  const groups = groupFindings(input);
  assert.equal(groups.test_suggestion[0].index,0);
  assert.equal(groups.security_concern[0].index,1);
  assert.equal(groups.bug[0].index,2);
});

test('default provider is selected only while its saved connection exists', () => {
  const storage = {getItem:key => key === 'ocr-default-provider' ? 'saved-one' : JSON.stringify([{name:'saved-one',provider:'openai',model:'m',apiKey:'k'}])};
  assert.equal(initialEndpoint(storage),'saved:saved-one');
  assert.equal(initialEndpoint({getItem:key => key === 'ocr-default-provider' ? 'removed' : '[]'}),'auto');
});

test('editing a renamed provider replaces the old entry instead of duplicating it', () => {
  assert.deepEqual(upsertProvider([{name:'old'},{name:'other'}],{name:'new'},'old'),[{name:'other'},{name:'new'}]);
});

test('cost display distinguishes estimates and does not duplicate the Cost label', () => {
  assert.equal(formatCost('Cost: Estimated $0.05'),'Estimated $0.05');
  assert.equal(formatCost({source:'estimated',amount:0.05}),'Estimated, not a confirmed charge: $0.0500');
  assert.equal(formatCost({source:'provider',total_cost:0}),'Provider-reported: $0.0000');
});
