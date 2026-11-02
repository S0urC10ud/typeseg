import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createHash, webcrypto } from 'node:crypto';

const source = new URL('../../viewers/content_type_segmentor_static/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('assets/model_manifest.json',source),'utf8'));

test('security model binaries and tensor metadata match their manifest checksums',()=>{
  for(const architecture of ['unet','mamba']){
    const config=manifest.model_sets.security[architecture];
    for(const [field,name] of [['bin',config.weights],['metadata',config.metadata]]){
      const bytes=readFileSync(new URL('assets/'+name,source));
      assert.equal(createHash('sha256').update(bytes).digest('hex'),config.sha256[field]);
    }
    const metadata=JSON.parse(readFileSync(new URL('assets/'+config.metadata,source),'utf8'));
    const length=readFileSync(new URL('assets/'+config.weights,source)).length/4;
    for(const tensor of Object.values(metadata))assert.ok(tensor.offset>=0 && tensor.offset+tensor.length<=length);
  }
});

test('actual worker switches General/Security/General, verifies assets, and enforces ASCII offsets',async()=>{
  const messages=[];let listener;
  const ctx={Float32Array,Int32Array,Uint8Array,Math,Map,Promise,TextEncoder,performance,crypto:webcrypto,navigator:{},console,
    self:{location:{search:'?test=1'},addEventListener(_type,fn){listener=fn;},postMessage(message){messages.push(message);}},
    fetch:async url=>{const clean=url.split('?')[0];const blob=readFileSync(new URL(clean,source));return {ok:true,text:async()=>blob.toString(),arrayBuffer:async()=>blob.buffer.slice(blob.byteOffset,blob.byteOffset+blob.byteLength)};}
  };
  vm.createContext(ctx);
  ctx.importScripts=url=>vm.runInContext(readFileSync(new URL(url.split('?')[0],source),'utf8'),ctx);
  vm.runInContext(readFileSync(new URL('segmentor-worker.js',source),'utf8'),ctx);
  const request=async(type,payload)=>{await listener({data:{type,requestId:messages.length+1,payload}});return messages.at(-1);};
  assert.equal((await request('load',{model:'unet',modelSet:'general'})).type,'result');
  const original=(await request('segment',{text:'SELECT name FROM samples;'})).payload;
  assert.equal((await request('switch_model',{model:'unet',modelSet:'security'})).payload.model_set,'security');
  const secured=(await request('segment',{text:'SELECT name FROM samples;'})).payload;
  assert.equal(secured.model_checkpoint_sha256,manifest.model_sets.security.unet.checkpoint_sha256);
  assert.equal(secured.window_count,1);
  assert.equal((await request('segment',{text:'cafÃ© ðŸ˜€'})).type,'error');
  assert.equal((await request('segment',{text:'A'.repeat(manifest.max_input_bytes+1)})).type,'error');
  assert.equal((await request('switch_model',{model:'unet',modelSet:'general'})).type,'result');
  const restored=(await request('segment',{text:'SELECT name FROM samples;'})).payload;
  assert.deepEqual(restored.segments,original.segments);
  assert.deepEqual(restored.char_top_probs,original.char_top_probs);
});
