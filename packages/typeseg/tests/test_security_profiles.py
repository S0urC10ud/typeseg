"""Shipping-profile guards: actual weights, caches, transport, and span offsets."""
import numpy as np
import pytest
import typeseg
from typeseg import _runtime as rt
from typeseg import _onnx_backend as ob
from typeseg._profiles import manifest, model_config, verify_asset
from typeseg._security import decode, merge_short_runs, visual_whitespace
from typeseg._tokenize import text_to_bytes


def test_profile_switch_uses_distinct_weights_and_returns_to_general():
    text = '<script language="VBScript">s = Chr(65) & Chr(66)</script>'
    before = typeseg.fast(text)
    security = typeseg.fast(text, model_set='security')
    after = typeseg.fast(text, model_set='general')
    np.testing.assert_array_equal(before.char_probs, after.char_probs)
    assert not np.array_equal(before.char_probs, security.char_probs)
    assert model_config('unet')['file'] != model_config('unet','security')['file']
    assert rt._weights(model_config('unet')['file']) is not rt._weights(model_config('unet','security')['file'])


@pytest.mark.parametrize('fn', [typeseg.fast,typeseg.precise])
def test_invalid_profile_is_rejected_even_for_empty_text(fn):
    with pytest.raises(ValueError,match='model_set'): fn('',model_set='unknown')


def test_absent_architecture_cannot_fall_back_to_general(monkeypatch):
    import typeseg._profiles as profiles
    original = manifest()
    modified = {**original,'model_sets':{'security':{**original['model_sets']['security'],'mamba':None}}}
    monkeypatch.setattr(profiles,'manifest',lambda: modified)
    with pytest.raises(ValueError,match='No mamba'): typeseg.precise('',model_set='security')


def test_security_asset_checksum_rejects_corruption():
    name = model_config('unet','security')['file']
    with pytest.raises(ValueError,match='checksum mismatch'): verify_asset(name,b'corrupt model')


@pytest.mark.parametrize('arch',['unet','mamba'])
def test_security_onnx_matches_numpy_logits(arch):
    cfg = model_config(arch,'security');tokens=text_to_bytes('s = Chr(65) & Chr(66)\n')
    if arch=='unet':
        row=np.full(1536,256,np.int64);row[:len(tokens)]=tokens
        ref=rt.unet_forward(rt._weights(cfg['file']),row,channels=tuple(cfg['channels']))
        got=ob.unet_window_logits(row[None],name=cfg['onnx'])[0]
    else:
        ref=rt.mamba_forward(rt._weights(cfg['file']),tokens,n_layers=6,d_state=16,dt_rank=16,d_conv=4)
        got=ob.mamba_logits(tokens,name=cfg['onnx'])
    np.testing.assert_allclose(got,ref,atol=0.0003,rtol=0.0003)
    np.testing.assert_array_equal(got.argmax(-1),ref.argmax(-1))


@pytest.mark.parametrize('fn',[typeseg.fast,typeseg.precise])
def test_security_unicode_offsets_cover_original_python_characters(fn):
    text='print("café ☕ 😀")\nSELECT 1;'
    result=fn(text,model_set='security')
    assert result.char_probs.shape==(len(text),35)
    np.testing.assert_allclose(result.char_probs.sum(1),1.,atol=1e-5)
    assert ''.join(s.text for s in result.segments)==text
    assert result.segments[0].start==0 and result.segments[-1].end==len(text)
    for a,b in zip(result.segments,result.segments[1:]): assert a.end==b.start
    for segment in result.segments: assert segment.text==text[segment.start:segment.end]


def test_security_decoding_keeps_raw_probabilities_and_terminal_runs():
    text='aaaaZbb'
    p=np.zeros((len(text),35),np.float32);p[:,8]=1.;p[4,:]=0.;p[4,17]=1.;p[5:,:]=0.;p[5:,16]=1.
    original=p.copy();out=decode(text,p,typeseg.Options.security(),35)
    np.testing.assert_array_equal(p,original)
    np.testing.assert_array_equal(out,[8,8,8,8,8,16,16])
    # The inherited research policy intentionally preserves end runs.
    np.testing.assert_array_equal(merge_short_runs(np.array([8,8,16,16]),3),[8,8,16,16])


def test_visual_whitespace_prefers_same_line_and_left_ties():
    text='A B\n \nC'
    result=visual_whitespace(text,np.array([1,9,2,9,9,9,3]))
    np.testing.assert_array_equal(result,[1,1,2,2,2,3,3])


def test_visual_whitespace_matches_nearest_visible_character_policy():
    rng = np.random.default_rng(615)
    alphabet = list("ab \t\r\n\f\v") + ["é", "😀", "\ud800"]
    for length in range(150):
        text = ''.join(rng.choice(alphabet, length))
        labels = rng.integers(0, 35, length)
        expected = labels.copy()
        visible = [i for i, ch in enumerate(text) if ch not in " \t\r\n"]
        for i, ch in enumerate(text):
            if ch not in " \t\r\n" or not visible:
                continue
            start = text.rfind('\n', 0, i) + 1
            end = text.find('\n', i)
            if end < 0:
                end = length
            candidates = [j for j in visible if start <= j < end] or visible
            expected[i] = labels[min(candidates, key=lambda j: (abs(i - j), j > i))]
        original = labels.copy()
        np.testing.assert_array_equal(visual_whitespace(text, labels), expected)
        np.testing.assert_array_equal(labels, original)


def test_fast_run_merging_matches_original_greedy_policy():
    def slow(labels,minimum):
        out=labels.copy()
        if minimum<=1 or len(out)<3:return out
        while True:
            starts=np.r_[0,np.flatnonzero(out[1:]!=out[:-1])+1]
            ends=np.r_[starts[1:],len(out)]
            for i in range(1,len(starts)-1):
                if ends[i]-starts[i]>=minimum:continue
                left,right=out[starts[i-1]],out[starts[i+1]]
                out[starts[i]:ends[i]]=left if left==right or ends[i-1]-starts[i-1]>=ends[i+1]-starts[i+1] else right
                break
            else:return out
    rng=np.random.default_rng(718)
    for _ in range(2500):
        labels=rng.integers(0,int(rng.integers(1,9)),int(rng.integers(0,140)))
        minimum=int(rng.integers(1,12))
        np.testing.assert_array_equal(merge_short_runs(labels,minimum),slow(labels,minimum))


def test_fast_run_merging_handles_fragmented_large_sources():
    import time
    labels=np.tile([0,1],100000)
    start=time.monotonic();result=merge_short_runs(labels,3)
    assert time.monotonic()-start<3.
    assert len(result)==len(labels) and (result[:-1]==0).all() and result[-1]==1
