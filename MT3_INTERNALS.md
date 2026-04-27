# MT3 Internals Reference

Deep-dive into the MT3 reference implementation. Each section links to the corresponding file in **this project** where the concept is implemented.

---

## 1. MIDI Tokenizer / Vocabulary

**MT3 reference:** `mt3/vocabularies.py`, `mt3/event_codec.py`, `mt3/run_length_encoding.py`, `mt3/note_sequences.py`
**This project:** [`src/data/midi_tokenizer.py`](src/data/midi_tokenizer.py)

---

### Special Token IDs

`GenericTokenVocabulary` (`mt3/vocabularies.py:148`) reserves three IDs at the front:

```
PAD = 0
EOS = 1
UNK = 2
```

Event tokens start at ID 3 (shifted by 3 relative to codec indices). There are no symbolic `NoteOn`/`NoteOff` constants — note direction is encoded by velocity value.

```python
# mt3/vocabularies.py:148
class GenericTokenVocabulary(seqio.Vocabulary):
  def __init__(self, regular_ids: int, extra_ids: int = 0):
    self._num_special_tokens = 3    # PAD=0, EOS=1, UNK=2
    self._num_regular_tokens = regular_ids
```

---

### Token Layout

`build_codec` (`mt3/vocabularies.py:119`) defines five event ranges stacked contiguously after the shift range:

| Range | Type | Values | Count |
|---|---|---|---|
| `[0..1000]` | **shift** | 0..1000 steps (100 steps/sec, max 10s) | 1001 |
| `[1001..1128]` | **pitch** | MIDI 0–127 | 128 |
| `[1129..1256]` | **velocity** | 0 = note-off, 1–127 = velocity bin | 128 |
| `[1257]` | **tie** | single sentinel value | 1 |
| `[1258..1385]` | **program** | MIDI program 0–127 | 128 |
| `[1386..1513]` | **drum** | MIDI pitch 0–127 | 128 |

```python
# mt3/vocabularies.py:119
def build_codec(vocab_config: VocabularyConfig):
  event_ranges = [
      EventRange('pitch', 0, 127),
      EventRange('velocity', 0, vocab_config.num_velocity_bins),  # 0..127
      EventRange('tie', 0, 0),
      EventRange('program', 0, 127),
      EventRange('drum', 0, 127),
  ]
  return Codec(
      max_shift_steps=100 * 10,   # 1000
      steps_per_second=100,
      event_ranges=event_ranges)
```

Codec encode/decode (`mt3/event_codec.py:79`):

```python
def encode_event(self, event: Event) -> int:
  offset = 0
  for er in self._event_ranges:
    if event.type == er.type:
      return offset + event.value - er.min_value
    offset += er.max_value - er.min_value + 1

def decode_event_index(self, index: int) -> Event:
  offset = 0
  for er in self._event_ranges:
    if offset <= index <= offset + er.max_value - er.min_value:
      return Event(type=er.type, value=er.min_value + index - offset)
    offset += er.max_value - er.min_value + 1
```

---

### Encode Pipeline: NoteSequence → Tokens

**Step 1** — Extract sorted `(time, NoteEventData)` pairs (`mt3/note_sequences.py:177`):

```python
def note_sequence_to_onsets_and_offsets_and_programs(ns):
  notes = sorted(ns.notes, key=lambda note: (note.is_drum, note.program, note.pitch))
  times = (
      [note.end_time for note in notes if not note.is_drum] +   # offsets first
      [note.start_time for note in notes]                        # then onsets
  )
  values = (
      [NoteEventData(pitch=note.pitch, velocity=0, program=note.program, is_drum=False)
       for note in notes if not note.is_drum] +    # velocity=0 → note-off
      [NoteEventData(pitch=note.pitch, velocity=note.velocity,
                     program=note.program, is_drum=note.is_drum)
       for note in notes]                           # velocity>0 → note-on
  )
```

**Step 2** — Convert each `NoteEventData` to `Event` objects (`mt3/note_sequences.py:215`):

```python
def note_event_data_to_events(state, value: NoteEventData, codec):
  num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
  velocity_bin = vocabularies.velocity_to_bin(value.velocity, num_velocity_bins)
  if value.is_drum:
    return [Event('velocity', velocity_bin), Event('drum', value.pitch)]
  else:
    # full multi-instrument: program + velocity + pitch
    return [Event('program', value.program),
            Event('velocity', velocity_bin),
            Event('pitch', value.pitch)]
```

Velocity binning (`mt3/vocabularies.py:63`):

```python
def velocity_to_bin(velocity, num_velocity_bins):
  if velocity == 0:
    return 0   # note-off
  else:
    return math.ceil(num_velocity_bins * velocity / MAX_MIDI_VELOCITY)
    # velocity=127, bins=127 → 127
    # velocity=64,  bins=127 → 64
```

**Step 3** — Interleave with 1-step shift events, then RLE (`mt3/run_length_encoding.py:63`):

Raw encoding emits individual `shift(1)` for each elapsed step, then RLE collapses them:

```python
# mt3/run_length_encoding.py:242
# Before RLE: [shift1, shift1, ...(×100)..., shift1, pitch_C4]
# After RLE:  [shift100, pitch_C4]

def run_length_encode_shifts(features):
  shift_steps = 0
  output = []
  for event in events:
    if codec.is_shift_event_index(event):
      shift_steps += 1
    else:
      while shift_steps > 0:
        output_steps = min(codec.max_shift_steps, shift_steps)  # max 1000 per token
        output.append(output_steps)
        shift_steps -= output_steps
      output.append(event)
  # trailing shifts are dropped
```

---

### Decode Pipeline: Tokens → NoteSequence

`decode_events` (`mt3/run_length_encoding.py:371`) walks the token stream:

```python
def decode_events(state, tokens, start_time, max_time, codec, decode_event_fn):
  cur_steps = 0
  cur_time = start_time
  for token in tokens:
    event = codec.decode_event_index(token)
    if event.type == 'shift':
      cur_steps += event.value
      cur_time = start_time + cur_steps / codec.steps_per_second
    else:
      cur_steps = 0    # non-shift resets step accumulator
      decode_event_fn(state, cur_time, event, codec)
```

`decode_note_event` state machine (`mt3/note_sequences.py:313`):

```python
def decode_note_event(state: NoteDecodingState, time, event, codec):
  if event.type == 'velocity':
    state.current_velocity = bin_to_velocity(event.value, num_velocity_bins)
  elif event.type == 'program':
    state.current_program = event.value
  elif event.type == 'pitch':
    if state.current_velocity == 0:   # note-off
      onset_time, onset_velocity = state.active_pitches.pop((pitch, program))
      _add_note_to_sequence(state.note_sequence, onset_time, time, pitch, ...)
    else:                             # note-on
      state.active_pitches[(pitch, program)] = (time, state.current_velocity)
  elif event.type == 'drum':
    _add_note_to_sequence(..., is_drum=True)
  elif event.type == 'tie':
    # close active notes not carried over as ties
    for (pitch, program) not in state.tied_pitches:
      _add_note_to_sequence(...)
    state.is_tie_section = False
```

**Three encoding specs** (`mt3/note_sequences.py:415`):

| Spec | Offset matching | Tie sections |
|---|---|---|
| `NoteOnsetEncodingSpec` | No | No |
| `NoteEncodingSpec` | Yes | No |
| `NoteEncodingWithTiesSpec` | Yes | Yes |

---

## 2. Encoder-Decoder Model

**MT3 reference:** `mt3/network.py`, `mt3/layers.py`
**This project:**
- [`src/model/mt3_model.py`](src/model/mt3_model.py) — top-level `MT3Model`
- [`src/model/encoder.py`](src/model/encoder.py) — `MT3Encoder`
- [`src/model/decoder.py`](src/model/decoder.py) — `MT3Decoder`
- [`src/model/positional_encoding.py`](src/model/positional_encoding.py) — `T5RelativePositionBias`

---

### T5Config

```python
# mt3/network.py:25
@struct.dataclass
class T5Config:
  vocab_size: int
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  head_dim: int = 64
  mlp_dim: int = 2048
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0.1
  logits_via_embedding: bool = False
```

---

### Forward Pass (`Transformer.__call__`, `mt3/network.py:363`)

```python
def __call__(self, encoder_input_tokens, decoder_input_tokens, decoder_target_tokens, ...):
  encoded = self.encode(encoder_input_tokens, ...)
  return self.decode(encoded, encoder_input_tokens, decoder_input_tokens, ...)
```

**Encoder** (`mt3/network.py:162`):
1. Project continuous spectrogram frames → `emb_dim` via `DenseGeneral`
2. Add **sinusoidal fixed PE** (not learned; no bucket calculation)
3. N × `EncoderLayer`: pre-norm → self-attention → residual → pre-norm → MLP → residual
4. Final `LayerNorm`

> **Difference from this project:** MT3 uses sinusoidal PE; `src/model/positional_encoding.py` implements learned **T5 relative position bias** with bucket calculation instead.

**Decoder** (`mt3/network.py:92`):
1. Embed decoder tokens → `emb_dim`
2. N × `DecoderLayer`:
   - Pre-norm → **causal self-attention** → residual
   - Pre-norm → **cross-attention** over encoder output → residual
   - Pre-norm → MLP → residual
3. Final `LayerNorm` → logits projection

---

### Attention (`mt3/layers.py:85`)

```python
def dot_product_attention(query, key, value, bias=None, ...):
  # [batch, num_heads, q_len, kv_len]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)
  if bias is not None:
    attn_weights = attn_weights + bias
  attn_weights = jax.nn.softmax(attn_weights)
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)
```

**Scaling** — NOT done with explicit `/ sqrt(head_dim)`. Instead, folded into query init (`mt3/layers.py:186`):

```python
depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
query_init = lambda *args: self.kernel_init(*args) / depth_scaling
```

**Autoregressive KV cache** — one-hot scatter trick (`mt3/layers.py:246`):

```python
# Cache stored as [batch, heads, head_dim, length] for TPU fusion
one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
key = cached_key.value + one_token_key * one_hot_indices   # scatter via add, no copy
```

---

### T5 LayerNorm (`mt3/layers.py:604`)

RMS-only normalization — no mean subtraction, no additive bias:

```python
class LayerNorm(nn.Module):
  def __call__(self, x):
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = param_with_axes('scale', self.scale_init, (features,))
    return y * scale     # no bias, no mean subtraction
```

---

### Relative Position Bias (MT3 reference: absent; this project: implemented)

MT3 uses sinusoidal fixed embeddings — no bucket calculation. This project's `T5RelativePositionBias` (`src/model/positional_encoding.py`) implements the T5 paper approach:

- Bidirectional for encoder (both directions bucketed)
- Causal (unidirectional) for decoder
- Bucket index = f(relative position, num_buckets, max_distance)
- Output added as bias to attention logits before softmax

---

## 3. Evaluation / Metrics

**MT3 reference:** `mt3/metrics_utils.py`, `mt3/metrics.py`
**This project:**
- [`src/evaluation/evaluate.py`](src/evaluation/evaluate.py) — `evaluate_test_set()`
- [`src/evaluation/decode.py`](src/evaluation/decode.py) — `greedy_decode()`, `beam_search_decode()`
- [`src/evaluation/metrics.py`](src/evaluation/metrics.py) — `compute_note_metrics()`, `compute_multi_instrument_metrics()`

---

### Tokens → NoteSequence

`event_predictions_to_ns` (`mt3/metrics_utils.py:119`) decodes per segment, sorted by `start_time`, with a `max_decode_time` fence preventing bleed into the next segment:

```python
def event_predictions_to_ns(predictions, codec, encoding_spec):
  ns, _, _ = decode_and_combine_predictions(
      predictions=predictions,
      decode_tokens_fn=functools.partial(
          run_length_encoding.decode_events,
          codec=codec,
          decode_event_fn=encoding_spec.decode_event_fn))
  return {'est_ns': ns, ...}
```

---

### Frame Metrics

NoteSequence → pretty_midi piano roll at **62.5 fps** (`mt3/metrics_utils.py:149`):

```python
def get_prettymidi_pianoroll(ns, fps, is_drum):
  for note in ns.notes:
    note.end_time = max(note.end_time, note.start_time + 0.05)  # min 50ms
  pm = note_seq.note_sequence_to_pretty_midi(ns)
  return pm.get_piano_roll(fs=fps)   # fps=62.5

def frame_metrics(ref_pianoroll, est_pianoroll, velocity_threshold=30):
  ref_bool = ref_pianoroll > velocity_threshold
  est_bool = est_pianoroll > 0
  precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
      ref_bool.flatten(), est_bool.flatten(), labels=[True, False])
```

---

### Note-Level Metrics (mir_eval)

Four variants computed per track (`mt3/metrics.py:255`):

| Variant | `offset_ratio` | Velocity |
|---|---|---|
| Onset-only | `None` | No |
| Onset + offset | `0.2` (default) | No |
| Onset + velocity | `None` | Yes |
| Onset + offset + velocity | `0.2` | Yes |

**Default tolerances** (mir_eval defaults, not overridden):
- Onset tolerance: **50 ms**
- Offset tolerance: **max(50 ms, 20% of note duration)**

**Tolerance sweep** (`mt3/metrics.py:149`) — P/R/F1 at six windows:

```python
tolerances = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)  # seconds: 10ms..500ms
for tol in tolerances:
  precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
      ..., onset_tolerance=tol, offset_min_tolerance=tol)
```

---

### Program-Aware Averaging (`mt3/metrics.py:36`)

Per `(program, is_drum)` pair → micro-averaged by note count:

```python
for program, is_drum in program_and_is_drum_tuples:
  est_track = extract_track(est_ns, program, is_drum)
  ref_track = extract_track(ref_ns, program, is_drum)
  if is_drum:
    args['offset_ratio'] = None   # drums: onset-only matching
  precision, recall, _, _ = mir_eval.transcription.precision_recall_f1_overlap(**args)
  precision_sum += precision * len(est_intervals)
  precision_count += len(est_intervals)

final_precision = precision_sum / precision_count
f_measure = mir_eval.util.f_measure(final_precision, final_recall)
```

Three program granularities (defined in `mt3/vocabularies.py:100`):

| Granularity | Mapping |
|---|---|
| `flat` | All programs → 0 |
| `midi_class` | Program → first in its 8-program MIDI class |
| `full` | Program as-is |
