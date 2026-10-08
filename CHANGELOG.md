# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- WAV: audio coded in blocks of several samples (ADPCM, DV audio, MP3,
  WMA, …) is packetized and timed as FFmpeg's wavdec and demuxer layer do:
  about 100 ms of whole blocks a packet (`ff_pcm_default_packet_size`),
  each packet's pts and duration in samples where the format fixes them
  (`get_audio_frame_duration`: IMA and MS ADPCM, Yamaha/OKI, G.726, WMA),
  else only the first packet after open or a seek timed; seeks by the byte
  rate (`ff_pcm_read_seek`); duration from the `fact` count or the byte
  rate; bit rate from `nAvgBytesPerSec`. Each block was timed as one
  sample, so packets after the first 1024 blocks were stamped far too
  early.

### Added

- `rawvideo` decoder implements `Decoder::output_video_dimensions` /
  `output_pixel_format`: the stream's `width` × `height` and pixel format,
  which every frame has.

## [0.0.10](https://github.com/OxideAV/oxideav-basic/compare/v0.0.9...v0.0.10) - 2026-08-18

### Other

- union overrides apply to the automatic EXTENSIBLE promotion
- Samples-union member selection — wSamplesPerBlock for block-compressed subformats
- EXTENSIBLE black-box interop fixtures + doc surface refresh
- automatic WAVE_FORMAT_EXTENSIBLE promotion in the muxer
- KSDATAFORMAT subtype naming, IEC 61937 family, resolver tag routing
- surface dwChannelMask through core ChannelLayout + mismatch flag
- EXTENSIBLE routing by container size + Samples-union validation
- describe Y4M little-endian + colorspace-token conventions on their own terms
- document WAV demuxer hostile-input robustness + bench
- add dependency-free read/write hot-path micro-benchmark
- per-chunk-id targeted fuzz reaching every parser directly
- cap sxml sub-chunk / alignment-point reservations to body size
- deterministic mutation-fuzz net over the demuxer parse path
- guard chunk-skip seeks against i64-overflow backward wrap
- cap ds64 table reservation to available records (OOM hardening)
- bound chunk-body allocation against size-lies (DoS hardening)
- add CI / crates.io / docs.rs / MIT-license badges
- smpl + inst chunks read+write symmetric — typed structs + writers
- document DISP / id3 / PAD chunks + LIST INFO write side
- typed WavDemuxer::disp() / info() accessors
- LIST INFO write side — InfoChunk/InfoEntry + with_info
- id3 chunk write side — WavMuxOptions::with_id3
- parse PAD (alignment-padding) chunk
- DISP (Display) chunk write side — typed DispChunk + with_disp
- parse DISP (Display) + id3 (embedded ID3v2 tag) chunks
- sxml (serialized-XML ADM carrier) read+write — README + CHANGELOG
- symmetric sxml (serialized-XML) muxer write support per BS.2088 §7
- read BW64/ADM sxml (serialized-XML) chunk per ITU-R BS.2088-2 §7
- end-to-end demux->re-emit->demux symmetry test for cue/plst/adtl
- cue/plst/adtl symmetric read+write + trailing-chunk scan (CHANGELOG + README)
- symmetric cue / plst / LIST adtl muxer write support
- typed cue / plst / LIST adtl read + post-data trailing-chunk scan
- test RF64 Reserve overflow→promotion via streamed >4 GiB sparse sink
- RF64/BW64 64-bit large-file write side (ds64 + JUNK on-the-fly conversion)
- integration round-trip test for combined acid + chna metadata
- classify chna audioID refs by ADM prefix + BS.2088 §3 definition scope
- BWF bext chunk read+write with typed BextChunk surface
- parse BW64/ADM bxml (compressed XML) chunk per ITU-R BS.2088-2 §6
- BW64/ADM chna (channel-allocation) chunk read+write
- recognise extended LIST INFO sub-IDs (ExifTool RIFF Info Tags)

### Fixed

- WAV `WAVE_FORMAT_EXTENSIBLE` codec routing now dispatches on the `wBitsPerSample` **container size**, not the union's `wValidBitsPerSample` — per `docs/container/riff/waveformatextensible/ms-waveformatextensible.html` §`Samples.wValidBitsPerSample` ("wBitsPerSample is the container size") and `ms-extensible-wave-format.html` ("a 20-bit sample can be stored left-justified within a three-byte container"), the `data` chunk carries container-sized samples and the union WORD only states how many bits carry signal. A 24-valid-bits-in-32-bit-container stream previously mis-resolved to `pcm_s24le` (wrong interleave stride); it now resolves to `pcm_s32le` with the precision surfaced separately via `wav:fmt.valid_bits_per_sample` / `WavDemuxer::valid_bits_per_sample`

### Other

- WAV mux `Samples`-union control decoupled from the explicit mask: `WavMuxOptions::with_valid_bits_per_sample` / `with_subformat` now apply whenever the EXTENSIBLE form is written — the automatic promotion included — instead of silently no-oping without `with_extensible` (a 20-in-24 stream needs no explicit mask any more). Mux/demux symmetry guard: a precision-family `wValidBitsPerSample` override above the container fails `write_header` rather than producing a file the demuxer rejects
- black-box `WAVE_FORMAT_EXTENSIBLE` interop suite (`tests/wav_extensible_interop.rs`): tool-generated multi-channel (6ch s16), 24-bit and 8-channel float fixtures demux with correct GUID routing / typed layouts, and the muxer's automatically-promoted EXTENSIBLE output decodes byte-exactly through an external validator invoked as a black box. Skips silently when the tool is absent
- WAV muxer now follows the staged encoder guidance automatically (`docs/container/riff/waveformatextensible/README.md` §"What's covered"): more than two channels or a container size above 16 bits promote the `fmt ` chunk to `WAVE_FORMAT_EXTENSIBLE` (40-byte form), with `dwChannelMask` derived from the stream's typed `CodecParameters::channel_layout` when declared, else from the channel-count default (`ChannelLayout::from_count`; a `DiscreteN` layout writes mask `0`, the legal "no assigned speaker positions" bitmap). Classic ≤ 2-channel / ≤ 16-bit output (incl. the 8-bit G.711 tags) stays byte-identical to the historical muxer. `WavMuxOptions::with_extensible` still forces the form with an explicit mask (and wins over everything); new `WavMuxOptions::with_legacy_waveformatex` suppresses the automatic promotion for byte-conservative re-muxing of legacy-tagged input. Round-trip symmetric: the mask the muxer derives maps back to the same typed layout on demux
- WAV codec-tag resolution now routes unmapped `wFormatTag` values through the `CodecResolver` threaded into `open()` — the registry owns the RFC 2361 tag value space (`docs/container/riff/rfc2361-wav.txt`), so ADPCM-family / MP3 / any registry-claimed tag resolves to its codec crate's id instead of hard-erroring (legacy path) or falling to a synthesised GUID id (EXTENSIBLE template-GUID path, per the documented tag↔GUID equivalence). Probes see the `CodecTag::WaveFormat` tag, `bits`/`channels`/`sample_rate` hints and the raw `fmt ` chunk as the header blob. Native PCM/IEEE-float/A-law/μ-law mappings never consult the resolver. New public `wav::open_wav_demuxer_with(input, &dyn CodecResolver)`; `open_wav_demuxer` keeps the resolver-free behaviour
- WAV EXTENSIBLE SubFormat GUIDs now surface their symbolic `KSDATAFORMAT_SUBTYPE_*` name (`wav:fmt.subformat_name`) for every row of the staged consolidated catalog (`docs/container/riff/waveformatextensible/ksdataformat-subtype-guids.md`) — both the `WAVEFORMATEX`-derived family and the IEC 61937 compressed-pass-through family (`Data2 == 0x0cea`, incl. the Dolby MAT 2.0/2.1 revisions from `ms-representing-formats-iec61937.md`). IEC 61937 GUIDs additionally surface `wav:fmt.iec61937_stream_type` (CEA-861 stream type, `Data1` low byte) and are guarded from the wave-format-tag route per the catalog's dispatch guidance — their `Data1` is a stream type, not a tag
- WAV `dwChannelMask` now surfaces through oxideav-core's typed `ChannelLayout` on `CodecParameters::channel_layout` — position-SET matching against the named core layouts via the staged `SPEAKER_*` bit table (`ms-waveformatextensible.html`), with the two back-pair alias rows from the staged standard-layouts table (`0x33` → `Quad`, `0x3F` "5.1 (Microsoft)" → `Surround51`) and a `DiscreteN` fallback for assigned-but-unnamed position sets so a consumer never gets a named layout that contradicts the explicit mask. Mask `0` (no assigned positions) and a set-bit/`nChannels` disagreement leave the field `None`; the disagreement additionally surfaces as `wav:fmt.channel_mask_mismatch` (the staged doc's "should be the same" is a flag, not a reject). New public `wav::channel_mask_for_layout` derives the mask for a core layout (muxer-side building block)
- WAV demuxer validates the EXTENSIBLE `Samples` union per the staged struct doc: `wValidBitsPerSample > wBitsPerSample` (a precision claim exceeding the container) and a container size that is zero or not a multiple of 8 are both rejected as malformed; a zero union WORD (writer left it unset) stays tolerated with the container size as the effective precision
- WAV EXTENSIBLE `Samples` union member selection: the union WORD now surfaces under the member the SubFormat gives it — `wValidBitsPerSample` for the precision-semantics family (PCM / IEEE-float / A-law / μ-law templates + the IEC 61937 family, whose staged worked examples all set 16-in-16) or `wSamplesPerBlock` (new `wav:fmt.samples_per_block` key + `WavDemuxer::samples_per_block` accessor, `0` = variable) for block-compressed subformats (other template tags + vendor GUIDs). The precision-family validation no longer fires on block-compressed streams, whose 4-bit nibble containers and large per-block sample counts are legal (`ms-waveformatextensible.html` §`Samples`); `WavDemuxer::valid_bits_per_sample` correspondingly returns `None` for them — the union has exactly one active member

- `smpl` (Sampler) + `inst` (Instrument) chunks **read+write symmetric** (RIFF MCI §3) — promoted from read-only metadata-key surfaces to typed `wav::SmplChunk` / `wav::SampleLoop` / `wav::InstChunk` public types with byte-lossless `parse`/`to_bytes` (the `smpl` loop array + vendor `sampler_data` tail round-trip verbatim; `num_sample_loops` re-derived from the actual loop count on write). `WavMuxOptions::with_smpl` / `with_inst` write the chunks ahead of `data` (RIFF §2 word-aligned), and the typed views are reachable via `WavDemuxer::smpl()` / `inst()`. The existing `wav:smpl.*` / `wav:inst.*` read keys are unchanged (the parsers now delegate to the typed structs)
- typed `WavDemuxer::disp()` / `info()` accessors — the `DISP` and `LIST INFO` typed views are now captured during the chunk walk and reachable on the concrete demuxer (matching the typed-view pattern of every other chunk), `None` when absent. `info()` carries every sub-ID verbatim (incl. unknown/vendor) for re-emission via `with_info`
- `LIST INFO` (text metadata) chunk **write side** (`WavMuxOptions::with_info`) — gives read/write symmetry for WAV text tags (RIFF MCI §3 "INFO List Chunk"). New public `InfoChunk` / `InfoEntry` types carry an ordered list of `(sub-ID FOURCC, text)` entries with `parse`/`to_bytes` (NUL-terminated, RIFF §2 word-aligned sub-chunks); the writer wraps them in a `LIST`(`INFO`) chunk ahead of `data`, and the demuxer reads recognised sub-IDs back through their snake_case metadata keys. Unknown / vendor sub-IDs round-trip through `InfoChunk`. The existing read path now delegates to `InfoChunk::parse`
- `DISP` (Display) chunk **read+write symmetric** — clipboard-format `type` DWORD + payload; the WAV "SoundSchemeTitle" convention (ExifTool RIFF tag table). Typed `DispChunk` public type with byte-lossless `parse`/`to_bytes` plus `DispChunk::text`/`title` for the common `CF_TEXT` (type==1) display-title form. The demuxer surfaces `wav:disp.{body_len,type}` always plus `wav:disp.title` for the `CF_TEXT` form (text trimmed at the first NUL); non-text clipboard formats surface header-only, truncated bodies are skipped-as-opaque. `WavMuxOptions::with_disp` writes the chunk ahead of `data` (RIFF §2 word-aligned)
- parse `PAD ` (Pad / alignment-padding) chunk — the alignment sibling of `JUNK` in the RIFF MCI §2 "skip/ignore" dispatch group. Semantic-free body, so only accounting surfaces under the parallel `wav:pad.*` key namespace (`count` / `total_bytes` / per-chunk `<n>.body_len`), mirroring the `JUNK` contract; `JUNK` and `PAD ` stay in separate namespaces. Absence is observable (no `wav:pad.*` keys emitted)
- `id3 ` / `ID3 ` (embedded ID3v2 tag) chunk **read+write symmetric** — the read side surfaces the 10-byte ID3v2 header fields for observability per `docs/container/id3/id3v2.3.0.html` §3.1: `wav:id3.{body_len,version,flags,tag_size}` plus the three flag bits `.unsynchronisation`/`.extended_header`/`.experimental`, with the §3.1 synchsafe (28-bit) tag-size decode. The write side (`WavMuxOptions::with_id3`) carries a complete ID3v2 tag verbatim as an `id3 ` chunk ahead of `data` (RIFF §2 word-aligned). Frame encoding/decoding is left to `oxideav-id3` per the codec/container split; malformed/non-`ID3`-magic bodies surface only `body_len`
- BW64/ADM `<sxml>` (serialized-XML) chunk **read+write symmetric** per ITU-R BS.2088-2 §7 — the third ADM XML carrier alongside `<axml>` (uncompressed) and `<bxml>` (compressed-whole). Typed `SxmlChunk` / `SubXmlChunk` / `AlignmentPoint` public types with byte-lossless `parse`/`to_bytes` (the on-wire `subXMLCkTbSize` is preserved verbatim; `computed_sub_table_byte_size` yields the canonical value). Segments the (optionally gzip) XML into per-sample-run SubXMLChunk records (binding a run of audio samples to an XML fragment for time-variant / Serial ADM metadata) plus an optional AlignmentPoint table for timestamp-based access. Surfaces `wav:sxml.{body_len,fmt_type,compression,sub_chunk_count,alignment_point_count,total_samples}` + per-record `.<n>.{samples,xml_len,xml}` (XML text only for the uncompressed `fmtType==0x0000` form; gzip/private codes surface header fields only, leaving RFC 1952 inflation to a higher-level ADM-aware consumer). `WavMuxOptions::with_sxml` writes the chunk ahead of `chna`/`data` (§2.1 order) with RIFF §2 word-alignment; `WavDemuxer::sxml()` accessor. An `sxml` chunk (like `chna`) marks the file ADM-carrying, so a forced/promoted 64-bit file uses the `BW64` top-level magic. Malformed/truncated bodies are skipped-as-opaque
- WAV cue / playlist / associated-data symmetric read+write: typed `CuePoint`/`CueChunk`, `PlaylistSegment`/`PlaylistChunk`, and `AdtlEntry`/`AdtlChunk` (`labl`/`note`/`ltxt`) public types with byte-exact `parse`/`to_bytes` per Microsoft RIFF MCI §3 (Cue-Points / Playlist / Associated Data Chunk). `WavMuxOptions::with_cue` / `with_plst` / `with_adtl` writers emit the chunks in the trailer (after `data`, the conventional placement for sample-position-referencing chunks); `WavDemuxer::cue()` / `plst()` / `adtl()` accessors. Overclaimed counts are clamped; `adtl` sub-chunks get the RIFF word-alignment pad automatically
- WAV demuxer no longer treats `data` as the chunk-scan terminator — cue / plst / `LIST adtl` chunks written *after* the waveform (the common convention) are now read, by seeking over the word-aligned `data` body and walking remaining chunks to EOF
- WAV muxer RF64/BW64 64-bit large-file **write** side (read/write symmetry): `WavMuxOptions::with_rf64(Rf64Mode)` — `Force` always emits a `ds64` chunk + `0xFFFFFFFF` sentinels + `RF64`/`BW64` magic; `Reserve` writes a `ds64`-sized `JUNK` placeholder and promotes it on-the-fly to `ds64` only if the finished file overflows 32 bits (ITU-R BS.2088-2 §3.6/§4.2 + EBU Tech 3306 v2 §3); `Never` (default) is byte-identical to the historical 32-bit muxer and errors on overflow. BW64 magic selected automatically when an ADM `chna` chunk is present. New public `Rf64Mode` enum
- test: `Rf64Mode::Reserve` overflow→promotion path driven through a genuine >4 GiB streamed `data` payload via a sparse seekable sink (no gigabyte allocation) — verifies the in-place `JUNK`→`ds64` rewrite, `RF64` magic flip, and 64-bit `riffSize`/`dataSize`/`sampleCount` body values
- integration test: full public mux → demux round-trip of a WAV carrying BOTH an Acidizer `acid` loop chunk and a BW64/ADM `chna` chunk — PCM byte-for-byte, typed WavDemuxer::{acid,chna}() accessors, and the ADM kind/definition-scope metadata keys (tests/wav_adm_metadata_roundtrip.rs)
- BW64/ADM chna: classify each audioID reference (ATU_/AT_/AC_/AP_ → AdmRefKind) and resolve its definition scope per ITU-R BS.2088-2 §8.1/§3 (trailing four hex digits ≤ 0x0FFF = BS.2094 common, ≥ 0x1000 = file-local custom) — surfaced as wav:chna.<n>.{track_ref,pack_ref}_{kind,definition} keys, AdmRefKind/DefinitionScope public enums + AudioId accessors
- BWF bext (Broadcast Audio Extension) chunk read+write with typed BextChunk surface (EBU Tech 3285 v2 §2.3) — WavMuxOptions::with_bext writer, WavDemuxer::bext() accessor, byte-exact parse/to_bytes round-trip
- parse BW64/ADM bxml (compressed XML) chunk per ITU-R BS.2088-2 §6 — fmtType header + uncompressed-XML text surface
- BW64/ADM chna (channel-allocation) chunk read+write with typed ChnaChunk/AudioId surface (ITU-R BS.2088-2 §8.1)
- recognise extended LIST INFO sub-IDs (ExifTool RIFF Info Tags) — IAS1..IAS9 audio-language slots, Windows-Media "more info" set, production-credit tags

## [0.0.9](https://github.com/OxideAV/oxideav-basic/compare/v0.0.8...v0.0.9) - 2026-06-15

### Other

- resolve every DEFINE_WAVEFORMATEX_GUID SubFormat through the legacy wFormatTag path
- parse LIST('wavl') wave-list waveform container (RIFF MCI §3)
- decode WAVEFORMATEXTENSIBLE dwChannelMask into SPEAKER_* layout
- Acidizer acid chunk read+write with typed AcidChunk surface
- complete LIST adtl coverage — file sub-chunk + ltxt locale WORDs per RIFF MCI §3
- parse slnt (Silence) chunk per Microsoft RIFF MCI §3
- parse _PMX (Adobe XMP packet) chunk for AVI/WAV files
- parse <axml> chunk per EBU Tech 3285 Supplement 5
- drop release-plz.toml — use release-plz defaults across the workspace
- parse JUNK (Filler) chunk per Microsoft RIFF MCI §2
- typed scalar Reinhard 2002 simple global tone-mapping operator
- RF64 / BW64 64-bit-extended form + ds64 chunk (EBU Tech 3306)
- full RIFF MCI §3 INFO List Chunk baseline (12 added sub-IDs)
- parse CSET (Character Set) chunk per Microsoft RIFF MCI §3
- parse iXML chunk — third-party production-recorder metadata
- parse fact chunk (RIFF MCI §3) — authoritative per-channel sample count

### Added

- WAV `WAVE_FORMAT_EXTENSIBLE` SubFormat-GUID resolution now implements
  the `KSMedia.h` `DEFINE_WAVEFORMATEX_GUID(x)` /
  `IS_VALID_WAVEFORMATEX_GUID` / `EXTRACT_WAVEFORMATEX_ID` macros (per
  `docs/container/riff/waveformatextensible/ms-converting-format-tags-and-subformat-guids.md`):
  any SubFormat GUID of the form `{0000xxxx-0000-0010-8000-00AA00389B71}`
  is recognised as equivalent to the legacy `wFormatTag` `x` and
  dispatches through the same `codec_for_tag` route the `WAVEFORMATEX`
  path uses. Generalises the four previously hand-listed GUID constants
  (PCM/IEEE_FLOAT/ALAW/MULAW) to every tag-derived GUID and surfaces the
  embedded legacy tag as `wav:fmt.subformat_tag` (e.g. `0x0055` for an
  EXTENSIBLE MP3-tagged file). Non-template GUIDs and template GUIDs
  whose embedded tag isn't a directly-mapped format still synthesise the
  `wav:guid_<canonical-text>` id.

- WAV `LIST('wavl')` wave-list waveform container — the segmented
  waveform form per Microsoft RIFF MCI §3 "Storage of WAVE Data"
  (`<wave-data> -> { <data-ck> | <data-list> }`,
  `<wave-list> -> LIST('wavl' { <data-ck> | <silence-ck> }... )`).
  Previously a `wavl`-form WAV had no decodable audio (the LIST type
  fell through and no `data` anchor was set). The demuxer now resolves
  the first embedded `data` sub-chunk as the decode anchor and walks
  every segment. New keys: `wav:wavl.segment_count`,
  `wav:wavl.data_count`, `wav:wavl.data_bytes`, and per-segment
  `wav:wavl.<n>.kind` (`data`/`slnt`) / `.length`. Embedded `slnt`
  silence segments feed the shared `wav:slnt.*` accounting so the
  silent-sample totals match a top-level-`slnt` file (no zero/baseline
  samples are synthesised — §3 is explicit that `slnt` is a *count* of
  silent samples). A silence-only `wavl` (no `data` segment) is
  rejected as having no waveform; odd-length `data` segments respect
  RIFF word-alignment.
- WAV `WAVEFORMATEXTENSIBLE.dwChannelMask` decoding — the channel
  bitmap is now expanded into a human-readable `SPEAKER_*` layout
  string (`FRONT_LEFT+FRONT_RIGHT+...`), joined least-significant-bit
  first per the 18 documented flag bits (`SPEAKER_FRONT_LEFT 0x1` ..
  `SPEAKER_TOP_BACK_RIGHT 0x20000`) in
  `docs/container/riff/waveformatextensible/ms-waveformatextensible.html`.
  Surfaced through the new `wav:fmt.channel_layout` metadata key and a
  typed `WavDemuxer::channel_layout()` accessor. A `0` mask (no
  assigned positions) yields `None`; bits above the highest defined
  flag are preserved verbatim as `UNKNOWN(0x...)` so no round-trip
  information is dropped.
- WAV Acidizer `acid` chunk — read AND write sides with a typed
  accessor surface, layout per
  `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF
  Acidizer Tags" (24-byte little-endian body: flags bit-field at
  offset 0, root note at 4, six reserved bytes carried verbatim at
  6..12, beats at 12, meter at 16, tempo at 20). New public
  `wav::AcidChunk` type with `parse` / `to_bytes` and per-bit helpers
  (`one_shot` / `root_note_set` / `stretch` / `disk_based` /
  `high_octave`) plus the 48..=71 root-note name table
  (`root_note_name`). Demuxer surfaces `wav:acid.flags` /
  `.one_shot` / `.root_note_set` / `.stretch` / `.disk_based` /
  `.high_octave` / `.root_note` / `.root_note_name` / `.num_beats` /
  `.meter` / `.tempo` (+ `.reserved` hex when nonzero and `.body_len`
  when extension bytes ride past the fixed struct); truncated bodies
  are skipped-as-opaque. Muxer gains
  `WavMuxOptions::with_acid(AcidChunk)` emitting the chunk ahead of
  `data`. Six new lib tests: byte-layout golden pin, full metadata
  surface, out-of-table root note + reserved/body_len observability,
  truncated-skip, mux→demux round-trip (serialized chunk pinned
  verbatim in the output bytes), and typed-accessor equality.

- New public `wav::open_wav_demuxer` returning the concrete
  `WavDemuxer` so the typed accessor surface (`format_tag`,
  `channel_mask`, `subformat`, `acid`, …) is reachable without
  downcasting; the registry path now wraps it.

- WAV demuxer completes `LIST adtl` (Associated Data List) coverage per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Associated
  Data Chunk" — all four spec-defined sub-chunks are now parsed. The
  `file` sub-chunk (§3 "Embedded File Information",
  `file( <dwName:DWORD> <dwMedType:DWORD> <fileData:BYTE>... )`) is no
  longer skipped: `wav:adtl.file.<dwName>.med_type` renders the media
  type as FOURCC text when printable, the spec-allowed zero value as
  plain `0` ("This field can contain a zero value"), and hex otherwise;
  `wav:adtl.file.<dwName>.body_len` carries the embedded `fileData`
  payload length (the payload bytes themselves are not exposed through
  the string-typed metadata API — `body_len` keeps the attachment
  observable without pretending the parser interprets the inner
  format). The `ltxt` sub-chunk's four previously-undecoded locale
  WORDs now surface: `wav:adtl.ltxt.<dwName>.country` / `.language` /
  `.dialect` / `.code_page` (raw decimals, always emitted), with
  `.country_name` / `.language_name` resolved through the same §3
  Chapter-2 "Country Codes" / "Language and Dialect Codes" tables the
  `CSET` parser uses (emitted only when the code is in the spec's
  enumerated set; zero resolves to the tables' explicit `None` rows).
  Sub-chunks shorter than their fixed headers (8 bytes for `file`,
  20 for `ltxt`) remain skipped-as-opaque. Three new lib tests cover
  the `file` FOURCC + zero-med_type + payload-length path, the
  truncated-`file` opaque path, and the zero-locale `ltxt` path; the
  existing `ltxt` test now drives non-zero locale fields
  (UK / UK English / code page 1252) end-to-end.

- WAV demuxer recognises the `slnt` (Silence) chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 "Wave Data"
  (`<silence-ck> ➝ slnt( <dwSamples:DWORD> )` — "Count of silent
  samples"). Per the §3 note the chunk records a *count of silent
  samples* rather than carrying any PCM payload, and the correct
  playback fill value is context-dependent ("not necessarily a repeated
  zero volume or baseline sample"), so the parser surfaces accounting
  metadata without synthesising real zero/baseline samples into the
  decoded stream. Each encounter emits `wav:slnt.<n>.samples`
  (zero-based per-chunk `dwSamples` count) and updates the rolling
  aggregates `wav:slnt.count` (total `slnt` chunks seen) and
  `wav:slnt.total_samples` (cumulative silent-sample count). The §3
  grammar `<wave-data> ➝ { <data-ck> | <data-list> }` lets the chunk
  appear at the top level as a sibling of `data` (as well as inside a
  `wavl` LIST); the demuxer accounts for every top-level occurrence. A
  body shorter than the 4-byte `dwSamples` field is counted but treated
  as opaque — it contributes `0` to the running total and omits its
  per-chunk `samples` key, mirroring the other fixed-struct parsers;
  a body longer than 4 bytes decodes the leading DWORD and tolerates
  trailing forward-extension bytes. An odd-length body forces the
  standard RIFF §2 word-align pad and the `data` chunk that follows is
  located correctly. Files with no `slnt` chunk emit no `wav:slnt.*`
  keys at all — absence is observable. Seven new lib tests cover the
  single-chunk path, the multi-chunk accumulation path, the
  zero-samples in-range path, the no-slnt absence guard, the
  under-length opaque-body path, the over-length leading-DWORD path,
  and the coexistence path with `JUNK` + `LIST INFO` interleaved.
- WAV demuxer parses the `_PMX` chunk (Adobe XMP packet, the WAV/AVI
  carrier for an XMP serialised packet, catalogued in
  `docs/container/riff/metadata/exiftool-riff-tags.html` § "RIFF Main
  tags" — entry `'_PMX'`, family `XMP`, scope "AVI and WAV files").
  The FOURCC is little-endian "XMP_" reversed, the convention RIFF
  uses for chunks whose payload originates in a little-endian
  DWORD-aligned authoring tool; the payload is the XMP packet text
  exactly as it would appear in an XMP sidecar (`x:xmpmeta` wrapped
  in `<?xpacket begin=...?>` / `<?xpacket end=...?>` processing
  instructions). Surface shape mirrors the sibling third-party XML
  parsers (`iXML` / `<axml>`) so the consumer keeps a single mental
  model: the UTF-8 XMP text surfaces verbatim under `wav:xmp`
  (trimmed at the first NUL and surrounding whitespace so writers
  that NUL-pad a fixed-size XMP region for in-place editing do not
  leak padding into the text key), and the raw on-wire chunk-body
  length always surfaces under `wav:xmp.body_len` when the chunk is
  present — even for empty / NUL-only / whitespace-only bodies — so
  downstream tooling can distinguish "no `_PMX` chunk" from "an
  `_PMX` chunk reserved for later XMP authoring". An odd-length body
  forces the standard RIFF 1-byte pad and the `data` chunk that
  follows is located correctly. The XMP schema (RDF, namespace
  prefixes, xpacket processing instructions) is not interpreted at
  this layer; a higher-level XMP-aware crate can apply
  schema-specific decoding without re-walking the RIFF tree. Tests
  cover a canonical Adobe-style packet, NUL-padded reservation
  bodies, an empty body (`body_len = 0`, no text key),
  whitespace-only bodies, odd-length pad rounding, and the absence
  guard (a file without `_PMX` emits no `wav:xmp.*` keys).

- WAV demuxer parses the `<axml>` chunk per
  `docs/container/riff/metadata/ebu-tech3285s5-ADM.pdf` §3 ("AXML
  chunk definition"). The chunk carries a UTF-8 XML document
  (typically an EBUCore wrapper around an `<audioFormatExtended>`
  ADM document — see §4.2 — or an ISRC identifier declaration —
  §4.1) and is supplement-5's vehicle for transporting ADM
  metadata inside BWF / RF64 / BW64 files. The parser surfaces
  the textual payload verbatim under `wav:axml` (trimmed at the
  first NUL and surrounding whitespace, so writers that NUL-pad
  to reserve room for in-place editing of the ADM document do
  not leak the padding into the text key) and always emits the
  raw on-wire chunk-body length under `wav:axml.body_len`
  whenever the chunk is present — even for empty / NUL-only /
  whitespace-only bodies — so downstream tooling can distinguish
  "no `<axml>` chunk" from "an `<axml>` chunk reserved for later
  ADM authoring". An odd-length body forces the standard RIFF
  1-byte pad and the `data` chunk that follows is located
  correctly. The XML schema is not interpreted at this layer —
  the parser is schema-agnostic so a higher-level ADM-aware
  crate (or a downstream tool walking `wav:axml`) can apply the
  EBUCore / BS.2076 decoding without re-walking the RIFF tree.
  Six new lib tests cover the canonical EBUCore-wrapped ADM
  document path, the ISRC identifier example from §4.1, the
  NUL-padded reservation path, the empty-body absence guard, the
  whitespace-only placeholder path, and the odd-length-body
  padding regression guard.
- WAV demuxer recognises the `JUNK` (Filler) chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §2 "JUNK
  (Filler) Chunk" ("A JUNK chunk represents padding, filler or
  outdated information. It contains no relevant data; it is a space
  filler of arbitrary size."). The chunk body is deliberately not
  surfaced — its bytes are spec-defined as having no semantic content
  — but the demuxer accounts for every `JUNK` chunk seen so a
  downstream tool can observe how much filler the producer reserved
  (commonly for in-place editing) without re-walking the file. Each
  encounter emits `wav:junk.<n>.body_len` (zero-based, per-chunk
  payload size) and updates the rolling aggregates `wav:junk.count`
  (total `JUNK` chunks seen) and `wav:junk.total_bytes` (cumulative
  payload size; excludes the 8-byte chunk header and the implicit
  word-align pad byte). Multiple `JUNK` chunks are allowed; empty
  bodies (`size = 0`) still increment the count. Files with no `JUNK`
  chunk emit no `wav:junk.*` keys at all — absence is observable.
  Odd-length bodies forward the chunk-walk past the implicit RIFF §2
  word-align pad byte correctly. Six new lib tests cover the single-
  chunk path, the multi-chunk accumulation path, the zero-length
  in-range path, the no-JUNK absence guard, the odd-length-body
  padding regression guard, and the coexistence path with `LIST INFO`
  + `CSET` interleaved around the JUNK chunks.
- New `filter` module exposing a typed scalar form of the Reinhard 2002
  simple global tone-mapping operator (`Ld = L / (1 + L)`) per
  `docs/image/filter/tone-mapping-operators.md` §2.2. The forward map is
  surfaced as `SceneLuminance::to_display`; the closed-form inverse
  `L = Ld / (1 − Ld)` is surfaced as `DisplayLuminance::to_scene`. The
  two newtypes separate scene (pre-tone-map, non-negative finite) and
  display (post-tone-map, `[0, 1)`) luminance domains so callers can't
  swap them at a function boundary. Constructors reject invalid inputs
  (negative scene values, NaN, non-finite, `Ld ≥ 1.0`). Round-trip is
  bit-exact at the three §2.2 reference points (`L = 0 → 0`,
  `L = 1 → 0.5`, `L = 3 → 0.75`) and stays within a relative error of
  `1e-9` across an 11-point span from `1e-9` to `1e6` of scene
  luminance. Six in-crate unit tests plus three integration tests on
  the public surface cover constructor rejection, known curve values,
  the `Ld < 1` asymptote, and round-trip in both directions.
- WAV demuxer recognises the `RF64` and `BW64` top-level form magics
  (EBU Tech 3306 v1 §3 and ITU-R BS.2088 / EBU Tech 3306 v2) and
  parses the mandatory `ds64` chunk that follows. The 28-byte fixed
  prefix decodes the 64-bit `riffSize`, `dataSize` and `sampleCount`
  overrides plus a `tableLength` count of `(chunkId, chunkSize64)`
  records — the optional table of per-chunk-ID 64-bit size overrides
  for any non-`data` chunk that exceeds 4 GiB. When a chunk's 32-bit
  on-wire size carries the `0xFFFFFFFF` sentinel the demuxer promotes
  it to 64-bit via the dedicated `ds64.dataSize` (for `data`) or the
  `ds64.table` lookup (for any other FOURCC). The legacy
  `fact.dwFileSize` is likewise promoted to `ds64.sampleCount` when
  it carries the sentinel. Surfaced under `wav:rf64.magic`,
  `wav:rf64.riff_size`, `wav:rf64.data_size`, `wav:rf64.sample_count`,
  `wav:rf64.table.count`, per-entry `wav:rf64.table.<i>.id` /
  `.size`, and `wav:rf64.body_len`. Five new tests cover the RF64
  ds64-promoted PCM path, the BW64 magic accepted on the same path,
  the non-`data` table-lookup path through a `LIST INFO` whose size
  is the sentinel, a sentinel-without-`ds64` rejection, and a
  short-body `ds64` rejection.
- WAV demuxer now resolves the complete Microsoft RIFF MCI §3 "INFO
  List Chunk" baseline — all 23 sub-IDs registered by the 1991 spec
  per `docs/container/riff/metadata/microsoft-riffmci.pdf` pp. 2-14
  to 2-16. The eleven previously-handled sub-IDs (`INAM`, `IART`,
  `IPRD`, `ICMT`, `ICRD`, `IGNR`, `ICOP`, `IENG`, `ITCH`, `ISFT`,
  `ISBJ`) keep their conventional short-form key names; the twelve
  additions surface under spec-derived snake_case names: `IARL` →
  `archival_location`, `ICMS` → `commissioned`, `ICRP` → `cropped`,
  `IDIM` → `dimensions`, `IDPI` → `dpi`, `IKEY` → `keywords`,
  `ILGT` → `lightness`, `IMED` → `medium`, `IPLT` →
  `palette_setting`, `ISHP` → `sharpness`, `ISRC` → `source`,
  `ISRF` → `source_form`. The non-baseline `ITRK` → `track`
  extension is retained for compatibility with the tag-writer
  ecosystem. Unknown sub-IDs are skipped silently rather than
  synthesised into ad-hoc keys, matching the §3 forward-compatibility
  rule that "new chunks may be defined".
- WAV demuxer parses the `CSET` (Character Set) chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3
  "CSET (Character Set) Chunk". The canonical 8-byte body carries four
  16-bit little-endian fields — `wCodePage`, `wCountryCode`,
  `wLanguageCode`, `wDialect` — declaring the code page, country,
  language and dialect that file elements (notably the `LIST INFO`
  ZSTR sub-chunks) are interpreted under. The parser surfaces raw
  values under `wav:cset.code_page` / `.country` / `.language` /
  `.dialect`, resolves the §3 "Country Codes" and "Language and
  Dialect Codes" enumerations to human-readable
  `wav:cset.country_name` / `wav:cset.language_name` keys, and always
  emits `wav:cset.body_len` so writers that extend the chunk past the
  canonical 8 bytes are still observable to downstream tooling.
  All-zero fields are honoured as the spec's "use defaults" form (ISO
  8859/1 / USA / US English). Bodies shorter than 8 bytes are treated
  as opaque (only `body_len` is emitted); bodies longer than 8 bytes
  tolerate the trailing region for forward compatibility. CSET coexists
  with `LIST INFO`: a single file may carry both, and the INFO sub-IDs
  still resolve through the existing standard-tag mapping.
- WAV demuxer parses the `iXML` chunk (the third-party
  production-recorder metadata block catalogued in
  `docs/container/riff/metadata/exiftool-riff-tags.html` § `iXML`
  and discussed in
  `docs/container/riff/metadata/README.md` § "iXML"). The chunk
  carries a UTF-8 XML document — `IXML_VERSION`, `PROJECT`,
  `SCENE`, `TAKE`, `TAPE`, `NOTE`, `UBITS`, `FILE_UID`, a `BWF`
  sub-group mirroring the `bext` fields and a `TRACK_LIST` of
  per-track `NAME` / `FUNCTION` / `CHANNEL_INDEX` mappings. The
  parser surfaces the text payload verbatim under `wav:ixml`
  (trimmed at the first NUL and surrounding whitespace so writers
  that NUL-pad the body to a fixed size for in-place editing do
  not surface spurious trailing bytes) and always emits the raw
  on-wire chunk-body length under `wav:ixml.body_len` whenever the
  chunk is present — even for empty / NUL-only / whitespace-only
  bodies — so downstream tooling can distinguish "no `iXML`
  chunk" from "an `iXML` chunk reserved for later population". An
  odd-length body forces the standard RIFF 1-byte pad and the
  `data` chunk that follows is located correctly.

## [0.0.8](https://github.com/OxideAV/oxideav-basic/compare/v0.0.7...v0.0.8) - 2026-05-30

### Other

- parse plst (Playlist) chunk per Microsoft RIFF MCI §3
- parse smpl (Sampler) + inst (Instrument) chunks
- parse cue + LIST adtl per Microsoft RIFF MCI §3
- parse bext Broadcast Audio Extension chunk (EBU Tech 3285)
- fix intra-doc links to WavMuxOptions / WavDemuxer methods
- A-law/μ-law tag 6/7 + WAVEFORMATEXTENSIBLE end-to-end

### Added

- WAV demuxer parses the `fact` chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 ("FACT
  Chunk"). The 4-byte `dwFileSize` field (per-channel sample
  count) surfaces under `wav:fact.sample_count` as a decimal
  string and becomes the authoritative `StreamInfo::duration`
  for the open stream — important for non-PCM WAV streams where
  the `data_size / block_align` heuristic is meaningless because
  one byte of payload no longer maps to one sample. For PCM
  streams the heuristic and the `fact` value should agree; when
  they don't (compressed bitstream riding a non-PCM
  `wFormatTag`, or a writer that lied about the count) the
  parser additionally surfaces `wav:fact.mismatch` with both
  numbers so a downstream tool can flag the file rather than
  silently trusting one over the other. The spec explicitly
  reserves trailing bytes past `dwFileSize` for future extension
  fields, so a body longer than 4 bytes surfaces its total
  length under `wav:fact.body_len` (the extension bytes
  themselves are opaque to this parser by spec); a body shorter
  than 4 bytes is treated as opaque-and-skipped without
  panicking the parser. The WAV muxer now emits a `fact` chunk
  between `fmt ` and `data` for every non-PCM on-wire
  `wFormatTag` (G.711 A-law / μ-law and the EXTENSIBLE
  `0xFFFE` escape hatch) — RIFF MCI §3 says the chunk is
  required for any `wFormatTag != WAVE_FORMAT_PCM`; for plain
  PCM where the chunk is optional we skip emitting it to keep
  the post-r193 PCM muxer output byte-identical to pre-r193.
- WAV demuxer parses the `plst` (Playlist) chunk per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3 ("Playlist
  Chunk"). The 4-byte `dwSegments` count is followed by N × 12-byte
  `<play-segment>` records (`dwName`, `dwLength`, `dwLoops`). Each
  segment surfaces under `wav:plst.count` plus per-segment
  `wav:plst.<n>.cue_id` / `.length` / `.loops` keys. Unlike the `cue `
  chunk which keys by `dwName`, the playlist is keyed by zero-based
  segment position because the spec explicitly allows a single cue
  id to appear in multiple segments (a cue replayed N times = N
  segments with identical `dwName`), so a `dwName`-indexed key would
  collide. A `dwSegments` count that exceeds the chunk body is
  clamped to the records that actually fit; bodies shorter than the
  4-byte segment-count header are treated as opaque and skipped.
- WAV demuxer parses the `smpl` (Sampler) and `inst` (Instrument)
  chunks per `docs/container/riff/metadata/exiftool-riff-tags.html` §
  "RIFF Sampler Tags" / "RIFF Instrument Tags" and summarised in
  `docs/container/riff/metadata/README.md` § "Sampler / Instrument
  chunks". `smpl` is a 36-byte fixed header (`Manufacturer`,
  `Product`, `SamplePeriod`, `MIDIUnityNote`, `MIDIPitchFraction`,
  `SMPTEFormat`, `SMPTEOffset`, `cSampleLoops`, `cbSamplerData`)
  followed by N × 24-byte loop records (`dwCuePointID`, `dwType`,
  `dwStart`, `dwEnd`, `dwFraction`, `dwPlayCount`). `inst` is a
  7-byte fixed struct (`UnshiftedNote`, `FineTune`, `Gain`, `LowNote`,
  `HighNote`, `LowVelocity`, `HighVelocity`). Both surface through
  `Demuxer::metadata` under `wav:smpl.*` / `wav:inst.*` keys; the
  `SMPTEOffset` `DWORD` is rendered as canonical `HH:MM:SS:FF`, and
  `FineTune` / `Gain` are decoded as signed `i8` (so `-3` cents shows
  as `-3`, not `253`). A `cSampleLoops` count exceeding what the chunk
  body actually carries is clamped to the records that fit; bodies
  shorter than the 36-byte (`smpl`) or 7-byte (`inst`) fixed header
  are treated as opaque and skipped.
- WAV demuxer parses the `cue ` chunk and the `LIST adtl` (Associated
  Data List) sub-chunks per
  `docs/container/riff/metadata/microsoft-riffmci.pdf` §3
  ("Cue-Points Chunk" + "Playlist Chunk" + "Associated Data Chunk").
  The cue-point table surfaces under `wav:cue.count` plus per-point
  `wav:cue.<dwName>.position` / `.fcc_chunk` / `.chunk_start` /
  `.block_start` / `.sample_offset` (all values as decimal strings;
  `fcc_chunk` rendered as a 4-byte ASCII FOURCC when printable). The
  `LIST adtl` body's `labl` / `note` / `ltxt` sub-chunks surface
  under `wav:adtl.labl.<dwName>` / `wav:adtl.note.<dwName>` /
  `wav:adtl.ltxt.<dwName>.{length,purpose,text}`. `file` sub-chunks
  (embedded media files) are skipped — their bytes do not fit the
  string-typed metadata API. A `dwCuePoints` count that exceeds the
  chunk body is clamped to the records that actually fit so a writer
  that lies about the count cannot panic the parser; `labl` / `note`
  shorter than the 4-byte `dwName` header and `ltxt` shorter than the
  20-byte fixed header are treated as opaque and skipped.
- WAV demuxer parses the `bext` Broadcast Audio Extension chunk per
  `docs/container/riff/metadata/ebu-tech3285-bwf.pdf` (EBU Tech 3285 v2
  §2.3 `BROADCAST_EXT`). The 602-byte fixed struct plus variable
  `CodingHistory` is surfaced through `Demuxer::metadata` under
  `wav:bext.description` / `.originator` / `.originator_reference` /
  `.origination_date` / `.origination_time` / `.time_reference` (64-bit
  sample count reassembled from the low/high `DWORD`s) / `.version` /
  `.umid` (hex-encoded SMPTE-330M UMID, v1+ only when non-zero) /
  `.coding_history`. For BWF v2 the five loudness `WORD`s are decoded
  from the §2.4 `round(100 × value)` fixed-point to two decimals under
  `.loudness_value` / `.loudness_range` / `.max_true_peak_level` /
  `.max_momentary_loudness` / `.max_short_term_loudness`. Loudness and
  UMID keys are omitted for v0/v1 streams that leave those fields zero;
  a `bext` chunk shorter than 602 bytes is skipped as opaque.
- WAV demuxer dispatches `WAVE_FORMAT_ALAW (0x0006)` and
  `WAVE_FORMAT_MULAW (0x0007)` streams to the `pcm_alaw` / `pcm_mulaw`
  codecs (host runtime applies G.711 decode through `oxideav-g711`).
  Decoded `SampleFormat` hint surfaces as `S16`; `bit_rate` reflects
  the on-wire 8-bit rate (not the post-decode S16 rate).
- WAV `WAVE_FORMAT_EXTENSIBLE (0xFFFE)` end-to-end handling per
  `docs/container/riff/waveformatextensible/README.md`: the 22-byte
  extension's `wValidBitsPerSample`, `dwChannelMask` and SubFormat
  GUID are parsed and exposed both through typed accessors on
  `WavDemuxer` (`format_tag`, `valid_bits_per_sample`, `channel_mask`,
  `subformat`, `subformat_text`) and through `Demuxer::metadata`
  under the keys `wav:fmt.valid_bits_per_sample` /
  `wav:fmt.channel_mask` / `wav:fmt.subformat`. Well-known
  `KSDATAFORMAT_SUBTYPE_*` GUIDs (PCM, IEEE_FLOAT, ALAW, MULAW)
  resolve to the legacy codec ids; unknown GUIDs synthesise a
  `wav:guid_<canonical-text>` id so downstream `make_decoder`
  failures name the actual GUID rather than the opaque `0xFFFE` tag.
- WAV muxer `open_muxer_with` + `WavMuxOptions::with_extensible(mask)`
  emits a 40-byte `WAVEFORMATEXTENSIBLE` `fmt ` chunk with caller-
  supplied `dwChannelMask`. `wValidBitsPerSample` defaults to the
  container `wBitsPerSample` and SubFormat defaults to the well-known
  GUID for the codec; both can be overridden via
  `with_valid_bits_per_sample` / `with_subformat`.
- New public constants `WAVE_FORMAT_PCM` / `WAVE_FORMAT_IEEE_FLOAT` /
  `WAVE_FORMAT_ALAW` / `WAVE_FORMAT_MULAW` / `WAVE_FORMAT_EXTENSIBLE`
  in the `wav` module for muxer callers.

### Fixed

- WAV demuxer rejects `WAVE_FORMAT_EXTENSIBLE` streams whose
  `cbSize < 22` (the spec mandates a 22-byte extension; previously
  the demuxer silently dropped the extension fields).
- WAV demuxer stamps the on-wire `wFormatTag` onto
  `CodecParameters.tag` so consumers can distinguish legacy
  `WAVEFORMATEX` from EXTENSIBLE round-trips without reparsing.

## [0.0.7](https://github.com/OxideAV/oxideav-basic/compare/v0.0.6...v0.0.7) - 2026-05-06

### Other

- reframe FFI claim — HW-engine crates use OS FFI by necessity
- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-basic/pull/502))

## [0.0.6](https://github.com/OxideAV/oxideav-basic/compare/v0.0.5...v0.0.6) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- add YUV4MPEG2 demuxer + muxer
- drop unused PcmDecoder.sample_rate field (slim-frame leftover)
- adopt slim AudioFrame shape
- pin release-plz to patch-only bumps

### Added

- Y4M (YUV4MPEG2) raw-video demuxer + muxer (`y4m` module). Handles
  4:2:0 / 4:2:2 / 4:4:4 / mono at 8/10/12-bit, preserves header
  `X<key>=<val>` extensions in `Demuxer::metadata`, and probes on the
  `YUV4MPEG2 ` magic. Frames are emitted as `rawvideo` packets.

## [0.0.5](https://github.com/OxideAV/oxideav-basic/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.4](https://github.com/OxideAV/oxideav-basic/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()
