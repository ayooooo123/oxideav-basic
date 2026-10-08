//! The fmt chunk's codec bytes and nBlockAlign reach the decoder as FFmpeg's
//! `ff_get_wav_header` (libavformat/riffdec.c) hands them over: `extradata`
//! is the cbSize bytes after WAVEFORMATEX (after the 22-byte extension for
//! WAVE_FORMAT_EXTENSIBLE, after the 12-byte HEAACWAVEINFO for 0x1610),
//! clamped to the chunk, and `options["block_align"]` is nBlockAlign.
//! EXTENSIBLE subformat GUIDs resolve as FFmpeg resolves them: the three
//! base GUIDs carry a WAVE tag in their first 32 bits, and the GUIDs of
//! FFmpeg's `ff_codec_wav_guids` name their codec.

use std::io::Cursor;

use oxideav_basic::wav::open_wav_demuxer_with;
use oxideav_core::{CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Demuxer};

/// A WAV of `fmt` and 4096 zero bytes of data.
fn wav_with_fmt(fmt: &[u8]) -> Vec<u8> {
    let mut body = b"WAVEfmt ".to_vec();
    body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    body.extend_from_slice(fmt);
    if fmt.len() % 2 == 1 {
        body.push(0);
    }
    body.extend_from_slice(b"data");
    body.extend_from_slice(&4096u32.to_le_bytes());
    body.extend_from_slice(&[0; 4096]);
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// The 16 bytes of WAVEFORMAT plus wBitsPerSample.
fn base(tag: u16, channels: u16, rate: u32, block_align: u16, bits: u16) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&tag.to_le_bytes());
    f.extend_from_slice(&channels.to_le_bytes());
    f.extend_from_slice(&rate.to_le_bytes());
    f.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
    f.extend_from_slice(&block_align.to_le_bytes());
    f.extend_from_slice(&bits.to_le_bytes());
    f
}

/// WAVEFORMATEX with a declared cbSize, followed by `extra`.
fn waveformatex(tag: u16, channels: u16, block_align: u16, bits: u16, cb_size: u16, extra: &[u8]) -> Vec<u8> {
    let mut f = base(tag, channels, 44_100, block_align, bits);
    f.extend_from_slice(&cb_size.to_le_bytes());
    f.extend_from_slice(extra);
    f
}

/// WAVEFORMATEXTENSIBLE with `guid` and `extra` after the extension.
fn extensible(channels: u16, block_align: u16, bits: u16, guid: [u8; 16], extra: &[u8]) -> Vec<u8> {
    let mut f = base(0xFFFE, channels, 44_100, block_align, bits);
    f.extend_from_slice(&(22 + extra.len() as u16).to_le_bytes());
    f.extend_from_slice(&bits.to_le_bytes());
    f.extend_from_slice(&(if channels == 2 { 3u32 } else { 4 }).to_le_bytes());
    f.extend_from_slice(&guid);
    f.extend_from_slice(extra);
    f
}

fn registry() -> CodecRegistry {
    let mut reg = CodecRegistry::new();
    oxideav_basic::register_codecs(&mut reg);
    for (id, tag) in [("atrac3", 0x0270), ("adpcm_ms", 0x0002), ("aac", 0x1610)] {
        reg.register(CodecInfo::new(CodecId::new(id)).tag(CodecTag::wave_format(tag)));
    }
    reg
}

fn params(fmt: &[u8]) -> CodecParameters {
    let reg = registry();
    let d = open_wav_demuxer_with(Box::new(Cursor::new(wav_with_fmt(fmt))), &reg).unwrap();
    d.streams()[0].params.clone()
}

fn block_align(p: &CodecParameters) -> Option<&str> {
    p.options.get("block_align")
}

/// FATE's atrac3/mc_sich_at3_066_small.wav fmt: ATRAC3 needs its 14 bytes
/// (the decoder rejects "extradata size 0") and its 384-byte frames.
const AT3_EXTRA: [u8; 14] = [0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];

#[test]
fn atrac3_gets_its_extradata_and_block_align() {
    let p = params(&waveformatex(0x0270, 2, 384, 0, 14, &AT3_EXTRA));
    assert_eq!(p.codec_id, CodecId::new("atrac3"));
    assert_eq!(p.extradata, AT3_EXTRA);
    assert_eq!(block_align(&p), Some("384"));
}

#[test]
fn ms_adpcm_coefficients_are_extradata() {
    let mut extra = vec![0xf4, 0x03, 7, 0];
    for (c1, c2) in [(256i16, 0i16), (512, -256), (0, 0), (192, 64), (240, 0), (460, -208), (392, -232)] {
        extra.extend_from_slice(&c1.to_le_bytes());
        extra.extend_from_slice(&c2.to_le_bytes());
    }
    let p = params(&waveformatex(0x0002, 2, 1024, 4, 32, &extra));
    assert_eq!(p.codec_id, CodecId::new("adpcm_ms"));
    assert_eq!(p.extradata, extra);
    assert_eq!(block_align(&p), Some("1024"));
}

#[test]
fn pcm_has_its_block_align_and_no_extradata() {
    let p = params(&base(1, 2, 44_100, 4, 16));
    assert_eq!((p.extradata.len(), block_align(&p)), (0, Some("4")));
    let p = params(&waveformatex(1, 2, 4, 16, 0, &[]));
    assert_eq!((p.extradata.len(), block_align(&p)), (0, Some("4")));
}

/// FFmpeg takes FFMIN(chunk bytes left, cbSize).
#[test]
fn a_cb_size_past_the_chunk_is_clamped() {
    let p = params(&waveformatex(0x0270, 2, 384, 0, 14, &AT3_EXTRA[..10]));
    assert_eq!(p.extradata, AT3_EXTRA[..10]);
}

const ATRAC3PLUS_GUID: [u8; 16] = [0xBF, 0xAA, 0x23, 0xE9, 0x58, 0xCB, 0x71, 0x44, 0xA1, 0x19, 0xFF, 0xFA, 0x01, 0xE4, 0xCE, 0x62];
const ATRAC9_GUID: [u8; 16] = [0xD2, 0x42, 0xE1, 0x47, 0xBA, 0x36, 0x8D, 0x4D, 0x88, 0xFC, 0x61, 0x65, 0x4F, 0x8C, 0x83, 0x6C];
const AC3_GUID: [u8; 16] = [0x2C, 0x80, 0x6D, 0xE0, 0x46, 0xDB, 0xCF, 0x11, 0xB4, 0xD1, 0x00, 0x80, 0x5F, 0x6C, 0xBB, 0xEA];

/// Sony's ATRAC3plus and ATRAC9 WAVs: the codec from the GUID, the codec
/// bytes after the extension.
#[test]
fn extensible_codec_guids_name_their_codec() {
    let extra: Vec<u8> = (1..=12).collect();
    for (guid, id) in [(ATRAC3PLUS_GUID, "atrac3plus"), (ATRAC9_GUID, "atrac9"), (AC3_GUID, "ac3")] {
        let p = params(&extensible(2, 1488, 0, guid, &extra));
        assert_eq!(p.codec_id, CodecId::new(id));
        assert_eq!(p.extradata, extra, "{id}");
        assert_eq!(block_align(&p), Some("1488"), "{id}");
    }
}

/// KSDATAFORMAT_SUBTYPE_ADPCM: MS ADPCM with its coefficients after the
/// extension.
#[test]
fn a_tag_guid_keeps_the_codec_bytes() {
    let mut guid = [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71];
    let extra = [0xf4, 0x03, 0, 0];
    let p = params(&extensible(2, 1024, 4, guid, &extra));
    assert_eq!((p.codec_id.as_str(), p.extradata.as_slice()), ("adpcm_ms", &extra[..]));
    // FFmpeg's other two base GUIDs carry the tag the same way.
    guid = [0x01, 0x00, 0x00, 0x00, 0x21, 0x07, 0xD3, 0x11, 0x86, 0x44, 0xC8, 0xC1, 0xCA, 0x00, 0x00, 0x00];
    assert_eq!(params(&extensible(2, 4, 16, guid, &[])).codec_id, CodecId::new("pcm_s16le"), "ambisonic base");
    guid = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA];
    assert_eq!(params(&extensible(2, 4, 16, guid, &[])).codec_id, CodecId::new("pcm_s16le"), "broken base");
}

/// HEAACWAVEFORMAT: the AudioSpecificConfig after the 12-byte
/// HEAACWAVEINFO.
#[test]
fn heaac_wave_info_is_not_extradata() {
    let mut extra = vec![0, 0, 0xfe, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    extra.extend_from_slice(&[0x12, 0x10]);
    let p = params(&waveformatex(0x1610, 2, 1, 0, 14, &extra));
    assert_eq!((p.codec_id.as_str(), p.extradata.as_slice()), ("aac", &[0x12u8, 0x10][..]));
}
