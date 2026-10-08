//! Every linear-PCM id a demuxer hands over decodes to its samples:
//! big-endian, unsigned (offset binary) and 64-bit integer ids convert to
//! the little-endian formats, D-Cinema 24-bit words (`pcm_s24daud`) to the
//! 16-bit samples FFmpeg decodes from them, and each decoder reports the
//! format it outputs.

use oxideav_core::{AudioFormat, CodecId, CodecParameters, CodecRegistry, Frame, Packet, SampleFormat, TimeBase};

/// Decodes one packet of `id`; the frame's bytes, its sample count and the
/// format the decoder reports.
fn decode(id: &str, channels: u16, data: Vec<u8>) -> (Vec<u8>, u32, Option<AudioFormat>) {
    let mut reg = CodecRegistry::new();
    oxideav_basic::register_codecs(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new(id));
    params.channels = Some(channels);
    params.sample_rate = Some(44_100);
    let mut dec = reg.first_decoder(&params).unwrap_or_else(|e| panic!("{id}: {e}"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 44_100), data)).unwrap();
    let Frame::Audio(af) = dec.receive_frame().unwrap() else { panic!("{id}: not audio") };
    (af.data.concat(), af.samples, dec.output_audio_format())
}

fn format(sample_format: SampleFormat, channels: u16) -> Option<AudioFormat> {
    Some(AudioFormat { sample_format, sample_rate: 44_100, channels })
}

const S16: [i16; 6] = [0, 1, -2, 0x1234, i16::MIN, i16::MAX];
const S24: [i32; 6] = [0, 1, -2, 0x12_3456, -0x80_0000, 0x7f_ffff];
const S32: [i32; 6] = [0, 1, -2, 0x1234_5678, i32::MIN, i32::MAX];

fn le16(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()).collect()
}
fn le24(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()[..3].to_vec()).collect()
}
fn le32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()).collect()
}
/// Each `width`-byte sample of `le` in the other byte order.
fn swapped(le: &[u8], width: usize) -> Vec<u8> {
    le.chunks(width).flat_map(|c| c.iter().rev().copied().collect::<Vec<_>>()).collect()
}
/// `le` as offset binary: the sign bit of each `width`-byte sample flipped.
fn offset_binary(le: &[u8], width: usize) -> Vec<u8> {
    let mut v = le.to_vec();
    for c in v.chunks_mut(width) {
        c[width - 1] ^= 0x80;
    }
    v
}

#[test]
fn big_endian_integers_decode_to_little_endian() {
    let cases = [
        ("pcm_s16be", 2, le16(&S16), SampleFormat::S16),
        ("pcm_s24be", 3, le24(&S24), SampleFormat::S24),
        ("pcm_s32be", 4, le32(&S32), SampleFormat::S32),
    ];
    for (id, width, le, fmt) in cases {
        let (out, samples, reported) = decode(id, 2, swapped(&le, width));
        assert_eq!(out, le, "{id}");
        assert_eq!(samples, 3, "{id}: 6 samples, 2 channels");
        assert_eq!(reported, format(fmt, 2), "{id}");
    }
}

#[test]
fn big_endian_floats_decode_to_little_endian() {
    let f32s: Vec<u8> = [0.5f32, -0.25, 1.0, -1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let f64s: Vec<u8> = [0.5f64, -0.25, 1.0, -1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let (out, samples, reported) = decode("pcm_f32be", 1, swapped(&f32s, 4));
    assert_eq!((out, samples, reported), (f32s, 4, format(SampleFormat::F32, 1)));
    let (out, samples, reported) = decode("pcm_f64be", 1, swapped(&f64s, 8));
    assert_eq!((out, samples, reported), (f64s, 4, format(SampleFormat::F64, 1)));
}

#[test]
fn unsigned_integers_decode_to_signed() {
    let cases = [
        ("pcm_u16le", 2, le16(&S16), false, SampleFormat::S16),
        ("pcm_u16be", 2, le16(&S16), true, SampleFormat::S16),
        ("pcm_u24le", 3, le24(&S24), false, SampleFormat::S24),
        ("pcm_u24be", 3, le24(&S24), true, SampleFormat::S24),
        ("pcm_u32le", 4, le32(&S32), false, SampleFormat::S32),
        ("pcm_u32be", 4, le32(&S32), true, SampleFormat::S32),
    ];
    for (id, width, le, big_endian, fmt) in cases {
        let wire = offset_binary(&le, width);
        let wire = if big_endian { swapped(&wire, width) } else { wire };
        let (out, samples, reported) = decode(id, 1, wire);
        assert_eq!(out, le, "{id}");
        assert_eq!(samples, 6, "{id}");
        assert_eq!(reported, format(fmt, 1), "{id}");
    }
    // Offset binary's midpoint is silence and its zero full scale.
    let (out, ..) = decode("pcm_u16le", 1, vec![0x00, 0x80, 0x00, 0x00, 0xff, 0xff]);
    assert_eq!(out, le16(&[0, i16::MIN, i16::MAX]));
}

#[test]
fn signed_64_bit_integers_decode_to_float() {
    let values = [i64::MIN, 1 << 62, 0, -(1 << 61), -1 << 63 >> 1];
    let expect: Vec<u8> = [-1.0f64, 0.5, 0.0, -0.25, -0.5].iter().flat_map(|v| v.to_le_bytes()).collect();
    let le: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let be: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    assert_eq!(decode("pcm_s64le", 1, le), (expect.clone(), 5, format(SampleFormat::F64, 1)));
    assert_eq!(decode("pcm_s64be", 1, be), (expect, 5, format(SampleFormat::F64, 1)));
}

/// FFmpeg 2da55bf's pcm_s24daud encoder wrote these 12 words for the
/// samples; its decoder returns the samples from them, and the 6 words
/// with every flag bit set decode as FFmpeg decodes them.
#[test]
fn d_cinema_words_decode_as_ffmpeg() {
    let words = hex("0000000800000ffff002c480033b700fffe00000100ff000000fe00aaaa00555500f0f00");
    let samples = [0, 1, -1, 4660, -4660, 32767, -32768, 255, 32512, 21845, -21846, 3855];
    let (out, n, reported) = decode("pcm_s24daud", 6, words);
    assert_eq!(out, le16(&samples));
    assert_eq!(n, 2, "12 words, 6 channels");
    assert_eq!(reported, format(SampleFormat::S16, 6));
    let flagged = hex("f1234ffabcdf00000ff00000123456fedcba");
    let (out, n, _) = decode("pcm_s24daud", 2, flagged);
    assert_eq!(out, le16(&[11336, -19499, 0, 0, -23868, -11337]));
    assert_eq!(n, 3);
}

/// The little-endian ids report what they output too (pcm_s8 is signed
/// whatever a container declares).
#[test]
fn native_ids_report_their_format() {
    assert_eq!(decode("pcm_s8", 1, vec![0x80, 0x7f]), (vec![0x80, 0x7f], 2, format(SampleFormat::S8, 1)));
    assert_eq!(decode("pcm_s16le", 2, le16(&S16)).2, format(SampleFormat::S16, 2));
}

#[test]
fn a_packet_cut_inside_a_sample_is_rejected() {
    let mut reg = CodecRegistry::new();
    oxideav_basic::register_codecs(&mut reg);
    for (id, len) in [("pcm_s24be", 4), ("pcm_s24daud", 5), ("pcm_s64le", 12), ("pcm_u16be", 3)] {
        let mut params = CodecParameters::audio(CodecId::new(id));
        params.channels = Some(1);
        params.sample_rate = Some(44_100);
        let mut dec = reg.first_decoder(&params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 44_100), vec![0; len])).unwrap();
        assert!(dec.receive_frame().is_err(), "{id}: {len} bytes");
    }
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}
