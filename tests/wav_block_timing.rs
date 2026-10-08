//! Block-coded WAV (ADPCM, Ulead DV audio, …) is timed in samples, as
//! FFmpeg's wavdec and demuxer layer time it, not one tick per block:
//! packets are FFmpeg's (`ff_pcm_default_packet_size`, about 100 ms of
//! whole blocks), each carries the pts its samples start at where the
//! block format fixes how many samples a byte holds
//! (`get_audio_frame_duration`), and otherwise only the first packet after
//! open or a seek carries one (the decoder's output places the rest). The
//! duration and seeks follow FFmpeg's `wav_read_header` and
//! `ff_pcm_read_seek`. The expected tables are FFmpeg 2da55bf's for the
//! same headers (ffprobe -show_packets).

use std::io::Cursor;

use oxideav_basic::wav::open_wav_demuxer_with;
use oxideav_core::{CodecId, CodecInfo, CodecRegistry, CodecTag, Demuxer, Error};

/// A WAV of `blocks` blocks of `block_align` bytes (each block's bytes its
/// index), with an optional `fact` sample count.
fn wav(tag: u16, channels: u16, rate: u32, byte_rate: u32, block_align: u16, bits: u16, blocks: usize, fact: Option<u32>) -> Vec<u8> {
    let data: Vec<u8> = (0..blocks).flat_map(|b| vec![b as u8; block_align as usize]).collect();
    let mut fmt = Vec::new();
    for v in [tag, channels] {
        fmt.extend_from_slice(&v.to_le_bytes());
    }
    fmt.extend_from_slice(&rate.to_le_bytes());
    fmt.extend_from_slice(&byte_rate.to_le_bytes());
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());
    let mut body = b"WAVE".to_vec();
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    body.extend_from_slice(&fmt);
    if let Some(n) = fact {
        body.extend_from_slice(b"fact");
        body.extend_from_slice(&4u32.to_le_bytes());
        body.extend_from_slice(&n.to_le_bytes());
    }
    body.extend_from_slice(b"data");
    body.extend_from_slice(&(data.len() as u32).to_le_bytes());
    body.extend_from_slice(&data);
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// A registry claiming the WAVE tags FFmpeg maps to these codec ids.
fn registry() -> CodecRegistry {
    let mut reg = CodecRegistry::new();
    for (id, tag) in [("adpcm_ms", 0x0002), ("adpcm_ima_wav", 0x0011), ("adpcm_yamaha", 0x0020), ("dvaudio", 0x0216), ("mp3", 0x0055)] {
        reg.register(CodecInfo::new(CodecId::new(id)).tag(CodecTag::wave_format(tag)));
    }
    reg
}

type Row = (Option<i64>, Option<i64>, usize);

fn packets(file: Vec<u8>) -> (Box<dyn Demuxer>, Vec<Row>) {
    let reg = registry();
    let mut d: Box<dyn Demuxer> = Box::new(open_wav_demuxer_with(Box::new(Cursor::new(file)), &reg).unwrap());
    let mut rows = Vec::new();
    loop {
        match d.next_packet() {
            Ok(p) => rows.push((p.pts, p.duration, p.data.len())),
            Err(Error::Eof) => break,
            Err(e) => panic!("{e}"),
        }
    }
    (d, rows)
}

/// The header FFmpeg writes for 3 s of stereo 22050 Hz IMA ADPCM
/// (ffmpeg -c:a adpcm_ima_wav): 1024-byte blocks of 1017 samples; FFmpeg
/// reads two blocks a packet, 2034 samples.
#[test]
fn ima_adpcm_packets_are_timed_by_their_samples() {
    let (d, rows) = packets(wav(0x0011, 2, 22050, 22201, 1024, 4, 66, Some(66150)));
    assert_eq!(rows.len(), 33);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(*row, (Some(2034 * i as i64), Some(2034), 2048), "packet {i}");
    }
    assert_eq!(d.streams()[0].duration, Some(66150), "the fact count");
}

/// MS ADPCM, FFmpeg's header (ffmpeg -c:a adpcm_ms): 2 + (1024 - 7 * 2) *
/// 2 / 2 = 1012 samples a block, two blocks a packet.
#[test]
fn ms_adpcm_packets_are_timed_by_their_samples() {
    let (_, rows) = packets(wav(0x0002, 2, 22050, 22311, 1024, 4, 66, Some(66150)));
    assert_eq!(rows.len(), 33);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(*row, (Some(2024 * i as i64), Some(2024), 2048), "packet {i}");
    }
}

/// Yamaha ADPCM has an exact 4 bits a sample: a packet of n bytes holds
/// n * 8 / (4 * channels) samples, and FFmpeg's packets are about 100 ms.
#[test]
fn yamaha_adpcm_is_timed_by_its_exact_bits() {
    // mono 8000 Hz, block 1024: byte rate 4000; ff_pcm_default_packet_size:
    // 4 * 8000 bits/s / 8 / 10 / 1024 -> 1 block a packet.
    let (_, rows) = packets(wav(0x0020, 1, 8000, 4000, 1024, 4, 6, None));
    assert_eq!(rows.len(), 6);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(*row, (Some(2048 * i as i64), Some(2048), 1024), "packet {i}");
    }
}

/// Ulead DV audio (PAL, 8640-byte blocks of 1920 samples at 48 kHz): FFmpeg
/// reads two blocks a packet; how many samples a block holds is the
/// decoder's to say, so packets after the first are untimed (FFmpeg's
/// first is at 0). 1125 blocks run 45 s, past the 1024 blocks the old
/// packets held.
#[test]
fn dv_audio_packets_after_the_first_are_left_to_the_decoder() {
    let (d, rows) = packets(wav(0x0216, 2, 48000, 216_000, 8640, 16, 1125, None));
    assert_eq!(rows.len(), 563, "FFmpeg's 563 packets");
    assert_eq!(rows[0], (Some(0), None, 17280));
    assert!(rows[1..562].iter().all(|r| *r == (None, None, 17280)));
    assert_eq!(rows[562], (None, None, 8640), "the last, odd block");
    assert_eq!(d.duration_micros(), Some(45_000_000), "45 s by the byte rate");
}

/// ff_pcm_read_seek: a seek lands on the block the byte rate puts the
/// target in (rounding down), and the next packet carries that block's
/// time by the byte rate.
#[test]
fn seeks_land_where_ffmpeg_lands() {
    let reg = registry();
    // DV audio: 10.3 s -> byte 10.3 * 216000 = 2224800 -> block 257
    // (2220480), 2220480 / 216000 s = 10.28 s = 493440 samples.
    let file = wav(0x0216, 2, 48000, 216_000, 8640, 16, 1125, None);
    let mut d = open_wav_demuxer_with(Box::new(Cursor::new(file)), &reg).unwrap();
    assert_eq!(d.seek_to(0, 494_400).unwrap(), 493_440);
    let p = d.next_packet().unwrap();
    // Each block's bytes are its index, mod 256.
    assert_eq!((p.pts, p.data[0]), (Some(493_440), (257 % 256) as u8));
    assert_eq!(d.next_packet().unwrap().pts, None, "the next is the decoder's");
    // IMA ADPCM: 1.0 s -> 22201 bytes -> block 21 (21504) -> pts
    // 21504 * 22050 / 22201 = 21357.7, rounded to 21358.
    let file = wav(0x0011, 2, 22050, 22201, 1024, 4, 66, Some(66150));
    let mut d = open_wav_demuxer_with(Box::new(Cursor::new(file)), &reg).unwrap();
    assert_eq!(d.seek_to(0, 22050).unwrap(), 21358);
    let p = d.next_packet().unwrap();
    assert_eq!((p.pts, p.data[0]), (Some(21358), 21u8));
    assert_eq!(d.next_packet().unwrap().pts, Some(21358 + 2034));
}

/// The stream's bit rate is the header's byte rate, not block_align times
/// the sample rate.
#[test]
fn block_coded_bit_rate_is_the_byte_rate() {
    let reg = registry();
    let d = open_wav_demuxer_with(Box::new(Cursor::new(wav(0x0216, 2, 48000, 216_000, 8640, 16, 4, None))), &reg).unwrap();
    assert_eq!(d.streams()[0].params.bit_rate, Some(216_000 * 8));
}
