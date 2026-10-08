//! PCM codec: uncompressed interleaved linear PCM.
//!
//! A PCM "codec" is trivial — the packet payload *is* the sample data. We
//! still funnel it through [`Decoder`] / [`Encoder`] so that pipelines treat it
//! uniformly.
//!
//! Codec IDs:
//! - `pcm_u8`   — unsigned 8-bit
//! - `pcm_s16le` — signed 16-bit little-endian
//! - `pcm_s24le` — signed 24-bit little-endian, packed
//! - `pcm_s32le` — signed 32-bit little-endian
//! - `pcm_f32le` — 32-bit IEEE float little-endian
//! - `pcm_f64le` — 64-bit IEEE float little-endian
//!
//! Decoded only (containers such as MOV, CAF, MXF, OMA and AIFF carry
//! them), to the formats above:
//! - `pcm_s16be`, `pcm_s24be`, `pcm_s32be`, `pcm_f32be`, `pcm_f64be` —
//!   big-endian
//! - `pcm_u16le`/`be`, `pcm_u24le`/`be`, `pcm_u32le`/`be` — unsigned
//!   (offset binary), to signed
//! - `pcm_s64le`/`be` — signed 64-bit, to 64-bit float (full scale ±1.0)
//! - `pcm_s24daud` — D-Cinema 24-bit words, to signed 16-bit
//!
//! Asterisk-style signed-linear aliases:
//! - `slin`, `slin8`, `slin16`, `slin24`, `slin32`, `slin44`, `slin48`,
//!   `slin96`, `slin192` — all map onto the `pcm_s16le` implementation.
//!   The trailing digits only indicate the implied sample rate of the
//!   surrounding headerless `.sln*` container (see `slin.rs`); as a codec
//!   they are indistinguishable from `pcm_s16le`.

use oxideav_core::{
    AudioFormat, AudioFrame, CodecCapabilities, CodecId, CodecParameters, CodecTag, Error, Frame,
    MediaType, Packet, ProbeContext, Result, SampleFormat, TimeBase,
};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder, Encoder};

pub fn register(reg: &mut CodecRegistry) {
    // WAVEFORMATEX tags handled by this crate:
    //   0x0001 WAVE_FORMAT_PCM — integer PCM, bit-depth disambiguation
    //     by bits_per_sample.
    //   0x0003 WAVE_FORMAT_IEEE_FLOAT — float PCM, same idea.
    let wf_int = CodecTag::wave_format(0x0001);
    let wf_flt = CodecTag::wave_format(0x0003);
    for (id, bits, tag, probe) in [
        (
            "pcm_u8",
            8u16,
            Some(&wf_int),
            probe_pcm_u8 as oxideav_core::ProbeFn,
        ),
        ("pcm_s8", 8, None, probe_pcm_s8 as oxideav_core::ProbeFn),
        (
            "pcm_s16le",
            16,
            Some(&wf_int),
            probe_pcm_s16le as oxideav_core::ProbeFn,
        ),
        (
            "pcm_s24le",
            24,
            Some(&wf_int),
            probe_pcm_s24le as oxideav_core::ProbeFn,
        ),
        (
            "pcm_s32le",
            32,
            Some(&wf_int),
            probe_pcm_s32le as oxideav_core::ProbeFn,
        ),
        (
            "pcm_f32le",
            32,
            Some(&wf_flt),
            probe_pcm_f32le as oxideav_core::ProbeFn,
        ),
        (
            "pcm_f64le",
            64,
            Some(&wf_flt),
            probe_pcm_f64le as oxideav_core::ProbeFn,
        ),
    ] {
        let _ = bits;
        // The encoder writes its one interleaved layout; pipelines
        // convert other layouts (planar, other widths) in front of it.
        let caps = CodecCapabilities::audio(format!("{id}_sw"))
            .with_lossless(true)
            .with_intra_only(true)
            .with_sample_formats(sample_format_for(&CodecId::new(id)).into_iter().collect());
        let mut info = CodecInfo::new(CodecId::new(id))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder);
        if let Some(t) = tag {
            info = info.probe(probe).tag(t.clone());
        }
        reg.register(info);
    }

    for id in SLIN_ALIASES {
        let caps = CodecCapabilities::audio(format!("{id}_sw"))
            .with_lossless(true)
            .with_intra_only(true)
            .with_sample_formats(vec![SampleFormat::S16]);
        // Same factories as pcm_s16le — `sample_format_for` maps all the
        // slin aliases to SampleFormat::S16 below. No WAVEFORMATEX claim.
        reg.register(
            CodecInfo::new(CodecId::new(*id))
                .capabilities(caps)
                .decoder(make_decoder)
                .encoder(make_encoder),
        );
    }

    for &(id, format, _) in DECODE_ONLY {
        let caps = CodecCapabilities::audio(format!("{id}_sw"))
            .with_lossless(true)
            .with_intra_only(true)
            .with_sample_formats(vec![format]);
        reg.register(CodecInfo::new(CodecId::new(id)).capabilities(caps).decoder(make_decoder));
    }
}

/// How a PCM id's bytes become those of the format it decodes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wire {
    /// The decoded format's own (little-endian) bytes.
    Native,
    /// The decoded format's samples, big-endian.
    BigEndian,
    /// Offset-binary samples of the decoded format's width.
    Unsigned { big_endian: bool },
    /// Signed 64-bit integers, decoded to 64-bit float.
    S64 { big_endian: bool },
    /// D-Cinema 24-bit big-endian words: 4 flag bits, 16 audio bits
    /// least-significant first, 4 flag bits; decoded to signed 16-bit.
    Daud,
}

/// The PCM ids decoded but not encoded here, with the format each decodes
/// to and its wire layout.
const DECODE_ONLY: &[(&str, SampleFormat, Wire)] = &[
    ("pcm_s16be", SampleFormat::S16, Wire::BigEndian),
    ("pcm_s24be", SampleFormat::S24, Wire::BigEndian),
    ("pcm_s32be", SampleFormat::S32, Wire::BigEndian),
    ("pcm_f32be", SampleFormat::F32, Wire::BigEndian),
    ("pcm_f64be", SampleFormat::F64, Wire::BigEndian),
    ("pcm_u16le", SampleFormat::S16, Wire::Unsigned { big_endian: false }),
    ("pcm_u16be", SampleFormat::S16, Wire::Unsigned { big_endian: true }),
    ("pcm_u24le", SampleFormat::S24, Wire::Unsigned { big_endian: false }),
    ("pcm_u24be", SampleFormat::S24, Wire::Unsigned { big_endian: true }),
    ("pcm_u32le", SampleFormat::S32, Wire::Unsigned { big_endian: false }),
    ("pcm_u32be", SampleFormat::S32, Wire::Unsigned { big_endian: true }),
    ("pcm_s64le", SampleFormat::F64, Wire::S64 { big_endian: false }),
    ("pcm_s64be", SampleFormat::F64, Wire::S64 { big_endian: true }),
    ("pcm_s24daud", SampleFormat::S16, Wire::Daud),
];

/// The format a PCM id decodes to and its wire layout.
fn decode_layout(id: &CodecId) -> Option<(SampleFormat, Wire)> {
    if let Some(format) = sample_format_for(id) {
        return Some((format, Wire::Native));
    }
    DECODE_ONLY.iter().find(|(name, ..)| *name == id.as_str()).map(|&(_, format, wire)| (format, wire))
}

impl Wire {
    /// Bytes a sample takes on the wire.
    fn width(self, decoded: SampleFormat) -> usize {
        match self {
            Wire::S64 { .. } => 8,
            Wire::Daud => 3,
            _ => decoded.bytes_per_sample(),
        }
    }

    /// Converts whole samples in place to the decoded format's bytes.
    fn decode(self, decoded: SampleFormat, data: &mut Vec<u8>) {
        let width = self.width(decoded);
        match self {
            Wire::Native => {}
            Wire::BigEndian => data.chunks_exact_mut(width).for_each(<[u8]>::reverse),
            Wire::Unsigned { big_endian } => {
                for sample in data.chunks_exact_mut(width) {
                    if big_endian {
                        sample.reverse();
                    }
                    sample[width - 1] ^= 0x80;
                }
            }
            Wire::S64 { big_endian } => {
                for sample in data.chunks_exact_mut(8) {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(sample);
                    let v = if big_endian { i64::from_be_bytes(bytes) } else { i64::from_le_bytes(bytes) };
                    sample.copy_from_slice(&(v as f64 / 9_223_372_036_854_775_808.0).to_le_bytes());
                }
            }
            Wire::Daud => {
                // Each 2-byte result lands before the 3-byte word it came
                // from, at or after the end of the ones already read.
                let words = data.len() / 3;
                for i in 0..words {
                    let word = u32::from(data[3 * i]) << 16 | u32::from(data[3 * i + 1]) << 8 | u32::from(data[3 * i + 2]);
                    let sample = ((word >> 4) as u16).reverse_bits().to_le_bytes();
                    data[2 * i..2 * i + 2].copy_from_slice(&sample);
                }
                data.truncate(2 * words);
            }
        }
    }
}

// --- Per-variant PCM probes -----------------------------------------------
// Each probe is selected by the WAVEFORMATEX tag at the call site; here we
// only need to disambiguate by bit depth. `bits_per_sample = None` returns
// 0.0 so the registry resolves nothing — the AVI demuxer then falls back
// to its static table which picks a sensible default.

fn match_bits(ctx: &ProbeContext, expected: u16) -> f32 {
    match ctx.bits_per_sample {
        Some(b) if b == expected => 1.0,
        _ => 0.0,
    }
}

fn probe_pcm_u8(ctx: &ProbeContext) -> f32 {
    match_bits(ctx, 8)
}
fn probe_pcm_s8(_ctx: &ProbeContext) -> f32 {
    // pcm_s8 has no canonical WAVEFORMATEX mapping; probe is unused but
    // required by the ProbeFn type alias in the register() table.
    0.0
}
fn probe_pcm_s16le(ctx: &ProbeContext) -> f32 {
    match ctx.bits_per_sample {
        // bits_per_sample == 0 occasionally means "unspecified" in the
        // wild — WAV files tagged as PCM that omit the depth. Treat
        // that as s16le (the most common default) with middling
        // confidence so a specific claimant (e.g. pcm_s24le with
        // bits=24) still wins if the depth is actually set.
        Some(0) => 0.5,
        Some(16) => 1.0,
        _ => 0.0,
    }
}
fn probe_pcm_s24le(ctx: &ProbeContext) -> f32 {
    match_bits(ctx, 24)
}
fn probe_pcm_s32le(ctx: &ProbeContext) -> f32 {
    match_bits(ctx, 32)
}
fn probe_pcm_f32le(ctx: &ProbeContext) -> f32 {
    match ctx.bits_per_sample {
        // IEEE float defaults to f32 when depth unspecified.
        None | Some(0) => 0.5,
        Some(32) => 1.0,
        _ => 0.0,
    }
}
fn probe_pcm_f64le(ctx: &ProbeContext) -> f32 {
    match_bits(ctx, 64)
}

/// Asterisk "signed linear" codec-id aliases. All are S16LE; the trailing
/// digits only matter at the container layer (see `slin.rs`).
pub(crate) const SLIN_ALIASES: &[&str] = &[
    "slin", "slin8", "slin16", "slin24", "slin32", "slin44", "slin48", "slin96", "slin192",
];

/// Return the [`SampleFormat`] implied by a PCM codec ID.
///
/// Also accepts the Asterisk `slin*` aliases, all of which describe 16-bit
/// signed linear PCM.
pub fn sample_format_for(id: &CodecId) -> Option<SampleFormat> {
    let s = id.as_str();
    Some(match s {
        "pcm_u8" => SampleFormat::U8,
        "pcm_s8" => SampleFormat::S8,
        "pcm_s16le" => SampleFormat::S16,
        "pcm_s24le" => SampleFormat::S24,
        "pcm_s32le" => SampleFormat::S32,
        "pcm_f32le" => SampleFormat::F32,
        "pcm_f64le" => SampleFormat::F64,
        _ if SLIN_ALIASES.contains(&s) => SampleFormat::S16,
        _ => return None,
    })
}

/// Return the canonical PCM codec ID for a [`SampleFormat`]. Planar formats
/// have no direct PCM codec — the caller must convert to interleaved first.
pub fn codec_id_for(fmt: SampleFormat) -> Option<CodecId> {
    Some(CodecId::new(match fmt {
        SampleFormat::U8 => "pcm_u8",
        SampleFormat::S8 => "pcm_s8",
        SampleFormat::S16 => "pcm_s16le",
        SampleFormat::S24 => "pcm_s24le",
        SampleFormat::S32 => "pcm_s32le",
        SampleFormat::F32 => "pcm_f32le",
        SampleFormat::F64 => "pcm_f64le",
        _ => return None,
    }))
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let (format, wire) = decode_layout(&params.codec_id)
        .ok_or_else(|| Error::CodecNotFound(params.codec_id.to_string()))?;
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("PCM decoder requires channels"))?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("PCM decoder requires sample_rate"))?;
    Ok(Box::new(PcmDecoder {
        id: params.codec_id.clone(),
        format,
        wire,
        channels,
        sample_rate,
        pending: None,
        eof: false,
    }))
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let format = sample_format_for(&params.codec_id)
        .ok_or_else(|| Error::CodecNotFound(params.codec_id.to_string()))?;
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("PCM encoder requires channels"))?;
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("PCM encoder requires sample_rate"))?;
    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(format);
    Ok(Box::new(PcmEncoder {
        format,
        channels,
        sample_rate,
        output,
        queue: std::collections::VecDeque::new(),
    }))
}

struct PcmDecoder {
    id: CodecId,
    format: SampleFormat,
    wire: Wire,
    channels: u16,
    sample_rate: u32,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for PcmDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "PCM decoder already has a buffered packet; call receive_frame first",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(mut pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let block = self.wire.width(self.format) * self.channels as usize;
        if block == 0 || pkt.data.len() % block != 0 {
            return Err(Error::invalid("PCM packet size not a multiple of block"));
        }
        let samples = (pkt.data.len() / block) as u32;
        self.wire.decode(self.format, &mut pkt.data);
        Ok(Frame::Audio(AudioFrame {
            samples,
            pts: pkt.pts,
            data: vec![pkt.data],
        }))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn output_audio_format(&self) -> Option<AudioFormat> {
        Some(AudioFormat { sample_format: self.format, sample_rate: self.sample_rate, channels: self.channels })
    }
}

struct PcmEncoder {
    format: SampleFormat,
    channels: u16,
    sample_rate: u32,
    output: CodecParameters,
    queue: std::collections::VecDeque<Packet>,
}

impl Encoder for PcmEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Audio(a) = frame else {
            return Err(Error::invalid("PCM encoder requires audio frames"));
        };
        // Per-frame format/channels/sample_rate are no longer carried on
        // AudioFrame — the encoder enforces its configured layout via the
        // stored fields and the byte-count check below.
        if self.format.is_planar() {
            return Err(Error::unsupported(
                "PCM encoder takes interleaved input; convert planar → interleaved first",
            ));
        }
        let data = a
            .data
            .first()
            .ok_or_else(|| Error::invalid("empty audio frame"))?
            .clone();
        let bps = self.format.bytes_per_sample() * self.channels as usize;
        let expected = bps * a.samples as usize;
        if data.len() != expected {
            return Err(Error::invalid("audio frame data length mismatch"));
        }
        let mut pkt = Packet::new(
            0,
            oxideav_core::TimeBase::new(1, self.sample_rate as i64),
            data,
        );
        pkt.pts = a.pts;
        pkt.dts = a.pts;
        pkt.duration = Some(a.samples as i64);
        pkt.flags.keyframe = true;
        self.queue.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.queue.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Helper to build codec parameters for a PCM stream.
pub fn params(format: SampleFormat, channels: u16, sample_rate: u32) -> Result<CodecParameters> {
    let codec_id = codec_id_for(format)
        .ok_or_else(|| Error::unsupported(format!("no PCM codec for {:?}", format)))?;
    let mut p = CodecParameters::audio(codec_id);
    p.sample_format = Some(format);
    p.channels = Some(channels);
    p.sample_rate = Some(sample_rate);
    p.bit_rate =
        Some((format.bytes_per_sample() as u64) * 8 * (channels as u64) * (sample_rate as u64));
    Ok(p)
}

/// Default time base for a PCM audio stream: 1 / sample_rate.
pub fn time_base_for(sample_rate: u32) -> TimeBase {
    TimeBase::new(1, sample_rate as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoders_declare_their_interleaved_layout() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        for (id, fmt) in [
            ("pcm_s16le", SampleFormat::S16),
            ("pcm_f32le", SampleFormat::F32),
            ("pcm_u8", SampleFormat::U8),
        ] {
            let imp = &reg.implementations(&CodecId::new(id))[0];
            assert_eq!(imp.caps.accepted_sample_formats, vec![fmt], "{id}");
        }
    }
}
