//! `rawvideo` codec: uncompressed pictures, one packet per frame.
//!
//! A packet holds the frame's image planes back to back, each plane's
//! rows tightly packed (no stride padding) — exactly the payload the
//! Y4M container stores after each `FRAME` marker and what other raw
//! carriers (`V_UNCOMPRESSED`, image-sequence sources) produce. The
//! plane geometry comes from the stream's `pixel_format` / `width` /
//! `height` ([`PixelFormat::plane_row_bytes`] /
//! [`PixelFormat::plane_dimensions`]).
//!
//! * **Decode**: split the packet into planes → [`VideoFrame`] with
//!   `stride == row bytes`.
//! * **Encode**: concatenate each image plane's visible rows (dropping
//!   any stride padding) → one packet.
//!
//! Palette formats (`Pal8`) are rejected: the palette is out-of-band
//! side data a raw payload has no room for.

use std::collections::VecDeque;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, Decoder, Encoder, Error,
    Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame, VideoPlane,
};

/// Codec id of uncompressed video.
pub const CODEC_ID: &str = "rawvideo";

/// Install the `rawvideo` decoder + encoder.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("rawvideo_sw")
        .with_lossless(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder),
    );
}

/// Validated picture geometry: the format, the picture size, and per
/// image plane `(row_bytes, rows)`.
#[derive(Debug, Clone)]
struct Layout {
    format: PixelFormat,
    size: (u32, u32),
    planes: Vec<(usize, usize)>,
    frame_bytes: usize,
}

impl Layout {
    fn from_params(params: &CodecParameters, who: &str) -> Result<Self> {
        let fmt = params
            .pixel_format
            .ok_or_else(|| Error::invalid(format!("rawvideo {who}: pixel_format is required")))?;
        let w = params
            .width
            .ok_or_else(|| Error::invalid(format!("rawvideo {who}: width is required")))?;
        let h = params
            .height
            .ok_or_else(|| Error::invalid(format!("rawvideo {who}: height is required")))?;
        if w == 0 || h == 0 {
            return Err(Error::invalid(format!(
                "rawvideo {who}: empty picture {w}x{h}"
            )));
        }
        if fmt == PixelFormat::Pal8 {
            return Err(Error::unsupported(format!(
                "rawvideo {who}: palette formats carry their palette out of band"
            )));
        }
        let mut planes = Vec::with_capacity(fmt.plane_count());
        let mut frame_bytes = 0usize;
        for p in 0..fmt.plane_count() {
            let row = fmt.plane_row_bytes(p, w);
            let dims = fmt.plane_dimensions(p, w, h);
            let (Some(row), Some((_, rows))) = (row, dims) else {
                return Err(Error::unsupported(format!(
                    "rawvideo {who}: no plane geometry for {fmt:?}"
                )));
            };
            let rows = rows as usize;
            frame_bytes = row
                .checked_mul(rows)
                .and_then(|n| frame_bytes.checked_add(n))
                .ok_or_else(|| Error::invalid(format!("rawvideo {who}: frame size overflow")))?;
            planes.push((row, rows));
        }
        Ok(Self {
            format: fmt,
            size: (w, h),
            planes,
            frame_bytes,
        })
    }
}

/// Build a `rawvideo` decoder for `params`.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(RawVideoDecoder {
        id: CodecId::new(CODEC_ID),
        layout: Layout::from_params(params, "decoder")?,
        pending: VecDeque::new(),
        eof: false,
    }))
}

/// Build a `rawvideo` encoder for `params`.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let layout = Layout::from_params(params, "encoder")?;
    let mut output = CodecParameters::video(CodecId::new(CODEC_ID));
    output.width = params.width;
    output.height = params.height;
    output.pixel_format = params.pixel_format;
    output.frame_rate = params.frame_rate;
    Ok(Box::new(RawVideoEncoder {
        layout,
        output,
        queue: VecDeque::new(),
    }))
}

struct RawVideoDecoder {
    id: CodecId,
    layout: Layout,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Decoder for RawVideoDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.len() != self.layout.frame_bytes {
            return Err(Error::invalid(format!(
                "rawvideo: packet holds {} bytes, a frame is {}",
                packet.data.len(),
                self.layout.frame_bytes
            )));
        }
        self.pending.push_back(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.pop_front() else {
            return Err(if self.eof {
                Error::Eof
            } else {
                Error::NeedMore
            });
        };
        let mut planes = Vec::with_capacity(self.layout.planes.len());
        let mut off = 0usize;
        for &(row, rows) in &self.layout.planes {
            let len = row * rows;
            planes.push(VideoPlane {
                stride: row,
                data: pkt.data[off..off + len].to_vec(),
            });
            off += len;
        }
        Ok(Frame::Video(VideoFrame {
            pts: pkt.pts,
            planes,
        }))
    }

    /// Every frame has the stream's geometry: a raw payload carries none.
    fn output_video_dimensions(&self) -> Option<(u32, u32)> {
        Some(self.layout.size)
    }

    fn output_pixel_format(&self) -> Option<PixelFormat> {
        Some(self.layout.format)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

struct RawVideoEncoder {
    layout: Layout,
    output: CodecParameters,
    queue: VecDeque<Packet>,
}

impl Encoder for RawVideoEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Video(v) = frame else {
            return Err(Error::invalid("rawvideo encoder requires video frames"));
        };
        let image = v.image_planes();
        if image.len() != self.layout.planes.len() {
            return Err(Error::invalid(format!(
                "rawvideo encoder: frame has {} planes, the format has {}",
                image.len(),
                self.layout.planes.len()
            )));
        }
        let mut data = Vec::with_capacity(self.layout.frame_bytes);
        for (plane, &(row, rows)) in image.iter().zip(&self.layout.planes) {
            if plane.stride < row || plane.data.len() < plane.stride * (rows - 1) + row {
                return Err(Error::invalid(
                    "rawvideo encoder: plane smaller than the picture geometry",
                ));
            }
            for r in 0..rows {
                let start = r * plane.stride;
                data.extend_from_slice(&plane.data[start..start + row]);
            }
        }
        // The frame pts is carried through unchanged; it counts in the
        // stream's time base, which for a constant-rate raw stream is
        // the frame period.
        let tb = self
            .output
            .frame_rate
            .filter(|r| r.num > 0 && r.den > 0)
            .map_or(TimeBase::new(1, 25), |r| TimeBase::new(r.den, r.num));
        let mut pkt = Packet::new(0, tb, data);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn params(fmt: PixelFormat, w: u32, h: u32) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(CODEC_ID));
        p.pixel_format = Some(fmt);
        p.width = Some(w);
        p.height = Some(h);
        p
    }

    #[test]
    fn decode_splits_yuv420p_planes() {
        let mut d = make_decoder(&params(PixelFormat::Yuv420P, 5, 3)).unwrap();
        // 5x3 luma + 3x2 per chroma plane.
        let data: Vec<u8> = (0..(15 + 6 + 6)).map(|i| i as u8).collect();
        let mut pkt = Packet::new(0, TimeBase::new(1, 25), data);
        pkt.pts = Some(4);
        d.send_packet(&pkt).unwrap();
        let Frame::Video(v) = d.receive_frame().unwrap() else {
            panic!("video frame expected");
        };
        assert_eq!(v.pts, Some(4));
        assert_eq!(v.planes.len(), 3);
        assert_eq!((v.planes[0].stride, v.planes[0].data.len()), (5, 15));
        assert_eq!((v.planes[1].stride, v.planes[1].data.len()), (3, 6));
        assert_eq!(v.planes[2].data[0], 21);
        assert!(matches!(d.receive_frame(), Err(Error::NeedMore)));
        d.flush().unwrap();
        assert!(matches!(d.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn decode_rejects_wrong_packet_size() {
        let mut d = make_decoder(&params(PixelFormat::Rgb24, 2, 2)).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), vec![0; 11]);
        assert!(d.send_packet(&pkt).is_err());
    }

    #[test]
    fn encode_drops_stride_padding_and_round_trips() {
        let p = params(PixelFormat::Gray8, 3, 2);
        let mut e = make_encoder(&p).unwrap();
        let frame = VideoFrame {
            pts: Some(9),
            planes: vec![VideoPlane {
                stride: 8,
                data: vec![1, 2, 3, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0],
            }],
        };
        e.send_frame(&Frame::Video(frame)).unwrap();
        let pkt = e.receive_packet().unwrap();
        assert_eq!(pkt.data, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(pkt.pts, Some(9));
        let mut d = make_decoder(&p).unwrap();
        d.send_packet(&pkt).unwrap();
        let Frame::Video(v) = d.receive_frame().unwrap() else {
            panic!()
        };
        assert_eq!(v.planes[0].data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn rejects_palette_and_missing_geometry() {
        assert!(make_decoder(&params(PixelFormat::Pal8, 2, 2)).is_err());
        let mut p = params(PixelFormat::Yuv420P, 2, 2);
        p.pixel_format = None;
        assert!(make_encoder(&p).is_err());
    }
}
