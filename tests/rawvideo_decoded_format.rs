//! The `rawvideo` decoder reports the stream geometry, which every frame
//! has (a raw payload carries none, so the size cannot change mid-stream):
//! odd sizes and 10-bit samples included.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};

#[test]
fn reports_the_geometry_every_frame_has() {
    let mut params = CodecParameters::video(CodecId::new("rawvideo"));
    params.pixel_format = Some(PixelFormat::Yuv420P10Le);
    params.width = Some(33);
    params.height = Some(17);
    let mut d = oxideav_basic::rawvideo::make_decoder(&params).unwrap();
    // 33x17 luma and 17x9 per chroma plane, two bytes per sample.
    let data = vec![0u8; (33 * 17 + 2 * 17 * 9) * 2];
    d.send_packet(&Packet::new(0, TimeBase::new(1, 25), data))
        .unwrap();
    let Frame::Video(v) = d.receive_frame().unwrap() else {
        panic!("video frame expected");
    };
    assert_eq!(d.output_video_dimensions(), Some((33, 17)));
    assert_eq!(d.output_pixel_format(), Some(PixelFormat::Yuv420P10Le));
    let geometry: Vec<_> = v
        .image_planes()
        .iter()
        .map(|p| (p.stride, p.data.len() / p.stride))
        .collect();
    assert_eq!(geometry, [(66, 17), (34, 9), (34, 9)]);
}
