import AVFoundation

/// Minimal PCM buffer converter used to feed the speech analyzer from the mic tap.
final class BufferConverter {
    static let shared = BufferConverter()

    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?

    func convert(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        if converter == nil || sourceFormat != buffer.format {
            converter = AVAudioConverter(from: buffer.format, to: format)
            sourceFormat = buffer.format
        }
        guard let converter else {
            throw NSError(domain: "BufferConverter", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Could not create audio converter"])
        }
        let ratio = format.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1024
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
            throw NSError(domain: "BufferConverter", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Could not allocate output buffer"])
        }
        var error: NSError?
        var consumed = false
        converter.convert(to: output, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return buffer
        }
        if let error { throw error }
        return output
    }
}
