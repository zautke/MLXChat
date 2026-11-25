import Foundation
import Combine
import AVFoundation
import Speech

/// On-device dictation via SpeechAnalyzer + SpeechTranscriber (iPadOS/iOS 26+).
/// No speech-recognition permission needed — only the microphone.
@available(iOS 26.0, *)
@MainActor
final class SpeechAnalyzerDictation: Dictation {
    @Published private(set) var transcript = ""
    @Published private(set) var isListening = false

    private var analyzer: SpeechAnalyzer?
    private var transcriber: SpeechTranscriber?
    private var inputBuilder: AsyncStream<AnalyzerInput>.Continuation?
    private var format: AVAudioFormat?
    private let audioEngine = AVAudioEngine()
    private var resultsTask: Task<Void, Never>?

    func start() async throws {
        guard await AVAudioApplication.requestRecordPermission() else {
            throw NSError(domain: "Dictation", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Microphone access denied"])
        }

        let transcriber = SpeechTranscriber(
            locale: .current,
            preset: .progressiveTranscription
        )
        try await ensureModel(for: transcriber)

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        self.analyzer = analyzer
        self.transcriber = transcriber
        self.format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])

        let (stream, builder) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputBuilder = builder

        resultsTask = Task { @MainActor [weak self] in
            do {
                for try await result in transcriber.results {
                    self?.transcript = String(result.text.characters)
                }
            } catch {
                // Stream ended or was cancelled; nothing to recover here.
            }
        }

        try await analyzer.start(inputSequence: stream)
        try startMic()
        isListening = true
    }

    func stop() async {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        inputBuilder?.finish()
        resultsTask?.cancel()
        if let analyzer {
            try? await analyzer.finalizeAndFinishThroughEndOfInput()
        }
        analyzer = nil
        transcriber = nil
        inputBuilder = nil
        isListening = false
    }

    private func startMic() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: [.allowBluetooth])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let inputNode = audioEngine.inputNode
        let nativeFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: nativeFormat) { [weak self] buffer, _ in
            guard let self, let format = self.format, let builder = self.inputBuilder else { return }
            if let converted = try? BufferConverter.shared.convert(buffer, to: format) {
                builder.yield(AnalyzerInput(buffer: converted))
            }
        }
        audioEngine.prepare()
        try audioEngine.start()
    }

    private func ensureModel(for transcriber: SpeechTranscriber) async throws {
        let locale = Locale.current
        let supported = await SpeechTranscriber.supportedLocales
        guard supported.contains(where: { $0.identifier == locale.identifier }) else {
            throw NSError(domain: "Dictation", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Locale \(locale.identifier) is not supported for on-device transcription"])
        }
        let installed = await SpeechTranscriber.installedLocales
        if installed.contains(where: { $0.identifier == locale.identifier }) { return }
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            try await request.downloadAndInstall()
        }
    }
}
