import SwiftUI

/// Mic button that dictates into the composer. Uses SpeechAnalyzer on iOS 26+,
/// SFSpeechRecognizer below.
struct VoiceInputButton: View {
    @Binding var text: String
    @State private var controller = DictationController()

    var body: some View {
        Button {
            Task { await controller.toggle() }
        } label: {
            Image(systemName: controller.isRecording ? "mic.fill" : "mic")
                .font(.title2)
                .foregroundStyle(controller.isRecording ? .red : .primary)
        }
        .accessibilityLabel(controller.isRecording ? "Stop dictation" : "Start dictation")
        .onChange(of: controller.transcript) { _, newValue in
            if controller.isRecording { text = newValue }
        }
        .onDisappear {
            Task { await controller.stop() }
        }
        .alert("Dictation", isPresented: Binding(
            get: { controller.errorMessage != nil },
            set: { if !$0 { controller.errorMessage = nil } }
        )) {
            Button("OK") { controller.errorMessage = nil }
        } message: {
            Text(controller.errorMessage ?? "")
        }
    }
}

/// Drives a Dictation backend and republishes its transcript for SwiftUI.
@MainActor
@Observable
final class DictationController {
    private(set) var transcript = ""
    private(set) var isRecording = false
    var errorMessage: String?

    private let impl: any Dictation
    private var pollTask: Task<Void, Never>?

    init() {
        if #available(iOS 26.0, *) {
            impl = SpeechAnalyzerDictation()
        } else {
            impl = SFSpeechDictation()
        }
    }

    func toggle() async {
        if isRecording {
            await stop()
        } else {
            await start()
        }
    }

    func start() async {
        do {
            try await impl.start()
            isRecording = true
            pollTask = Task { @MainActor [weak self] in
                while let self, self.isRecording, !Task.isCancelled {
                    self.transcript = self.impl.transcript
                    try? await Task.sleep(for: .milliseconds(150))
                }
            }
        } catch {
            errorMessage = error.localizedDescription
            isRecording = false
        }
    }

    func stop() async {
        pollTask?.cancel()
        pollTask = nil
        await impl.stop()
        transcript = impl.transcript
        isRecording = false
    }
}
