import Foundation

/// Backend-agnostic dictation interface. The UI talks only to this.
@MainActor
protocol Dictation: AnyObject, ObservableObject {
    var transcript: String { get }
    var isListening: Bool { get }
    func start() async throws
    func stop() async
}
