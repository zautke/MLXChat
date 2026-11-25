import Foundation
import Observation
import UIKit
import MLXChatCore
import MLXChatEngineMLX
import os

/// Watches memory pressure and surfaces available headroom.
/// On warning: drop the MLX buffer cache first; escalate after repeated warnings.
@MainActor
@Observable
final class MemoryGovernor {
    private(set) var warningCount = 0
    private(set) var availableBytes: UInt64 = 0

    private var observer: (any NSObjectProtocol)?
    private var timer: Timer?
    private let logger = Logger(subsystem: "com.zautke.mlxchat", category: "memory")

    func start() {
        observer = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.handleWarning() }
        }
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
        refresh()
    }

    func stop() {
        if let observer { NotificationCenter.default.removeObserver(observer) }
        timer?.invalidate()
        observer = nil
        timer = nil
    }

    private func refresh() {
        availableBytes = UInt64(os_proc_available_memory())
    }

    private func handleWarning() {
        warningCount += 1
        logger.warning("Memory warning #\(self.warningCount), available: \(self.availableBytes)")
        MLXEngine.clearGPUCache()
        if warningCount >= 3 {
            NotificationCenter.default.post(name: .mlxChatNeedsSmallerModel, object: nil)
        }
    }
}

extension Notification.Name {
    static let mlxChatNeedsSmallerModel = Notification.Name("mlxChatNeedsSmallerModel")
}
