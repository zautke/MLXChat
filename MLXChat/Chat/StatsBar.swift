import SwiftUI
import MLXChatCore

struct StatsBar: View {
    let stats: EngineStats

    var body: some View {
        HStack(spacing: 16) {
            Label("\(String(format: "%.1f", stats.completionTokensPerSecond)) tok/s",
                  systemImage: "speedometer")
            Label("\(stats.completionTokens) tok", systemImage: "text.word.spacing")
            Label(Self.memoryString(stats.peakResidentMemoryBytes), systemImage: "memorychip")
            Spacer()
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
        .padding(.horizontal)
        .padding(.vertical, 4)
    }

    private static func memoryString(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
}
