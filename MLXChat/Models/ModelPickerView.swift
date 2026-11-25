import SwiftUI
import MLXChatCore

struct ModelPickerView: View {
    let store: EngineStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(ModelCatalog.all) { model in
                        Button {
                            Task {
                                await store.switchModel(to: model)
                                dismiss()
                            }
                        } label: {
                            ModelRow(
                                descriptor: model,
                                isSelected: model == store.selectedModel
                            )
                        }
                        .buttonStyle(.plain)
                    }
                } header: {
                    Text("Switching models unloads the current one and downloads the new one if needed. Already-cached models load in seconds.")
                }
            }
            .navigationTitle("Models")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

private struct ModelRow: View {
    let descriptor: ModelDescriptor
    let isSelected: Bool

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(descriptor.displayName)
                    .font(.body)
                HStack(spacing: 8) {
                    Text(sizeString)
                    Text("·")
                    Text(descriptor.backend == .mlx ? "MLX" : "GGUF")
                    if descriptor.supportsThinking {
                        Text("·")
                        Text("thinking")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer()
            if isSelected {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.tint)
            }
        }
        .padding(.vertical, 4)
    }

    private var sizeString: String {
        ByteCountFormatter.string(fromByteCount: Int64(descriptor.approximateBytesOnDisk), countStyle: .file)
    }
}
