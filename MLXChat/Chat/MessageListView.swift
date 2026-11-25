import SwiftUI
import MLXChatCore

struct MessageListView: View {
    let messages: [ChatMessage]
    let streamingText: String

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(messages.filter { $0.role != .system }) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }
                    if !streamingText.isEmpty {
                        MessageBubble(message: ChatMessage(role: .assistant, content: streamingText + " ▌"))
                            .id("streaming")
                    }
                }
                .padding()
            }
            .onChange(of: messages.count) {
                withAnimation { proxy.scrollTo(messages.last?.id, anchor: .bottom) }
            }
            .onChange(of: streamingText) {
                proxy.scrollTo("streaming", anchor: .bottom)
            }
        }
    }
}
