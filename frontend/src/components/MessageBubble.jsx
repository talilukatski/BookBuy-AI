/**
 * MessageBubble Component
 * Simple component to display the text of a message.
 */
function MessageBubble({ message }) {

    // Convert objects to formatted JSON if needed
    const safeText =
        typeof message.text === "object"
            ? JSON.stringify(message.text, null, 2)
            : message.text;

    return (
        <div className={`message-bubble ${message.sender}`}>

            {/* Display message content while preserving line breaks */}
            <div
                className="message-text"
                style={{ whiteSpace: "pre-wrap" }}
            >
                {safeText}
            </div>

        </div>
    );
}

export default MessageBubble;