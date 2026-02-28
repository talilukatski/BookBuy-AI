import { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'

// Default JSON template shown in the textarea
const DEFAULT_JSON = `{
  "prompt": "",
  "address": "",
  "payment_token": "",
  "user_preferences": [],
  "disliked_titles": [],
  "already_read_titles": []
}`

function ChatInterface({ messages, onSendMessage }) {

  // Initialize the textarea with a JSON template
  const [inputValue, setInputValue] = useState(DEFAULT_JSON)

  // State for JSON validation errors
  const [jsonError, setJsonError] = useState('')

  const messagesEndRef = useRef(null)

  // Automatically scroll to the newest message
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = (e) => {

    e.preventDefault()
    setJsonError('')

    // 1. Try to parse the JSON from the textarea
    let payload

    try {
      payload = JSON.parse(inputValue)
    } catch (err) {
      setJsonError('Invalid JSON. Check commas, quotes, and brackets.')
      return
    }

    // 2. Validate required fields
    const missing = []

    if (!payload.prompt || String(payload.prompt).trim() === '')
      missing.push('prompt')

    if (!payload.address || String(payload.address).trim() === '')
      missing.push('address')

    if (!payload.payment_token || String(payload.payment_token).trim() === '')
      missing.push('payment_token')

    if (missing.length > 0) {
      setJsonError(`Missing required field(s): ${missing.join(', ')}`)
      return
    }

    // 3. Ensure optional fields are arrays
    const normalizeList = (v) => {
      if (Array.isArray(v)) return v
      if (v === null || v === undefined || v === '') return []
      return [String(v)]
    }

    payload.user_preferences = normalizeList(payload.user_preferences)
    payload.disliked_titles = normalizeList(payload.disliked_titles)
    payload.already_read_titles = normalizeList(payload.already_read_titles)

    // 4. Send the JSON payload to the parent component
    onSendMessage(payload)
  }

  // Prevent UI crash if objects are returned
  const safeRender = (content) => {

    if (typeof content === 'object' && content !== null) {
      return JSON.stringify(content);
    }

    return content;
  }

  return (
    <div className="chat-interface">

      <div className="messages-list">

        {messages.map((msg) => (
          <div key={msg.id} className={`message-container ${msg.sender}`}>

            <MessageBubble message={msg} />

            {/* Display reasoning steps returned by the agent */}
            {msg.sender === 'agent' && (msg.steps && msg.steps.length > 0) && (

              <div className="reasoning-steps" style={{
                fontSize: '0.85em',
                backgroundColor: '#f0fdf4',
                borderLeft: '3px solid #22c55e',
                padding: '10px',
                margin: '5px 20px',
                borderRadius: '5px',
                color: '#166534'
              }}>

                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '5px'
                }}>
                  <strong>Agent Actions:</strong>
                </div>

                {msg.steps && msg.steps.map((step, index) => (
                  <div key={index} style={{
                    marginBottom: '8px',
                    borderBottom: '1px solid #dcfce7',
                    paddingBottom: '5px'
                  }}>
                    <div><strong>Module:</strong> {step.module}</div>
                    <div><strong>Prompt:</strong> {safeRender(step.prompt)}</div>
                    <div><strong>Response:</strong> {safeRender(step.response)}</div>
                  </div>
                ))}

              </div>

            )}

          </div>
        ))}

        <div ref={messagesEndRef} />

      </div>

      <form className="input-area" onSubmit={handleSubmit}>

        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => {

            // Enter submits, Shift+Enter creates a new line
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }

          }}
          placeholder="Fill the JSON request here..."
          rows={10}
          style={{
            flex: 1,
            padding: '10px',
            borderRadius: '8px',
            border: jsonError ? '1px solid #ef4444' : '1px solid #ddd',
            resize: 'none',
            fontSize: '0.95rem',
            fontFamily: 'monospace'
          }}
        />

        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-end',
          gap: '8px'
        }}>

          {jsonError && (
            <div style={{
              color: '#b91c1c',
              background: '#fee2e2',
              border: '1px solid #fecaca',
              padding: '8px 10px',
              borderRadius: '8px',
              maxWidth: '320px'
            }}>
              {jsonError}
            </div>
          )}

          <button type="submit" style={{
            padding: '10px 20px',
            backgroundColor: '#2563eb',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: '600',
            height: 'fit-content'
          }}>
            Run Agent
          </button>

        </div>

      </form>

    </div>
  )
}

export default ChatInterface
