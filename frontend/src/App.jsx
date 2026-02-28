import { useState } from 'react'
import './App.css'
import ChatInterface from './components/ChatInterface'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

function App() {
  const [messages, setMessages] = useState([
  {
    id: 1,
    sender: 'agent',
    text: `Hi! I'm your Book Buying Agent 📚

    Tell me what book you want and I'll do the rest:
    I'll find the best match, compare prices across bookstores, choose the best deal, and buy it for you automatically.
    
    Fill the request below and click "Run Agent".`
  }
  ])

  // This function now receives a full JSON payload instead of plain text
  const handleSendMessage = async (payload) => {

    // 1. Add the user message to the chat (display the JSON sent)
    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: JSON.stringify(payload, null, 2)
    }

    setMessages(prev => [...prev, userMsg])

    // 2. Add a temporary agent message that will be updated when the API responds
    const agentMsgId = Date.now() + 1;

    const initialAgentMsg = {
      id: agentMsgId,
      sender: 'agent',
      text: 'Thinking...',
      steps: []
    }

    setMessages(prev => [...prev, initialAgentMsg])

    try {

      // 3. Send the entire JSON payload directly to the backend API
      const response = await fetch(`${API_BASE_URL}/api/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json();

      // 4. Update the temporary agent message with the response
      setMessages(prev => prev.map(msg => {

        if (msg.id === agentMsgId) {

          if (data.status === 'error') {
            return { ...msg, text: `Error: ${data.error}` };
          }

          return { ...msg, text: data.response, steps: data.steps };
        }

        return msg;

      }));

    } catch (error) {

      console.error('Error fetching from agent:', error)

      // 5. Handle connection errors
      setMessages(prev => prev.map(msg => {

        if (msg.id === agentMsgId) {
          return { ...msg, text: "Sorry, I'm having trouble connecting to the server." };
        }

        return msg;

      }));
    }
  }

  return (
    <div className="app-container">

      <header className="app-header">
        <h1>Book Buying Agent</h1>
      </header>

      <main className="main-content">
        <ChatInterface
          messages={messages}
          onSendMessage={handleSendMessage}
        />
      </main>

    </div>
  )
}

export default App
