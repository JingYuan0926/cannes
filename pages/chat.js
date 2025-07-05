import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function Chat() {
  const [userInput, setUserInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [sampleDataReady, setSampleDataReady] = useState(false);
  const [conversation, setConversation] = useState([]); // [{role: 'user'|'assistant', content: string}]
  const [sessionId, setSessionId] = useState(''); // Store sessionId for follow-up requests

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim() && files.length === 0) {
      setConversation((prev) => [...prev, { role: 'assistant', content: 'Please enter your question or upload at least one file.' }]);
      return;
    }
    setLoading(true);
    try {
      let result;
      if (files.length > 0 && !sampleDataReady) {
        // Upload files and get sample data ready
        const formData = new FormData();
        formData.append('prompt', userInput);
        files.forEach((file) => {
          formData.append('files', file);
        });
        
        const res = await fetch('/api/chatbot', {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) throw new Error(`Error: ${res.status}`);
        result = await res.json();
        setSampleDataReady(true);
        setFiles([]); // Clear files after upload
        setConversation([]); // Start new conversation after sample data
        setSessionId(result.sessionId); // Store the sessionId
        setConversation((prev) => [...prev, { role: 'assistant', content: result.message || 'Analysis Finished!' }]);
        setUserInput('');
        setLoading(false);
        return;
      } else {
        // Continue conversation, send user message and sampleDataReady flag
        const newConversation = [...conversation, { role: 'user', content: userInput }];
        setConversation(newConversation); // Optimistically add user message
        
        // Log the values for debugging
        console.log('Sending sessionId:', sessionId);
        console.log('sampleDataReady:', sampleDataReady);
        
        const res = await fetch('/api/chatbot', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            prompt: userInput,
            sampleDataReady: true,
            conversation: newConversation,
            sessionId: sessionId
          }),
        });
        if (!res.ok) throw new Error(`Error: ${res.status}`);
        result = await res.json();
        setConversation((prev) => [...prev, { role: 'assistant', content: result.message }]);
        setUserInput('');
      }
    } catch (err) {
      console.error('Error:', err);
      setConversation((prev) => [...prev, { role: 'assistant', content: 'Something went wrong.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px', color: '#ddd', backgroundColor: '#121212' }}>
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
        <textarea
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          rows="4"
          placeholder={sampleDataReady ? "Ask a data question..." : "Ask a data question or upload sample data..."}
          style={{
            width: '100%',
            padding: '10px',
            borderRadius: '8px',
            border: '1px solid #333',
            backgroundColor: '#1e1e1e',
            color: '#eee',
            resize: 'vertical',
          }}
        />
        <input
          type="file"
          accept="application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          multiple
          onChange={e => setFiles(Array.from(e.target.files))}
          style={{ marginTop: '10px', marginBottom: '10px', color: '#eee' }}
          disabled={sampleDataReady}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            marginTop: '10px',
            padding: '10px 20px',
            backgroundColor: '#3366cc',
            color: '#fff',
            border: 'none',
            borderRadius: '6px',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Analyzing...' : sampleDataReady ? 'Send' : 'Run Analysis'}
        </button>
      </form>
      {conversation.length > 0 && (
        <div style={{ backgroundColor: '#1a1a1a', padding: '15px', borderRadius: '8px', fontSize: '14px', marginBottom: '20px' }}>
          {conversation.map((msg, idx) => (
            <div key={idx} style={{ marginBottom: '10px' }}>
              <b style={{ color: msg.role === 'user' ? '#6cf' : '#9f6' }}>{msg.role === 'user' ? 'You' : 'AI'}:</b>{' '}
              {msg.role === 'assistant'
                ? (
                    <div className="chat-markdown-response">
                      <div className="chat-markdown">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    </div>
                  )
                : msg.content}
            </div>
          ))}
        </div>
      )}
      <style jsx>{`
        .chat-markdown-response {
          margin-top: 8px;
          margin-bottom: 8px;
          white-space: normal;
        }
        .chat-markdown h1 {
          font-size: 2rem;
          font-weight: bold;
          text-decoration: underline;
          margin: 1.2em 0 0.6em 0;
          color: #7ecfff;
        }
        .chat-markdown h2 {
          font-size: 1.5rem;
          font-weight: bold;
          text-decoration: underline;
          margin: 1em 0 0.5em 0;
          color: #7ecfff;
        }
        .chat-markdown h3 {
          font-size: 1.2rem;
          font-weight: bold;
          margin: 0.8em 0 0.4em 0;
          color: #7ecfff;
        }
        .chat-markdown ul {
          margin: 0.5em 0 0.5em 1.5em;
          padding-left: 1.2em;
        }
        .chat-markdown li {
          margin-bottom: 0.3em;
        }
        .chat-markdown p {
          margin: 0.5em 0;
        }
      `}</style>
    </div>
  );
}
