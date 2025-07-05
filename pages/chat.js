import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import rehypeRaw from 'rehype-raw';

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
        files.forEach((file) => {
          formData.append('files', file);
        });
        formData.append('prompt', userInput);
        
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
        <div
          style={{
            backgroundColor: '#1a1a1a',
            padding: '20px',
            borderRadius: '12px',
            fontSize: '15px',
            marginBottom: '24px',
            lineHeight: 1.7,
          }}
        >
          {conversation.map((msg, idx) => (
            <div
              key={idx}
              style={{
                marginBottom: '24px',
                padding: '16px',
                background: msg.role === 'assistant' ? '#181f1a' : 'transparent',
                borderRadius: '8px',
                border: msg.role === 'assistant' ? '1px solid #333' : 'none',
              }}
            >
              <b style={{ color: msg.role === 'user' ? '#6cf' : '#9f6' }}>
                {msg.role === 'user' ? 'You' : 'AI'}:
              </b>{' '}
              {msg.role === 'assistant'
                ? (
                    (() => { console.log('AI content:', JSON.stringify(msg.content)); return null; })(),
                    <div className="react-markdown">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkBreaks]}
                        rehypePlugins={[rehypeRaw]}
                        skipHtml={true}
                        style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )
                : <span>{msg.content}</span>
              }
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
