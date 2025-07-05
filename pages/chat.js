import { useState } from 'react';

export default function Chat() {
  const [userInput, setUserInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [sampleDataReady, setSampleDataReady] = useState(false);
  const [conversation, setConversation] = useState([]); // [{role: 'user'|'assistant', content: string}]
  const [sessionId, setSessionId] = useState(''); // Store sessionId for follow-up requests
  const [csvFile, setCsvFile] = useState(null);

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

  const handleCsvChange = (e) => {
    const file = e.target.files[0];
    setCsvFile(file);
  };

  const handleCsvUpload = async () => {
    if (!csvFile) {
      setConversation((prev) => [...prev, { role: 'assistant', content: 'Please choose a file first.' }]);
      return;
    }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', csvFile);
      
      const res = await fetch('/api/chatbot', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`Error: ${res.status}`);
      const result = await res.json();
      setConversation((prev) => [...prev, { role: 'assistant', content: result.message || 'Analysis Finished!' }]);
      setCsvFile(null);
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
          placeholder="Ask a data question or upload sample data..."
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
        <div style={{ display: 'flex', alignItems: 'center', marginTop: '10px', gap: '16px' }}>
          <label htmlFor="csv-upload" style={{
            backgroundColor: '#222',
            color: '#fff',
            padding: '8px 16px',
            borderRadius: '6px',
            cursor: 'pointer',
            border: '1px solid #444',
            fontWeight: 'bold',
          }}>
            Choose File
            <input
              id="csv-upload"
              type="file"
              accept=".csv"
              onChange={handleCsvChange}
              style={{ display: 'none' }}
            />
          </label>
          <span style={{ color: '#ccc', fontSize: '14px' }}>{csvFile ? csvFile.name : 'No file chosen'}</span>
          <button
            type="button"
            onClick={handleCsvUpload}
            disabled={loading}
            style={{
              padding: '10px 24px',
              backgroundColor: '#3366cc',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontWeight: 'bold',
            }}
          >
            {loading ? 'Uploading...' : 'Run Analysis'}
          </button>
        </div>
      </form>
      {conversation.length > 0 && (
        <div style={{ backgroundColor: '#1a1a1a', padding: '15px', borderRadius: '8px', fontSize: '14px', marginBottom: '20px' }}>
          {conversation.map((msg, idx) => (
            <div key={idx} style={{ marginBottom: '10px' }}>
              <b style={{ color: msg.role === 'user' ? '#6cf' : '#9f6' }}>{msg.role === 'user' ? 'You' : 'AI'}:</b> {msg.content}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
