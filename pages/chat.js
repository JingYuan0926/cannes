import { useState } from 'react';

export default function Chat({ walletAddress }) {
  const [userInput, setUserInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState('');
  const [actionApiUrl, setActionApiUrl] = useState('');
  const [justCopied, setJustCopied] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim()) {
      setResponse('Please enter your question.');
      return;
    }

    setLoading(true);

    try {
      const res = await fetch('/api/chatbot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: userInput, walletAddress }),
      });

      if (!res.ok) throw new Error(`Error: ${res.status}`);
      const result = await res.json();
      setResponse(result.message);

      const newApiUrl = `http://localhost:3000/api/actions/${walletAddress}/runAnalysis`;
      setActionApiUrl(newApiUrl);

    } catch (err) {
      console.error('Error:', err);
      setResponse('Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(actionApiUrl).then(() => {
      setJustCopied(true);
      setTimeout(() => setJustCopied(false), 2000);
    });
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px', color: '#ddd', backgroundColor: '#121212' }}>
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
        <textarea
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          rows="4"
          placeholder="Ask a data question..."
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
          {loading ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </form>

      {actionApiUrl && (
        <div style={{ marginBottom: '20px', backgroundColor: '#1e1e1e', padding: '15px', borderRadius: '8px' }}>
          <h3 style={{ marginBottom: '10px' }}>Analysis Ready</h3>
          <p style={{ wordBreak: 'break-all', fontSize: '14px' }}>{actionApiUrl}</p>
          <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
            <button
              onClick={copyToClipboard}
              style={{
                padding: '6px 12px',
                backgroundColor: '#444',
                color: '#ccc',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              {justCopied ? 'Copied!' : 'Copy Link'}
            </button>
            <button
              onClick={async () => {
                try {
                  const res = await fetch('/api/runDataAnalysis', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                      dataset: 'User Dataset', 
                      type: 'Analysis', 
                      title: 'Data Analysis', 
                      action: 'Perform comprehensive analysis',
                      prompt: userInput 
                    }),
                  });
                  
                  if (!res.ok) throw new Error(`Error: ${res.status}`);
                  const result = await res.json();
                  setResponse(result.analysis || 'Analysis completed successfully.');
                } catch (err) {
                  console.error('Analysis Error:', err);
                  setResponse('Failed to run analysis. Please try again.');
                }
              }}
              style={{
                padding: '6px 12px',
                backgroundColor: '#3366cc',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Run Analysis Now
            </button>
          </div>
        </div>
      )}

      {response && (
        <div style={{ backgroundColor: '#1a1a1a', padding: '15px', borderRadius: '8px', fontSize: '14px' }}>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
}
