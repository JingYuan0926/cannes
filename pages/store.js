import { useState, useEffect } from 'react';

export default function WalrusStorePage() {
  const [textToStore, setTextToStore] = useState('');
  const [blobIdToRead, setBlobIdToRead] = useState('');
  const [storedBlobId, setStoredBlobId] = useState('');
  const [readData, setReadData] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [signerAddress, setSignerAddress] = useState('');
  const [status, setStatus] = useState('');

  useEffect(() => {
    // Initialize and get signer address when component mounts
    const initStore = async () => {
      try {
        const response = await fetch('/api/walrus/signer');
        const result = await response.json();
        
        if (response.ok && result.signerAddress) {
          setSignerAddress(result.signerAddress);
          setStatus('Walrus store initialized successfully');
        } else {
          throw new Error(result.error || 'Failed to get signer address');
        }
      } catch (err) {
        setError(`Failed to initialize: ${err.message}`);
      }
    };

    initStore();
  }, []);

  const handleStoreText = async () => {
    if (!textToStore.trim()) {
      setError('Please enter some text to store');
      return;
    }

    setIsLoading(true);
    setError('');
    setStatus('Writing text to Walrus...');

    try {
      const response = await fetch('/api/walrus/write', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: textToStore,
          epochs: 3,
          deletable: false,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Failed to write to Walrus');
      }

      setStoredBlobId(result.blobId);
      setStatus(`Text written successfully! Blob ID: ${result.blobId}`);
    } catch (err) {
      setError(`Failed to write text: ${err.message}`);
      setStatus('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReadText = async () => {
    if (!blobIdToRead.trim()) {
      setError('Please enter a blob ID to read');
      return;
    }

    setIsLoading(true);
    setError('');
    setStatus('Reading text from Walrus...');

    try {
      const response = await fetch(`/api/walrus/read?blobId=${encodeURIComponent(blobIdToRead)}`);
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Failed to read from Walrus');
      }

      setReadData(result.text);
      setStatus(`Text read successfully from blob: ${blobIdToRead}`);
    } catch (err) {
      setError(`Failed to read text: ${err.message}`);
      setStatus('');
    } finally {
      setIsLoading(false);
    }
  };

  const clearError = () => {
    setError('');
  };

  const clearReadData = () => {
    setReadData('');
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-6">Walrus Text Storage</h1>
          
          {/* Status Section */}
          <div className="mb-6 p-4 bg-blue-50 rounded-lg">
            <h2 className="text-lg font-semibold text-blue-900 mb-2">Status</h2>
            {signerAddress && (
              <p className="text-sm text-blue-700 mb-2">
                <strong>Signer Address:</strong> {signerAddress}
              </p>
            )}
            {status && (
              <p className="text-sm text-blue-700">{status}</p>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex justify-between items-start">
                <p className="text-red-700 text-sm">{error}</p>
                <button
                  onClick={clearError}
                  className="text-red-500 hover:text-red-700 ml-2"
                >
                  ×
                </button>
              </div>
            </div>
          )}

          {/* API Architecture Info */}
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <h3 className="text-lg font-semibold text-green-800 mb-2">✅ Server-Side API Architecture</h3>
            <p className="text-sm text-green-700 mb-2">
              Using Next.js API routes for optimal performance and reliability:
            </p>
            <ul className="text-sm text-green-700 list-disc list-inside space-y-1">
              <li>Server-side Walrus SDK execution (no browser limitations)</li>
              <li>Faster operations with reduced network requests</li>
              <li>Better error handling and retry logic</li>
              <li>No CORS or SSL certificate issues</li>
            </ul>
          </div>

          {/* Write Text Section */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Write Text to Walrus</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Text to Store:
                </label>
                <textarea
                  value={textToStore}
                  onChange={(e) => setTextToStore(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows="6"
                  placeholder="Enter text to store in Walrus..."
                />
                <p className="text-xs text-gray-500 mt-1">
                  Characters: {textToStore.length} | Bytes: {new TextEncoder().encode(textToStore).length}
                </p>
              </div>
              <button
                onClick={handleStoreText}
                disabled={isLoading || !textToStore.trim()}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Writing to Walrus...' : 'Write Text to Walrus'}
              </button>
            </div>
          </div>

          {/* Stored Blob ID Display */}
          {storedBlobId && (
            <div className="mb-8 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="text-lg font-semibold text-green-800 mb-2">Written Successfully!</h3>
              <p className="text-sm text-green-700 break-all mb-2">
                <strong>Blob ID:</strong> {storedBlobId}
              </p>
              <button
                onClick={() => setBlobIdToRead(storedBlobId)}
                className="text-xs bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700"
              >
                Use this ID to read
              </button>
            </div>
          )}

          {/* Read Text Section */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Read Text from Walrus</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Blob ID to Read:
                </label>
                <input
                  type="text"
                  value={blobIdToRead}
                  onChange={(e) => setBlobIdToRead(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Enter blob ID to read... (try: OFrKO0ofGc4inX8roHHaAB-pDHuUiIA08PW4N2B2gFk)"
                />
              </div>
              <button
                onClick={handleReadText}
                disabled={isLoading || !blobIdToRead.trim()}
                className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Reading from Walrus...' : 'Read Text from Walrus'}
              </button>
            </div>
          </div>

          {/* Read Data Display */}
          {readData && (
            <div className="mb-8">
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-lg font-semibold text-gray-800">Retrieved Text:</h3>
                <button
                  onClick={clearReadData}
                  className="text-gray-500 hover:text-gray-700"
                >
                  Clear
                </button>
              </div>
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <pre className="text-sm text-gray-700 whitespace-pre-wrap break-words">
                  {readData}
                </pre>
                <p className="text-xs text-gray-500 mt-2">
                  Characters: {readData.length} | Bytes: {new TextEncoder().encode(readData).length}
                </p>
              </div>
            </div>
          )}

          {/* API Routes Info */}
          <div className="mt-8 p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">API Routes</h3>
            <div className="text-sm text-gray-700 space-y-1">
              <p><code className="bg-gray-200 px-2 py-1 rounded">POST /api/walrus/write</code> - Write text to Walrus</p>
              <p><code className="bg-gray-200 px-2 py-1 rounded">GET /api/walrus/read?blobId=...</code> - Read text from Walrus</p>
              <p><code className="bg-gray-200 px-2 py-1 rounded">GET /api/walrus/signer</code> - Get signer address</p>
            </div>
          </div>

          {/* Direct API Usage Examples */}
          <div className="mt-8 p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Direct API Usage</h3>
            <div className="text-sm text-gray-700 space-y-2">
              <p><strong>Write Text:</strong></p>
              <pre className="bg-gray-200 p-2 rounded text-xs overflow-x-auto">
{`fetch('/api/walrus/write', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Hello Walrus!', epochs: 3, deletable: false })
})`}
              </pre>
              <p><strong>Read Text:</strong></p>
              <pre className="bg-gray-200 p-2 rounded text-xs overflow-x-auto">
{`fetch('/api/walrus/read?blobId=YOUR_BLOB_ID')`}
              </pre>
              <p><strong>Get Signer:</strong></p>
              <pre className="bg-gray-200 p-2 rounded text-xs overflow-x-auto">
{`fetch('/api/walrus/signer')`}
              </pre>
            </div>
          </div>

          {/* Setup Instructions */}
          <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-800 mb-2">Setup Instructions:</h3>
            <ol className="text-sm text-blue-700 space-y-1">
              <li>1. Install dependencies: <code className="bg-blue-100 px-1 rounded">npm install @mysten/walrus @mysten/sui</code></li>
              <li>2. Set environment variable: <code className="bg-blue-100 px-1 rounded">NEXT_PUBLIC_WALRUS_SEED_PHRASE="your seed phrase"</code></li>
              <li>3. Ensure your account has SUI tokens for gas fees and WAL tokens for storage</li>
              <li>4. This demo uses Sui testnet - make sure to use testnet tokens</li>
              <li>5. API routes handle all Walrus operations server-side for optimal performance</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
