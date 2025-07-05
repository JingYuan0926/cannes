import { useState, useEffect } from 'react';
import { readFWDirect } from '../utils/readFromWalrus';
import { uploadFile } from '../utils/writeToWalrus';

export default function WalrusStorePage() {
  const [textToStore, setTextToStore] = useState('');
  const [fileToUpload, setFileToUpload] = useState(null);
  const [uploadMode, setUploadMode] = useState('text'); // 'text' or 'file'
  const [blobIdToRead, setBlobIdToRead] = useState('');
  const [storedBlobId, setStoredBlobId] = useState('');
  const [readData, setReadData] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('Walrus HTTP API ready');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setFileToUpload(file);
  };

  const handleStoreContent = async () => {
    if (uploadMode === 'text') {
      if (!textToStore.trim()) {
        setError('Please enter some text to store');
        return;
      }
    } else {
      if (!fileToUpload) {
        setError('Please select a file to upload');
        return;
      }
    }

    setIsLoading(true);
    setError('');
    setStatus(uploadMode === 'text' ? 'Writing text to Walrus...' : 'Uploading file to Walrus...');

    try {
      let result;
      
      if (uploadMode === 'text') {
        // Handle text upload
        const response = await fetch('/api/walrus/write', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: textToStore,
            // Using defaults: epochs: 1, deletable: true
          }),
        });

        result = await response.json();

        if (!response.ok) {
          throw new Error(result.error || 'Failed to write to Walrus');
        }
      } else {
        // Handle file upload
        result = await uploadFile(fileToUpload);
      }

      setStoredBlobId(result.blobId);
      setStatus(
        uploadMode === 'text' 
          ? `Text written successfully! Blob ID: ${result.blobId}`
          : `File "${result.filename}" uploaded successfully! Blob ID: ${result.blobId}`
      );
    } catch (err) {
      setError(`Failed to ${uploadMode === 'text' ? 'write text' : 'upload file'}: ${err.message}`);
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
    setStatus('Reading content from Walrus...');

    try {
      // Use direct HTTP API
      const text = await readFWDirect(blobIdToRead);
      setStatus(`Content read successfully from blob: ${blobIdToRead}`);
      setReadData(text);
    } catch (err) {
      setError(`Failed to read content: ${err.message}`);
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

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-6">Walrus Storage</h1>
          
          {/* Status Section */}
          <div className="mb-6 p-4 bg-blue-50 rounded-lg">
            <h2 className="text-lg font-semibold text-blue-900 mb-2">Status</h2>
            <p className="text-sm text-blue-700">{status}</p>
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

          {/* HTTP API Architecture Info */}
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <h3 className="text-lg font-semibold text-green-800 mb-2">✅ HTTP API Architecture</h3>
            <p className="text-sm text-green-700 mb-2">
              Using Walrus HTTP API for simple and reliable storage:
            </p>
            <ul className="text-sm text-green-700 list-disc list-inside space-y-1">
              <li>No SDK dependencies or complex setup required</li>
              <li>Direct HTTP requests to public Walrus publishers/aggregators</li>
              <li>No keypair management or seed phrases needed</li>
              <li>Works seamlessly across all browsers and environments</li>
              <li>Supports both text content and file uploads (CSV, JSON, images, etc.)</li>
              <li>Automatic fallback to multiple public endpoints</li>
              <li><strong>Default settings:</strong> 1 epoch storage, deletable blobs</li>
            </ul>
          </div>

          {/* Write Content Section */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Store Content to Walrus</h2>
            <div className="space-y-4">
              {/* Upload Mode Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Content Type:
                </label>
                <div className="flex items-center space-x-4">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="uploadMode"
                      value="text"
                      checked={uploadMode === 'text'}
                      onChange={(e) => setUploadMode(e.target.value)}
                      className="mr-2"
                    />
                    <span className="text-sm">Text Content</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="uploadMode"
                      value="file"
                      checked={uploadMode === 'file'}
                      onChange={(e) => setUploadMode(e.target.value)}
                      className="mr-2"
                    />
                    <span className="text-sm">File Upload</span>
                  </label>
                </div>
              </div>

              {/* Text Input */}
              {uploadMode === 'text' && (
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
              )}

              {/* File Input */}
              {uploadMode === 'file' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    File to Upload:
                  </label>
                  <input
                    type="file"
                    onChange={handleFileChange}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                    accept=".csv,.json,.txt,.xml,.pdf,.jpg,.jpeg,.png,.gif,.webp,.doc,.docx,.xls,.xlsx"
                  />
                  {fileToUpload && (
                    <div className="mt-2 p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-700">
                        <strong>Selected:</strong> {fileToUpload.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        Size: {formatFileSize(fileToUpload.size)} | Type: {fileToUpload.type || 'Unknown'}
                      </p>
                    </div>
                  )}
                  <p className="text-xs text-gray-500 mt-1">
                    Supported: CSV, JSON, TXT, XML, PDF, Images, Documents, and more
                  </p>
                </div>
              )}

              <div className="flex items-center space-between">
                <div className="flex-1">
                  <p className="text-xs text-blue-600">
                    Storage: 1 epoch (≈ 1 day), deletable blob
                  </p>
                </div>
              </div>
              
              <button
                onClick={handleStoreContent}
                disabled={isLoading || (uploadMode === 'text' && !textToStore.trim()) || (uploadMode === 'file' && !fileToUpload)}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading 
                  ? (uploadMode === 'text' ? 'Writing to Walrus...' : 'Uploading to Walrus...') 
                  : (uploadMode === 'text' ? 'Write Text to Walrus' : 'Upload File to Walrus')
                }
              </button>
            </div>
          </div>

          {/* Stored Blob ID Display */}
          {storedBlobId && (
            <div className="mb-8 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="text-lg font-semibold text-green-800 mb-2">Upload Successful!</h3>
              <p className="text-sm text-green-700 break-all mb-2">
                <strong>Blob ID:</strong> {storedBlobId}
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setBlobIdToRead(storedBlobId)}
                  className="text-xs bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700"
                >
                  Use this ID to read
                </button>
                <button
                  onClick={() => navigator.clipboard.writeText(storedBlobId)}
                  className="text-xs bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700"
                >
                  Copy Blob ID
                </button>
                <a
                  href={`https://aggregator.walrus-testnet.walrus.space/v1/blobs/${storedBlobId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs bg-purple-600 text-white px-3 py-1 rounded hover:bg-purple-700"
                >
                  View on Walrus
                </a>
              </div>
            </div>
          )}

          {/* Read Content Section */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Read Content from Walrus</h2>
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
                <p className="text-xs text-blue-600 mt-1">
                  Using direct HTTP API to Walrus aggregators
                </p>
              </div>
              
              <button
                onClick={handleReadText}
                disabled={isLoading || !blobIdToRead.trim()}
                className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Reading from Walrus...' : 'Read Content from Walrus'}
              </button>
            </div>
          </div>

          {/* Read Data Display */}
          {readData && (
            <div className="mb-8">
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-lg font-semibold text-gray-800">Retrieved Content:</h3>
                <button
                  onClick={clearReadData}
                  className="text-gray-500 hover:text-gray-700"
                >
                  Clear
                </button>
              </div>
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <pre className="text-sm text-gray-700 whitespace-pre-wrap break-words max-h-96 overflow-y-auto">
                  {readData}
                </pre>
                <p className="text-xs text-gray-500 mt-2">
                  Characters: {readData.length} | Bytes: {new TextEncoder().encode(readData).length}
                </p>
              </div>
            </div>
          )}

          {/* HTTP API Usage */}
          <div className="mt-8 p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">HTTP API Usage</h3>
            <div className="text-sm text-gray-700 space-y-1">
              <p><code className="bg-gray-200 px-2 py-1 rounded">POST /api/walrus/write</code> - Write text/upload files to Walrus (proxies to publisher)</p>
              <p><code className="bg-gray-200 px-2 py-1 rounded">Direct HTTP API</code> - Read content from Walrus (direct to aggregators)</p>
            </div>
          </div>

          {/* Direct HTTP API Usage Examples */}
          <div className="mt-8 p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Direct HTTP API Usage</h3>
            <div className="text-sm text-gray-700 space-y-2">
              <p><strong>Write Text via API Route:</strong></p>
              <pre className="bg-gray-200 p-2 rounded text-xs overflow-x-auto">
{`fetch('/api/walrus/write', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    text: 'Hello Walrus!'
    // Uses defaults: epochs: 1, deletable: true
  })
})`}
              </pre>
              <p><strong>Upload File via API Route:</strong></p>
              <pre className="bg-gray-200 p-2 rounded text-xs overflow-x-auto">
{`const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('epochs', '1');
formData.append('deletable', 'true');

fetch('/api/walrus/write', {
  method: 'POST',
  body: formData
})`}
              </pre>
              <p><strong>Read Content via Direct HTTP API:</strong></p>
              <pre className="bg-gray-200 p-2 rounded text-xs overflow-x-auto">
{`fetch('https://aggregator.walrus-testnet.walrus.space/v1/blobs/YOUR_BLOB_ID')`}
              </pre>
              <p><strong>Read by Object ID:</strong></p>
              <pre className="bg-gray-200 p-2 rounded text-xs overflow-x-auto">
{`fetch('https://aggregator.walrus-testnet.walrus.space/v1/blobs/by-object-id/YOUR_OBJECT_ID')`}
              </pre>
            </div>
          </div>

          {/* Setup Instructions */}
          <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-800 mb-2">Setup Instructions:</h3>
            <ol className="text-sm text-blue-700 space-y-1">
              <li>1. No SDK dependencies required - uses simple HTTP requests</li>
              <li>2. Optional: Set environment variables for custom endpoints:</li>
              <li className="ml-4">
                <code className="bg-blue-100 px-1 rounded">WALRUS_PUBLISHER_URL</code> (default: testnet publisher)
              </li>
              <li className="ml-4">
                <code className="bg-blue-100 px-1 rounded">NEXT_PUBLIC_WALRUS_AGGREGATOR_URL</code> (client-side, default: testnet aggregator)
              </li>
              <li>3. Uses public Walrus testnet endpoints - no tokens required for reading</li>
              <li>4. Writing may require WAL tokens depending on the publisher</li>
              <li>5. HTTP API handles all operations with automatic endpoint selection</li>
              <li>6. <strong>File Support:</strong> CSV, JSON, TXT, XML, PDF, Images, Documents</li>
              <li>7. <strong>Default settings:</strong> Deletable blobs, 1 epoch storage duration</li>
            </ol>
          </div>

          {/* Public Endpoints Info */}
          <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <h3 className="text-lg font-semibold text-yellow-800 mb-2">Public Endpoints</h3>
            <div className="text-sm text-yellow-700 space-y-1">
              <p><strong>Default Publisher:</strong> https://publisher.walrus-testnet.walrus.space</p>
              <p><strong>Default Aggregator:</strong> https://aggregator.walrus-testnet.walrus.space</p>
              <p className="text-xs mt-2">
                Multiple public endpoints available for redundancy. The API automatically handles failover.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
