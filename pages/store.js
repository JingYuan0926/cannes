import { useState, useEffect } from 'react';
import { readFW, readContentWithType, parseCSV, detectFileType, downloadFile } from '../utils/readFromWalrus';
import { uploadFile } from '../utils/writeToWalrus';

export default function WalrusStorePage() {
  const [textToStore, setTextToStore] = useState('');
  const [fileToUpload, setFileToUpload] = useState(null);
  const [uploadMode, setUploadMode] = useState('text'); // 'text' or 'file'
  const [blobIdToRead, setBlobIdToRead] = useState('');
  const [storedBlobId, setStoredBlobId] = useState('');
  const [readData, setReadData] = useState(null);
  const [manualFileType, setManualFileType] = useState('auto');
  const [showDecoder, setShowDecoder] = useState(false);
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

  const handleReadContent = async () => {
    if (!blobIdToRead.trim()) {
      setError('Please enter a blob ID to read');
      return;
    }

    setIsLoading(true);
    setError('');
    setStatus('Reading content from Walrus...');

    try {
      // Use enhanced read function with content type detection
      const result = await readFW(blobIdToRead);
      
      // Detect file type and add additional metadata
      let fileType = detectFileType(result.content, result.contentType, result.blobId);
      
      // Override with manual file type if specified
      if (manualFileType !== 'auto') {
        fileType = getManualFileType(manualFileType, result.content);
      }
      
      const contentData = {
        ...result,
        fileType,
      };

      // Parse CSV data if it's a CSV file
      if (fileType.type === 'csv' && result.isText) {
        try {
          contentData.csvData = parseCSV(result.content);
        } catch (csvError) {
          console.warn('Failed to parse CSV:', csvError);
          contentData.csvData = null;
        }
      }

      // Try to parse JSON if it's JSON content
      if (fileType.type === 'json' && result.isText) {
        try {
          contentData.jsonData = JSON.parse(result.content);
        } catch (jsonError) {
          console.warn('Failed to parse JSON:', jsonError);
          contentData.jsonData = null;
        }
      }

      // If it's binary but manual type suggests text, try to decode
      if (result.isBinary && ['text', 'json', 'csv', 'xml'].includes(fileType.type)) {
        try {
          const decodedText = atob(result.content);
          contentData.content = decodedText;
          contentData.isText = true;
          contentData.isBinary = false;
          
          // Re-parse if needed
          if (fileType.type === 'csv') {
            contentData.csvData = parseCSV(decodedText);
          } else if (fileType.type === 'json') {
            contentData.jsonData = JSON.parse(decodedText);
          }
        } catch (decodeError) {
          console.warn('Failed to decode binary as text:', decodeError);
        }
      }

      setReadData(contentData);
      setStatus(`Content read successfully from blob: ${blobIdToRead} (${fileType.displayName})`);
    } catch (err) {
      setError(`Failed to read content: ${err.message}`);
      setStatus('');
    } finally {
      setIsLoading(false);
    }
  };

  const getManualFileType = (type, content) => {
    const typeMap = {
      'text': { type: 'text', category: 'text', displayName: 'Text File', canPreview: true, icon: '📝' },
      'json': { type: 'json', category: 'data', displayName: 'JSON Data', canPreview: true, icon: '📊' },
      'csv': { type: 'csv', category: 'data', displayName: 'CSV Data', canPreview: true, icon: '📈' },
      'xml': { type: 'xml', category: 'data', displayName: 'XML Data', canPreview: true, icon: '🏷️' },
      'image': { type: 'image', category: 'image', displayName: 'Image', canPreview: true, icon: '🖼️' },
    };
    
    return typeMap[type] || { type: 'unknown', category: 'other', displayName: 'Unknown File', canPreview: false, icon: '📄' };
  };

  const handleDownload = async () => {
    if (!readData) return;
    
    try {
      const filename = `walrus-${readData.blobId.substring(0, 8)}.${getFileExtension(readData.fileType.type)}`;
      await downloadFile(readData.blobId, filename);
      setStatus(`File downloaded: ${filename}`);
    } catch (err) {
      setError(`Failed to download file: ${err.message}`);
    }
  };

  const decodeAsText = () => {
    if (!readData || !readData.isBinary) return;
    
    try {
      const decodedText = atob(readData.content);
      const updatedData = {
        ...readData,
        content: decodedText,
        isText: true,
        isBinary: false,
        fileType: { 
          type: 'text', 
          category: 'text', 
          displayName: 'Decoded Text', 
          canPreview: true, 
          icon: '📝' 
        }
      };
      setReadData(updatedData);
      setStatus('Successfully decoded binary content as text');
    } catch (err) {
      setError('Failed to decode binary content as text: ' + err.message);
    }
  };

  const getFileExtension = (fileType) => {
    const extensions = {
      'text': 'txt',
      'json': 'json',
      'csv': 'csv',
      'xml': 'xml',
      'image': 'jpg',
      'pdf': 'pdf',
    };
    return extensions[fileType] || 'bin';
  };

  const clearError = () => {
    setError('');
  };

  const clearReadData = () => {
    setReadData(null);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const renderContent = () => {
    if (!readData) return null;

    const { content, fileType, isText, isBinary, csvData, jsonData } = readData;

    // Render different content types
    switch (fileType.type) {
      case 'image':
        return (
          <div className="text-center">
            <img 
              src={`data:${readData.contentType};base64,${content}`}
              alt="Walrus Image"
              className="max-w-full max-h-96 mx-auto rounded-lg shadow-md"
            />
          </div>
        );

      case 'csv':
        return csvData && csvData.length > 0 ? (
          <div>
            <h4 className="font-semibold mb-2">CSV Data Table:</h4>
            <div className="overflow-x-auto max-h-96">
              <table className="min-w-full border border-gray-300 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    {Object.keys(csvData[0]).map((header) => (
                      <th key={header} className="border border-gray-300 px-2 py-1 text-left font-medium">
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {csvData.slice(0, 20).map((row, index) => (
                    <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      {Object.values(row).map((cell, cellIndex) => (
                        <td key={cellIndex} className="border border-gray-300 px-2 py-1">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {csvData.length > 20 && (
                <p className="text-xs text-gray-500 mt-2">
                  Showing first 20 rows of {csvData.length} total rows
                </p>
              )}
            </div>
          </div>
        ) : (
          <pre className="text-sm text-gray-700 whitespace-pre-wrap break-words max-h-96 overflow-y-auto bg-gray-50 p-4 rounded">
            {content}
          </pre>
        );

      case 'json':
        return (
          <div>
            <h4 className="font-semibold mb-2">JSON Data:</h4>
            <pre className="text-sm text-gray-700 whitespace-pre-wrap break-words max-h-96 overflow-y-auto bg-gray-50 p-4 rounded border">
              {jsonData ? JSON.stringify(jsonData, null, 2) : content}
            </pre>
          </div>
        );

      case 'xml':
        return (
          <div>
            <h4 className="font-semibold mb-2">XML Data:</h4>
            <pre className="text-sm text-gray-700 whitespace-pre-wrap break-words max-h-96 overflow-y-auto bg-gray-50 p-4 rounded border">
              {content}
            </pre>
          </div>
        );

      case 'text':
        return (
          <pre className="text-sm text-gray-700 whitespace-pre-wrap break-words max-h-96 overflow-y-auto bg-gray-50 p-4 rounded">
            {content}
          </pre>
        );

      case 'pdf':
      default:
        if (isBinary) {
          return (
            <div className="text-center p-8 bg-gray-50 rounded border">
              <div className="text-4xl mb-2">{fileType.icon}</div>
              <p className="text-gray-600 mb-4">Binary file: {fileType.displayName}</p>
              <p className="text-sm text-gray-500">
                Size: {formatFileSize(readData.bytes)} • Content Type: {readData.contentType}
              </p>
              <button
                onClick={handleDownload}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Download File
              </button>
            </div>
          );
        } else {
          return (
            <pre className="text-sm text-gray-700 whitespace-pre-wrap break-words max-h-96 overflow-y-auto bg-gray-50 p-4 rounded">
              {content}
            </pre>
          );
        }
    }
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
            <h3 className="text-lg font-semibold text-green-800 mb-2">✅ Multi-Format Document Viewer</h3>
            <p className="text-sm text-green-700 mb-2">
              Enhanced Walrus storage with intelligent document display:
            </p>
            <ul className="text-sm text-green-700 list-disc list-inside space-y-1">
              <li><strong>📈 CSV Files:</strong> Rendered as interactive tables with sorting</li>
              <li><strong>📊 JSON Data:</strong> Pretty-printed with syntax highlighting</li>
              <li><strong>🖼️ Images:</strong> Direct image preview (JPG, PNG, GIF, WebP)</li>
              <li><strong>📄 PDF Documents:</strong> Download and view externally</li>
              <li><strong>🏷️ XML/HTML:</strong> Formatted markup display</li>
              <li><strong>📝 Text Files:</strong> Plain text with proper formatting</li>
              <li><strong>🔄 Auto-detection:</strong> Content type recognition and appropriate rendering</li>
              <li><strong>⬇️ Download:</strong> Save any file type to your device</li>
            </ul>
          </div>

          {/* Decoder Help Section */}
          <div className="mb-6 p-4 bg-orange-50 border border-orange-200 rounded-lg">
            <h3 className="text-lg font-semibold text-orange-800 mb-2">🔧 Downloaded .bin Files? Here's How to Decode Them</h3>
            <p className="text-sm text-orange-700 mb-2">
              If your downloaded files have .bin extensions but should be text/CSV/JSON:
            </p>
            <div className="text-sm text-orange-700 space-y-2">
              <div className="bg-orange-100 p-3 rounded">
                <h4 className="font-semibold mb-1">🎯 Quick Fix - Use the Decoder Above:</h4>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Enter your blob ID in the "Read Content" section</li>
                  <li>Change "Force File Type" from "Auto-detect" to the correct type</li>
                  <li>Click "Read Content from Walrus" - it will auto-decode!</li>
                  <li>Or use the "Binary File Decoder" buttons if content appears as binary</li>
                </ol>
              </div>
              <div className="bg-orange-100 p-3 rounded">
                <h4 className="font-semibold mb-1">🛠️ Manual Decoding (for downloaded files):</h4>
                <ul className="list-disc list-inside space-y-1">
                  <li><strong>Windows:</strong> Right-click → "Open with" → Notepad/Excel/Browser</li>
                  <li><strong>Mac:</strong> Right-click → "Open with" → TextEdit/Numbers/Browser</li>
                  <li><strong>Rename:</strong> Change .bin to .txt/.csv/.json and double-click</li>
                  <li><strong>Online:</strong> Use online base64 decoders if needed</li>
                </ul>
              </div>
              <p className="text-xs">
                💡 <strong>Why this happens:</strong> Walrus sometimes doesn't provide proper content-type headers, 
                so files get treated as binary. The decoder above fixes this automatically!
              </p>
            </div>
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
                  Auto-detects content type and renders appropriately
                </p>
              </div>

              {/* Manual File Type Override */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Force File Type (for .bin files):
                </label>
                <select
                  value={manualFileType}
                  onChange={(e) => setManualFileType(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                >
                  <option value="auto">🔄 Auto-detect</option>
                  <option value="text">📝 Text File</option>
                  <option value="json">📊 JSON Data</option>
                  <option value="csv">📈 CSV Data</option>
                  <option value="xml">🏷️ XML Data</option>
                  <option value="image">🖼️ Image</option>
                </select>
                <p className="text-xs text-orange-600 mt-1">
                  Use this if your file was incorrectly detected as binary (.bin)
                </p>
              </div>
              
              <button
                onClick={handleReadContent}
                disabled={isLoading || !blobIdToRead.trim()}
                className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Reading from Walrus...' : 'Read Content from Walrus'}
              </button>
            </div>
          </div>

          {/* Binary Decoder Helper */}
          {readData && readData.isBinary && (
            <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <h3 className="text-lg font-semibold text-yellow-800 mb-2">🔧 Binary File Decoder</h3>
              <p className="text-sm text-yellow-700 mb-3">
                This file was detected as binary. If it should be text, try these decoding options:
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={decodeAsText}
                  className="text-sm bg-yellow-600 text-white px-3 py-1 rounded hover:bg-yellow-700"
                >
                  📝 Decode as Text
                </button>
                <button
                  onClick={() => {
                    setManualFileType('json');
                    handleReadContent();
                  }}
                  className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700"
                >
                  📊 Try as JSON
                </button>
                <button
                  onClick={() => {
                    setManualFileType('csv');
                    handleReadContent();
                  }}
                  className="text-sm bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700"
                >
                  📈 Try as CSV
                </button>
              </div>
            </div>
          )}

          {/* Read Data Display */}
          {readData && (
            <div className="mb-8">
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">{readData.fileType.icon}</span>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">{readData.fileType.displayName}</h3>
                    <p className="text-sm text-gray-500">
                      {formatFileSize(readData.bytes)} • {readData.contentType}
                      {readData.csvData && ` • ${readData.csvData.length} rows`}
                    </p>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={handleDownload}
                    className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700"
                  >
                    Download
                  </button>
                  <button
                    onClick={clearReadData}
                    className="text-sm bg-gray-500 text-white px-3 py-1 rounded hover:bg-gray-600"
                  >
                    Clear
                  </button>
                </div>
              </div>
              
              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                {renderContent()}
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