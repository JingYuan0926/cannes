// Walrus HTTP API utilities for reading
const WALRUS_AGGREGATOR = process.env.NEXT_PUBLIC_WALRUS_AGGREGATOR_URL || 'https://aggregator.walrus-testnet.walrus.space';

/**
 * Read From Walrus via API Route (readFW)
 * Uses the Next.js API route which proxies to Walrus aggregator
 * @param {string} blobId - The blob ID to read from Walrus
 * @returns {Promise<object>} - The content with metadata
 */
export const readFW = async (blobId) => {
  if (!blobId || typeof blobId !== 'string') {
    throw new Error('Blob ID must be a non-empty string');
  }

  try {
    console.log('Reading from Walrus via API:', blobId);
    
    const response = await fetch(`/api/walrus/read?blobId=${encodeURIComponent(blobId)}`);
    
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Failed to read from Walrus');
    }
    
    console.log('Content read successfully from Walrus:', result.isText ? result.length + ' characters' : 'binary data');
    return result;
  } catch (error) {
    console.error('Failed to read from Walrus:', error);
    throw error;
  }
};

/**
 * Read From Walrus via Direct HTTP API (readFWDirect)
 * Makes a direct HTTP request to a Walrus aggregator
 * @param {string} blobId - The blob ID to read from Walrus
 * @param {Object} options - Read options
 * @param {string} options.aggregatorUrl - Custom aggregator URL (optional)
 * @returns {Promise<string>} - The text content
 */
export const readFWDirect = async (blobId, options = {}) => {
  if (!blobId || typeof blobId !== 'string') {
    throw new Error('Blob ID must be a non-empty string');
  }

  const { aggregatorUrl = WALRUS_AGGREGATOR } = options;

  try {
    console.log(`Reading from Walrus directly: ${blobId}`);

    const response = await fetch(`${aggregatorUrl}/v1/blobs/${encodeURIComponent(blobId)}`);

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Blob not found. Please check the blob ID and try again.');
      }
      
      const errorText = await response.text();
      throw new Error(`Walrus aggregator error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const text = await response.text();
    console.log(`Text read successfully from Walrus: ${text.length} characters`);
    return text;
  } catch (error) {
    console.error('Failed to read from Walrus:', error);
    throw error;
  }
};

/**
 * Read Content with Type Detection from Walrus
 * @param {string} blobId - The blob ID to read from Walrus
 * @param {Object} options - Read options
 * @param {string} options.aggregatorUrl - Custom aggregator URL (optional)
 * @returns {Promise<object>} - Content with type information
 */
export const readContentWithType = async (blobId, options = {}) => {
  if (!blobId || typeof blobId !== 'string') {
    throw new Error('Blob ID must be a non-empty string');
  }

  const { aggregatorUrl = WALRUS_AGGREGATOR } = options;

  try {
    console.log(`Reading content with type detection: ${blobId}`);

    const response = await fetch(`${aggregatorUrl}/v1/blobs/${encodeURIComponent(blobId)}`);

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Blob not found. Please check the blob ID and try again.');
      }
      
      const errorText = await response.text();
      throw new Error(`Walrus aggregator error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const contentType = response.headers.get('content-type') || 'application/octet-stream';
    
    let content;
    let isText = false;
    let isBinary = false;
    
    // Detect if content is likely text or binary
    if (contentType.startsWith('text/') || 
        contentType.includes('json') || 
        contentType.includes('xml') || 
        contentType.includes('csv')) {
      content = await response.text();
      isText = true;
    } else {
      // Handle binary content
      const buffer = await response.arrayBuffer();
      const uint8Array = new Uint8Array(buffer);
      content = Buffer.from(uint8Array).toString('base64');
      isBinary = true;
    }

    console.log(`Content read successfully: ${isText ? content.length + ' characters' : 'binary data'}`);
    
    return {
      content,
      contentType,
      isText,
      isBinary,
      blobId,
      length: isText ? content.length : Buffer.from(content, 'base64').length,
      bytes: isText ? new TextEncoder().encode(content).length : Buffer.from(content, 'base64').length,
    };
  } catch (error) {
    console.error('Failed to read content from Walrus:', error);
    throw error;
  }
};

/**
 * Download File from Walrus
 * @param {string} blobId - The blob ID to download
 * @param {string} filename - Filename for download
 * @param {Object} options - Download options
 * @param {string} options.aggregatorUrl - Custom aggregator URL (optional)
 */
export const downloadFile = async (blobId, filename = 'download', options = {}) => {
  if (!blobId || typeof blobId !== 'string') {
    throw new Error('Blob ID must be a non-empty string');
  }

  try {
    // Use API route for raw download
    const response = await fetch(`/api/walrus/read?blobId=${encodeURIComponent(blobId)}&format=raw`);

    if (!response.ok) {
      throw new Error('Failed to download file from Walrus');
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    console.log(`File downloaded: ${filename}`);
  } catch (error) {
    console.error('Failed to download file:', error);
    throw error;
  }
};

/**
 * Parse CSV content into array of objects
 * @param {string} csvContent - CSV text content
 * @returns {Array<object>} - Parsed CSV data
 */
export const parseCSV = (csvContent) => {
  const lines = csvContent.trim().split('\n');
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  const data = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || '';
    });
    data.push(row);
  }

  return data;
};

/**
 * Detect file type from content and blob ID
 * @param {string} content - File content
 * @param {string} contentType - Content type header
 * @param {string} blobId - Blob ID (may contain filename hints)
 * @returns {object} - File type information
 */
export const detectFileType = (content, contentType, blobId) => {
  const typeInfo = {
    type: 'unknown',
    category: 'other',
    displayName: 'Unknown File',
    canPreview: false,
    icon: '📄',
  };

  // Check content type first
  if (contentType.includes('image/')) {
    typeInfo.type = 'image';
    typeInfo.category = 'image';
    typeInfo.displayName = 'Image';
    typeInfo.canPreview = true;
    typeInfo.icon = '🖼️';
  } else if (contentType.includes('pdf')) {
    typeInfo.type = 'pdf';
    typeInfo.category = 'document';
    typeInfo.displayName = 'PDF Document';
    typeInfo.canPreview = false;
    typeInfo.icon = '📄';
  } else if (contentType.includes('json') || (content && content.trim().startsWith('{'))) {
    typeInfo.type = 'json';
    typeInfo.category = 'data';
    typeInfo.displayName = 'JSON Data';
    typeInfo.canPreview = true;
    typeInfo.icon = '📊';
  } else if (contentType.includes('csv') || (content && content.includes(',') && content.includes('\n'))) {
    typeInfo.type = 'csv';
    typeInfo.category = 'data';
    typeInfo.displayName = 'CSV Data';
    typeInfo.canPreview = true;
    typeInfo.icon = '📈';
  } else if (contentType.includes('xml') || (content && content.trim().startsWith('<'))) {
    typeInfo.type = 'xml';
    typeInfo.category = 'data';
    typeInfo.displayName = 'XML Data';
    typeInfo.canPreview = true;
    typeInfo.icon = '🏷️';
  } else if (contentType.startsWith('text/')) {
    typeInfo.type = 'text';
    typeInfo.category = 'text';
    typeInfo.displayName = 'Text File';
    typeInfo.canPreview = true;
    typeInfo.icon = '📝';
  }

  return typeInfo;
};

/**
 * Read Blob by Object ID via Direct HTTP API
 * @param {string} objectId - The Sui object ID of the blob
 * @param {Object} options - Read options
 * @param {string} options.aggregatorUrl - Custom aggregator URL (optional)
 * @returns {Promise<string>} - The text content
 */
export const readByObjectId = async (objectId, options = {}) => {
  if (!objectId || typeof objectId !== 'string') {
    throw new Error('Object ID must be a non-empty string');
  }

  const { aggregatorUrl = WALRUS_AGGREGATOR } = options;

  try {
    console.log(`Reading from Walrus by object ID: ${objectId}`);

    const response = await fetch(`${aggregatorUrl}/v1/blobs/by-object-id/${encodeURIComponent(objectId)}`);

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Blob not found. Please check the object ID and try again.');
      }
      
      const errorText = await response.text();
      throw new Error(`Walrus aggregator error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const text = await response.text();
    console.log(`Text read successfully from Walrus: ${text.length} characters`);
    return text;
  } catch (error) {
    console.error('Failed to read from Walrus:', error);
    throw error;
  }
};

/**
 * Check if a blob exists on Walrus
 * @param {string} blobId - The blob ID to check
 * @param {Object} options - Check options
 * @param {string} options.aggregatorUrl - Custom aggregator URL (optional)
 * @returns {Promise<boolean>} - Whether the blob exists
 */
export const blobExists = async (blobId, options = {}) => {
  if (!blobId || typeof blobId !== 'string') {
    throw new Error('Blob ID must be a non-empty string');
  }

  const { aggregatorUrl = WALRUS_AGGREGATOR } = options;

  try {
    console.log(`Checking if blob exists: ${blobId}`);

    const response = await fetch(`${aggregatorUrl}/v1/blobs/${encodeURIComponent(blobId)}`, {
      method: 'HEAD', // Use HEAD to check existence without downloading content
    });

    return response.ok;
  } catch (error) {
    console.error('Failed to check blob existence:', error);
    return false;
  }
};

/**
 * Get available Walrus testnet aggregators
 * @returns {string[]} - Array of aggregator URLs
 */
export const getTestnetAggregators = () => [
  'https://agg.test.walrus.eosusa.io',
  'https://aggregator.testnet.walrus.atalma.io',
  'https://aggregator.testnet.walrus.mirai.cloud',
  'https://aggregator.walrus-01.tududes.com',
  'https://aggregator.walrus-testnet.walrus.space',
  'https://aggregator.walrus.banansen.dev',
  'https://aggregator.walrus.testnet.mozcomputing.dev',
  'https://sm1-walrus-testnet-aggregator.stakesquid.com',
  'https://sui-walrus-tn-aggregator.bwarelabs.com',
  'https://suiftly-testnet-agg.mhax.io',
]; 