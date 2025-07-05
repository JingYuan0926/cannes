// Simple HTTP API proxy for Walrus reading
const WALRUS_AGGREGATOR = process.env.WALRUS_AGGREGATOR_URL || 'https://aggregator.walrus-testnet.walrus.space';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { blobId, format } = req.query;

    if (!blobId || typeof blobId !== 'string') {
      return res.status(400).json({ error: 'Blob ID must be a non-empty string' });
    }

    console.log('Reading from Walrus:', blobId);
    
    // Make HTTP GET request to Walrus aggregator
    const response = await fetch(`${WALRUS_AGGREGATOR}/v1/blobs/${encodeURIComponent(blobId)}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        return res.status(404).json({ 
          error: 'Blob not found. Please check the blob ID and try again.' 
        });
      }
      
      const errorText = await response.text();
      throw new Error(`Walrus aggregator error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const contentType = response.headers.get('content-type') || 'application/octet-stream';
    
    // Handle different format requests
    if (format === 'raw') {
      // Return raw binary data for downloads
      const buffer = await response.arrayBuffer();
      const uint8Array = new Uint8Array(buffer);
      
      res.setHeader('Content-Type', contentType);
      res.setHeader('Content-Length', buffer.byteLength);
      res.status(200).send(Buffer.from(uint8Array));
      return;
    }

    // For JSON response, handle both text and binary content
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
      // Handle binary content (images, PDFs, etc.)
      const buffer = await response.arrayBuffer();
      const uint8Array = new Uint8Array(buffer);
      content = Buffer.from(uint8Array).toString('base64');
      isBinary = true;
    }
    
    console.log(`Content read successfully from Walrus: ${isText ? content.length + ' characters' : 'binary data'}`);
    
    res.status(200).json({
      success: true,
      content: content,
      blobId: blobId,
      contentType: contentType,
      isText: isText,
      isBinary: isBinary,
      length: isText ? content.length : Buffer.from(content, 'base64').length,
      bytes: isText ? new TextEncoder().encode(content).length : Buffer.from(content, 'base64').length,
    });

  } catch (error) {
    console.error('Failed to read from Walrus:', error);
    
    // Handle specific network errors
    if (error.message.includes('Failed to fetch') || 
        error.message.includes('ERR_CERT_DATE_INVALID') ||
        error.message.includes('ERR_CONNECTION_REFUSED')) {
      return res.status(500).json({ 
        error: 'Network connectivity issues with Walrus aggregator. Please try again later.' 
      });
    }
    
    res.status(500).json({ error: error.message });
  }
} 