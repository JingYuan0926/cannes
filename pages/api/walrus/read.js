// Simple HTTP API proxy for Walrus reading
const WALRUS_AGGREGATOR = process.env.WALRUS_AGGREGATOR_URL || 'https://aggregator.walrus-testnet.walrus.space';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { blobId } = req.query;

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

    // Get the text content
    const text = await response.text();
    
    console.log('Text read successfully from Walrus:', text.length, 'characters');
    
    res.status(200).json({
      success: true,
      text: text,
      blobId: blobId,
      length: text.length,
      bytes: new TextEncoder().encode(text).length,
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