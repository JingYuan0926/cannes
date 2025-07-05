// Simple HTTP API proxy for Walrus storage
const WALRUS_PUBLISHER = process.env.WALRUS_PUBLISHER_URL || 'https://publisher.walrus-testnet.walrus.space';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { text, epochs = 1, deletable = false } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Text must be a non-empty string' });
    }

    console.log(`Writing text to Walrus: ${text.length} characters for ${epochs} epochs`);

    // Build query parameters
    const params = new URLSearchParams();
    if (epochs > 1) {
      params.append('epochs', epochs.toString());
    }
    if (deletable) {
      params.append('deletable', 'true');
    }

    // Make HTTP PUT request to Walrus publisher
    const walrusUrl = `${WALRUS_PUBLISHER}/v1/blobs${params.toString() ? '?' + params.toString() : ''}`;
    
    const response = await fetch(walrusUrl, {
      method: 'PUT',
      headers: {
        'Content-Type': 'text/plain',
      },
      body: text,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Walrus publisher error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const result = await response.json();
    console.log('Text written successfully to Walrus:', result);
    
    // Extract blob ID from response
    let blobId;
    let blobObject;
    
    if (result.newlyCreated) {
      blobId = result.newlyCreated.blobObject.blobId;
      blobObject = result.newlyCreated.blobObject.id;
    } else if (result.alreadyCertified) {
      blobId = result.alreadyCertified.blobId;
      blobObject = null; // Not provided in alreadyCertified response
    } else {
      throw new Error('Unexpected response format from Walrus publisher');
    }

    res.status(200).json({
      success: true,
      blobId,
      blobObject,
      result, // Include full response for debugging
    });

  } catch (error) {
    console.error('Failed to write text to Walrus:', error);
    
    // Handle specific network errors
    if (error.message.includes('Failed to fetch') || 
        error.message.includes('ERR_CERT_DATE_INVALID') ||
        error.message.includes('ERR_CONNECTION_REFUSED')) {
      return res.status(500).json({ 
        error: 'Network connectivity issues with Walrus publisher. Please try again later.' 
      });
    }
    
    res.status(500).json({ error: error.message });
  }
} 