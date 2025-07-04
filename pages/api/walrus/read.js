import { getFullnodeUrl, SuiClient } from '@mysten/sui/client';
import { WalrusClient, RetryableWalrusClientError } from '@mysten/walrus';

// Global client instances for reading
let suiClient;
let walrusClient;
let initialized = false;

/**
 * Initialize Walrus clients for reading
 */
const initializeWalrusRead = async () => {
  if (initialized) return;
  
  // Create SUI client with Walrus extension
  suiClient = new SuiClient({
    url: getFullnodeUrl('testnet'),
    network: 'testnet',
  }).$extend(
    WalrusClient.experimental_asClientExtension({
      storageNodeClientOptions: {
        timeout: 60_000,
      },
    }),
  );
  
  walrusClient = suiClient.walrus;
  initialized = true;
  
  console.log('Walrus read client initialized successfully');
};

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { blobId } = req.query;

    if (!blobId || typeof blobId !== 'string') {
      return res.status(400).json({ error: 'Blob ID must be a non-empty string' });
    }

    // Initialize if not already done
    await initializeWalrusRead();

    console.log('Reading from Walrus:', blobId);
    
    // Read blob using Walrus client
    const blob = await walrusClient.readBlob({ blobId });
    
    // Convert Uint8Array back to text
    const text = new TextDecoder().decode(blob);
    
    console.log('Text read successfully from Walrus:', text.length, 'characters');
    
    res.status(200).json({
      success: true,
      text: text,
      blobId: blobId,
      length: text.length,
      bytes: blob.length,
    });

  } catch (error) {
    console.error('Failed to read from Walrus:', error);
    
    // Handle specific network/certificate errors
    if (error.message.includes('Failed to fetch') || 
        error.message.includes('ERR_CERT_DATE_INVALID') ||
        error.message.includes('ERR_CONNECTION_REFUSED')) {
      return res.status(500).json({ 
        error: 'Network connectivity issues with Walrus storage nodes. This may be due to SSL certificate problems or CORS restrictions.' 
      });
    }
    
    if (error instanceof RetryableWalrusClientError) {
      return res.status(500).json({ 
        error: 'Temporary network issue with Walrus nodes. Please try again.' 
      });
    }
    
    res.status(500).json({ error: error.message });
  }
} 