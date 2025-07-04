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
  
  if (typeof window !== 'undefined') {
    // Create SUI client with Walrus extension
    suiClient = new SuiClient({
      url: getFullnodeUrl('testnet'),
      network: 'testnet',
    }).$extend(
      WalrusClient.experimental_asClientExtension({
        storageNodeClientOptions: {
          timeout: 60_000,
          // Add custom fetch with better error handling
          fetch: async (url, options) => {
            try {
              const response = await fetch(url, {
                ...options,
                headers: {
                  ...options?.headers,
                  'Access-Control-Allow-Origin': '*',
                },
              });
              return response;
            } catch (error) {
              console.warn(`Fetch failed for ${url}:`, error.message);
              throw error;
            }
          },
        },
      }),
    );
    
    walrusClient = suiClient.walrus;
    initialized = true;
    
    console.log('Walrus read client initialized successfully');
  }
};

/**
 * Read From Walrus (readFW) - API Route Version
 * @param {string} blobId - The blob ID to read from Walrus
 * @returns {Promise<string>} - The text content
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
    
    console.log('Text read successfully from Walrus:', result.length, 'characters');
    return result.text;
  } catch (error) {
    console.error('Failed to read from Walrus:', error);
    throw error;
  }
}; 