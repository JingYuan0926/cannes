import { getFullnodeUrl, SuiClient } from '@mysten/sui/client';
import { WalrusClient, RetryableWalrusClientError } from '@mysten/walrus';
import { Ed25519Keypair } from '@mysten/sui/keypairs/ed25519';

// Global client instances for writing
let suiClient;
let walrusClient;
let keypair;
let initialized = false;

/**
 * Initialize Walrus clients and keypair for writing
 */
const initializeWalrusWrite = async () => {
  if (initialized) return;
  
  if (typeof window !== 'undefined') {
    // Get seed phrase from environment variable
    const seedPhrase = process.env.NEXT_PUBLIC_WALRUS_SEED_PHRASE;
    if (!seedPhrase) {
      throw new Error('NEXT_PUBLIC_WALRUS_SEED_PHRASE environment variable is required');
    }

    // Create keypair from seed phrase
    keypair = Ed25519Keypair.deriveKeypair(seedPhrase);

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
    
    console.log('Walrus write client initialized successfully');
    console.log('Signer address:', keypair.toSuiAddress());
  }
};

/**
 * Write Text to Walrus (writeTW) - API Route Version
 * @param {string} text - Text to store in Walrus
 * @param {Object} options - Storage options
 * @param {number} options.epochs - Number of epochs to store for (default: 3)
 * @param {boolean} options.deletable - Whether blob should be deletable (default: false)
 * @returns {Promise<{blobId: string, blobObject: string}>} - Blob ID and object reference
 */
export const writeTW = async (text, options = {}) => {
  if (!text || typeof text !== 'string') {
    throw new Error('Text must be a non-empty string');
  }

  const { epochs = 3, deletable = false } = options;

  try {
    console.log(`Writing text to Walrus via API: ${text.length} characters for ${epochs} epochs`);

    const response = await fetch('/api/walrus/write', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        epochs,
        deletable,
      }),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Failed to write to Walrus');
    }

    console.log('Text written successfully to Walrus:', result.blobId);
    return {
      blobId: result.blobId,
      blobObject: result.blobObject,
    };
  } catch (error) {
    console.error('Failed to write text to Walrus:', error);
    throw error;
  }
};

/**
 * Get the current signer address - API Route Version
 * @returns {Promise<string>} - The signer's SUI address
 */
export const getSignerAddress = async () => {
  try {
    const response = await fetch('/api/walrus/signer');
    const result = await response.json();
    
    if (response.ok && result.signerAddress) {
      return result.signerAddress;
    }
    
    throw new Error(result.error || 'Failed to get signer address');
  } catch (error) {
    console.error('Failed to get signer address:', error);
    return 'Unable to retrieve signer address';
  }
}; 