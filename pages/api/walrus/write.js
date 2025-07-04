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
      },
    }),
  );
  
  walrusClient = suiClient.walrus;
  initialized = true;
  
  console.log('Walrus write client initialized successfully');
  console.log('Signer address:', keypair.toSuiAddress());
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { text, epochs = 3, deletable = false } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Text must be a non-empty string' });
    }

    // Initialize if not already done
    await initializeWalrusWrite();

    // Convert text to Uint8Array
    const blob = new TextEncoder().encode(text);

    console.log(`Writing text to Walrus: ${text.length} characters (${blob.length} bytes) for ${epochs} epochs`);

    // Write blob using Walrus client
    const result = await walrusClient.writeBlob({
      blob,
      deletable,
      epochs,
      signer: keypair,
    });

    console.log('Text written successfully to Walrus:', result.blobId);
    
    res.status(200).json({
      success: true,
      blobId: result.blobId,
      blobObject: result.blobObject,
      signerAddress: keypair.toSuiAddress(),
    });

  } catch (error) {
    console.error('Failed to write text to Walrus:', error);
    
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