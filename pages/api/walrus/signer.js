import { Ed25519Keypair } from '@mysten/sui/keypairs/ed25519';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Get seed phrase from environment variable
    const seedPhrase = process.env.NEXT_PUBLIC_WALRUS_SEED_PHRASE;
    if (!seedPhrase) {
      return res.status(500).json({ error: 'NEXT_PUBLIC_WALRUS_SEED_PHRASE environment variable is required' });
    }

    // Create keypair from seed phrase
    const keypair = Ed25519Keypair.deriveKeypair(seedPhrase);
    
    res.status(200).json({
      success: true,
      signerAddress: keypair.toSuiAddress(),
    });

  } catch (error) {
    console.error('Failed to get signer address:', error);
    res.status(500).json({ error: error.message });
  }
} 