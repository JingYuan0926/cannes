// Walrus HTTP API utilities
const WALRUS_AGGREGATOR = process.env.NEXT_PUBLIC_WALRUS_AGGREGATOR_URL || 'https://aggregator.walrus-testnet.walrus.space';
const WALRUS_PUBLISHER = process.env.NEXT_PUBLIC_WALRUS_PUBLISHER_URL || 'https://publisher.walrus-testnet.walrus.space';

/**
 * Write Text to Walrus via API Route (writeTW)
 * Uses the Next.js API route which proxies to Walrus publisher
 * @param {string} text - Text to store in Walrus
 * @param {Object} options - Storage options
 * @param {number} options.epochs - Number of epochs to store for (default: 1)
 * @param {boolean} options.deletable - Whether blob should be deletable (default: false)
 * @returns {Promise<{blobId: string, blobObject: string}>} - Blob ID and object reference
 */
export const writeTW = async (text, options = {}) => {
  if (!text || typeof text !== 'string') {
    throw new Error('Text must be a non-empty string');
  }

  const { epochs = 1, deletable = false } = options;

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
 * Write Text to Walrus via Direct HTTP API (writeTWDirect)
 * Makes a direct HTTP request to a Walrus publisher
 * @param {string} text - Text to store in Walrus
 * @param {Object} options - Storage options
 * @param {number} options.epochs - Number of epochs to store for (default: 1)
 * @param {boolean} options.deletable - Whether blob should be deletable (default: false)
 * @param {string} options.publisherUrl - Custom publisher URL (optional)
 * @returns {Promise<{blobId: string, blobObject: string}>} - Blob ID and object reference
 */
export const writeTWDirect = async (text, options = {}) => {
  if (!text || typeof text !== 'string') {
    throw new Error('Text must be a non-empty string');
  }

  const { epochs = 1, deletable = false, publisherUrl = WALRUS_PUBLISHER } = options;

  try {
    console.log(`Writing text to Walrus directly: ${text.length} characters for ${epochs} epochs`);

    // Build query parameters
    const params = new URLSearchParams();
    if (epochs > 1) {
      params.append('epochs', epochs.toString());
    }
    if (deletable) {
      params.append('deletable', 'true');
    }

    // Make HTTP PUT request to Walrus publisher
    const walrusUrl = `${publisherUrl}/v1/blobs${params.toString() ? '?' + params.toString() : ''}`;
    
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

    return {
      blobId,
      blobObject,
    };
  } catch (error) {
    console.error('Failed to write text to Walrus:', error);
    throw error;
  }
};

/**
 * Read Text from Walrus via HTTP API
 * @param {string} blobId - The blob ID to read
 * @param {Object} options - Read options
 * @param {string} options.aggregatorUrl - Custom aggregator URL (optional)
 * @returns {Promise<string>} - The text content
 */
export const readTW = async (blobId, options = {}) => {
  if (!blobId || typeof blobId !== 'string') {
    throw new Error('Blob ID must be a non-empty string');
  }

  const { aggregatorUrl = WALRUS_AGGREGATOR } = options;

  try {
    console.log(`Reading text from Walrus: ${blobId}`);

    const response = await fetch(`${aggregatorUrl}/v1/blobs/${blobId}`);

    if (!response.ok) {
      throw new Error(`Failed to read from Walrus: ${response.status} ${response.statusText}`);
    }

    const text = await response.text();
    console.log(`Text read successfully from Walrus: ${text.length} characters`);
    return text;
  } catch (error) {
    console.error('Failed to read text from Walrus:', error);
    throw error;
  }
};

/**
 * Get available Walrus testnet publishers
 * @returns {string[]} - Array of publisher URLs
 */
export const getTestnetPublishers = () => [
  'https://publisher.testnet.walrus.atalma.io',
  'https://publisher.walrus-01.tududes.com',
  'https://publisher.walrus-testnet.walrus.space',
  'https://publisher.walrus.banansen.dev',
  'https://sm1-walrus-testnet-publisher.stakesquid.com',
  'https://sui-walrus-testnet-publisher.bwarelabs.com',
  'https://suiftly-testnet-pub.mhax.io',
  'https://testnet-publisher-walrus.kiliglab.io',
  'https://testnet-publisher.walrus.graphyte.dev',
  'https://testnet.publisher.walrus.silentvalidator.com',
  'https://wal-publisher-testnet.staketab.org',
];

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