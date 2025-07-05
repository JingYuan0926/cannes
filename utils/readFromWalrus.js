// Walrus HTTP API utilities for reading
const WALRUS_AGGREGATOR = process.env.NEXT_PUBLIC_WALRUS_AGGREGATOR_URL || 'https://aggregator.walrus-testnet.walrus.space';

/**
 * Read From Walrus via API Route (readFW)
 * Uses the Next.js API route which proxies to Walrus aggregator
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