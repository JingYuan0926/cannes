/**
 * Encryption utilities for securing files before Walrus upload
 * Uses AES-GCM for authenticated encryption with Web Crypto API
 */

/**
 * Generate a random encryption key
 * @returns {Promise<CryptoKey>} - Generated AES-GCM key
 */
export const generateEncryptionKey = async () => {
  return await crypto.subtle.generateKey(
    {
      name: 'AES-GCM',
      length: 256,
    },
    true, // extractable
    ['encrypt', 'decrypt']
  );
};

/**
 * Export encryption key to a format that can be stored
 * @param {CryptoKey} key - The encryption key to export
 * @returns {Promise<ArrayBuffer>} - Exported key data
 */
export const exportKey = async (key) => {
  return await crypto.subtle.exportKey('raw', key);
};

/**
 * Import encryption key from stored data
 * @param {ArrayBuffer} keyData - The key data to import
 * @returns {Promise<CryptoKey>} - Imported encryption key
 */
export const importKey = async (keyData) => {
  return await crypto.subtle.importKey(
    'raw',
    keyData,
    {
      name: 'AES-GCM',
      length: 256,
    },
    true,
    ['encrypt', 'decrypt']
  );
};

/**
 * Generate a random initialization vector (IV)
 * @returns {Uint8Array} - Random IV
 */
export const generateIV = () => {
  return crypto.getRandomValues(new Uint8Array(12)); // 96-bit IV for AES-GCM
};

/**
 * Encrypt a file using AES-GCM
 * @param {File|ArrayBuffer} file - File or data to encrypt
 * @param {CryptoKey} key - Encryption key
 * @returns {Promise<{encryptedData: ArrayBuffer, iv: Uint8Array}>} - Encrypted data and IV
 */
export const encryptFile = async (file, key) => {
  try {
    // Convert file to ArrayBuffer if it's a File object
    let dataBuffer;
    if (file instanceof File) {
      dataBuffer = await file.arrayBuffer();
    } else if (file instanceof ArrayBuffer) {
      dataBuffer = file;
    } else {
      throw new Error('Input must be a File object or ArrayBuffer');
    }

    // Generate random IV
    const iv = generateIV();

    // Encrypt the data
    const encryptedData = await crypto.subtle.encrypt(
      {
        name: 'AES-GCM',
        iv: iv,
      },
      key,
      dataBuffer
    );

    return {
      encryptedData,
      iv,
    };
  } catch (error) {
    console.error('Encryption failed:', error);
    throw new Error(`Encryption failed: ${error.message}`);
  }
};

/**
 * Decrypt data using AES-GCM
 * @param {ArrayBuffer} encryptedData - Encrypted data
 * @param {Uint8Array} iv - Initialization vector used for encryption
 * @param {CryptoKey} key - Decryption key
 * @returns {Promise<ArrayBuffer>} - Decrypted data
 */
export const decryptData = async (encryptedData, iv, key) => {
  try {
    const decryptedData = await crypto.subtle.decrypt(
      {
        name: 'AES-GCM',
        iv: iv,
      },
      key,
      encryptedData
    );

    return decryptedData;
  } catch (error) {
    console.error('Decryption failed:', error);
    throw new Error(`Decryption failed: ${error.message}`);
  }
};

/**
 * Create an encrypted file from original file
 * @param {File} file - Original file to encrypt
 * @param {CryptoKey} key - Encryption key
 * @returns {Promise<{encryptedFile: File, iv: Uint8Array, metadata: object}>} - Encrypted file and metadata
 */
export const createEncryptedFile = async (file, key) => {
  try {
    const { encryptedData, iv } = await encryptFile(file, key);
    
    // Create metadata
    const metadata = {
      originalName: file.name,
      originalSize: file.size,
      originalType: file.type,
      timestamp: new Date().toISOString(),
    };

    // Create encrypted file blob
    const encryptedBlob = new Blob([encryptedData], { type: 'application/octet-stream' });
    const encryptedFile = new File([encryptedBlob], `encrypted_${file.name}`, {
      type: 'application/octet-stream',
    });

    return {
      encryptedFile,
      iv,
      metadata,
    };
  } catch (error) {
    console.error('Failed to create encrypted file:', error);
    throw error;
  }
};

/**
 * Get or create user's encryption key from localStorage
 * @returns {Promise<CryptoKey>} - User's encryption key
 */
export const getUserEncryptionKey = async () => {
  try {
    // Check if key exists in localStorage
    const storedKeyData = localStorage.getItem('userEncryptionKey');
    
    if (storedKeyData) {
      // Import existing key
      const keyData = new Uint8Array(JSON.parse(storedKeyData));
      return await importKey(keyData.buffer);
    } else {
      // Generate new key
      const newKey = await generateEncryptionKey();
      const keyData = await exportKey(newKey);
      
      // Store key in localStorage
      localStorage.setItem('userEncryptionKey', JSON.stringify(Array.from(new Uint8Array(keyData))));
      
      return newKey;
    }
  } catch (error) {
    console.error('Failed to get/create user encryption key:', error);
    throw error;
  }
};

/**
 * Convert ArrayBuffer to Base64 string
 * @param {ArrayBuffer} buffer - Buffer to convert
 * @returns {string} - Base64 encoded string
 */
export const arrayBufferToBase64 = (buffer) => {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

/**
 * Convert Base64 string to ArrayBuffer
 * @param {string} base64 - Base64 encoded string
 * @returns {ArrayBuffer} - Decoded buffer
 */
export const base64ToArrayBuffer = (base64) => {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
};

/**
 * Convert Uint8Array to Base64 string
 * @param {Uint8Array} uint8Array - Array to convert
 * @returns {string} - Base64 encoded string
 */
export const uint8ArrayToBase64 = (uint8Array) => {
  return arrayBufferToBase64(uint8Array.buffer);
};

/**
 * Convert Base64 string to Uint8Array
 * @param {string} base64 - Base64 encoded string
 * @returns {Uint8Array} - Decoded array
 */
export const base64ToUint8Array = (base64) => {
  return new Uint8Array(base64ToArrayBuffer(base64));
}; 