// Simple HTTP API proxy for Walrus storage
import { IncomingForm } from 'formidable';
import fs from 'fs/promises';

const WALRUS_PUBLISHER = process.env.WALRUS_PUBLISHER_URL || 'https://publisher.walrus-testnet.walrus.space';

export const config = {
  api: {
    bodyParser: false, // Disable body parser to handle multipart/form-data
  },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const contentType = req.headers['content-type'] || '';
    let content, epochs = 1, deletable = true, filename = null;

    if (contentType.startsWith('multipart/form-data')) {
      // Handle file upload
      const form = new IncomingForm();
      const [fields, files] = await form.parse(req);
      
      const uploadedFile = Array.isArray(files.file) ? files.file[0] : files.file;
      if (!uploadedFile) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      // Read file content
      content = await fs.readFile(uploadedFile.filepath);
      filename = uploadedFile.originalFilename || uploadedFile.newFilename;
      
      // Get options from form fields
      if (fields.epochs) {
        epochs = parseInt(Array.isArray(fields.epochs) ? fields.epochs[0] : fields.epochs) || 1;
      }
      if (fields.deletable) {
        deletable = (Array.isArray(fields.deletable) ? fields.deletable[0] : fields.deletable) === 'true';
      }

      console.log(`Uploading file to Walrus: ${filename} (${content.length} bytes) for ${epochs} epochs (deletable: ${deletable})`);
    } else {
      // Handle text content (JSON)
      const body = await parseBody(req);
      const { text } = body;
      epochs = body.epochs || 1;
      deletable = body.deletable !== undefined ? body.deletable : true;

      if (!text || typeof text !== 'string') {
        return res.status(400).json({ error: 'Text must be a non-empty string' });
      }

      content = Buffer.from(text, 'utf8');
      console.log(`Writing text to Walrus: ${text.length} characters for ${epochs} epochs (deletable: ${deletable})`);
    }

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
        'Content-Type': filename ? 'application/octet-stream' : 'text/plain',
      },
      body: content,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Walrus publisher error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const result = await response.json();
    console.log('Content written successfully to Walrus:', result);
    
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
      filename: filename || null,
      size: content.length,
      result, // Include full response for debugging
    });

  } catch (error) {
    console.error('Failed to write to Walrus:', error);
    
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

// Helper function to parse JSON body
async function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        resolve(JSON.parse(body));
      } catch (error) {
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', reject);
  });
} 