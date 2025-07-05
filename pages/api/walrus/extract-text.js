import pdfParse from 'pdf-parse';
import mammoth from 'mammoth';

const WALRUS_AGGREGATOR = process.env.NEXT_PUBLIC_WALRUS_AGGREGATOR_URL || 'https://aggregator.walrus-testnet.walrus.space';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { blobId, fileName, fileType } = req.body;

  if (!blobId) {
    return res.status(400).json({ error: 'Blob ID is required' });
  }

  try {
    console.log(`Extracting text from ${fileName} (${fileType})`);
    
    // Fetch the file from Walrus
    const response = await fetch(`${WALRUS_AGGREGATOR}/v1/blobs/${encodeURIComponent(blobId)}`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch file from Walrus: ${response.status}`);
    }

    const buffer = await response.arrayBuffer();
    const dataBuffer = Buffer.from(buffer);

    let extractedText = '';

    // Determine file type and extract text accordingly
    if (fileType?.includes('pdf') || fileName?.toLowerCase().endsWith('.pdf')) {
      console.log('Extracting text from PDF');
      const pdfData = await pdfParse(dataBuffer);
      extractedText = pdfData.text;
    } else if (fileType?.includes('wordprocessingml.document') || fileName?.toLowerCase().endsWith('.docx')) {
      console.log('Extracting text from DOCX');
      const docxData = await mammoth.extractRawText({ buffer: dataBuffer });
      extractedText = docxData.value;
    } else if (fileType?.startsWith('text/') || fileName?.toLowerCase().endsWith('.txt') || fileName?.toLowerCase().endsWith('.csv')) {
      console.log('Reading as text file');
      extractedText = dataBuffer.toString('utf-8');
    } else {
      console.log('Unsupported file type, returning as binary');
      return res.status(400).json({ error: 'Unsupported file type for text extraction' });
    }

    console.log(`Text extracted successfully: ${extractedText.length} characters`);
    
    res.status(200).json({ 
      text: extractedText,
      length: extractedText.length,
      blobId,
      fileName
    });

  } catch (error) {
    console.error('Error extracting text:', error);
    res.status(500).json({ error: 'Failed to extract text from file' });
  }
} 