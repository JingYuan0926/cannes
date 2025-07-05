import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';
import multer from 'multer';
import pdfParse from 'pdf-parse';
import mammoth from 'mammoth';
import { v4 as uuidv4 } from 'uuid';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const upload = multer({
  dest: '/tmp',
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Only PDF and DOCX files are allowed.'));
    }
  }
});

export const config = {
  api: {
    bodyParser: false,
  },
};

function runMiddleware(req, res, fn) {
  return new Promise((resolve, reject) => {
    fn(req, res, (result) => {
      if (result instanceof Error) {
        return reject(result);
      }
      return resolve(result);
    });
  });
}

export default async function handler(req, res) {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ message: 'OpenAI API key is missing.' });
    }

    if (req.method !== 'POST') {
      return res.status(405).json({ message: 'Only POST requests allowed' });
    }

    // Handle file upload (multipart/form-data)
    if (req.headers['content-type'] && req.headers['content-type'].includes('multipart/form-data')) {
      let sessionId = req.headers['x-session-id'] || uuidv4();
      const sampleDataPath = path.join('/tmp', `sample_data_${sessionId}.txt`);
      
      await runMiddleware(req, res, upload.array('files'));
      const files = req.files || [];
      let extractedText = '';

      for (const file of files) {
        try {
          const dataBuffer = fs.readFileSync(file.path);
          if (file.mimetype === 'application/pdf') {
            const pdfData = await pdfParse(dataBuffer);
            extractedText += '\n' + pdfData.text;
          } else if (file.mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
            const docxData = await mammoth.extractRawText({ buffer: dataBuffer });
            extractedText += '\n' + docxData.value;
          }
        } catch (err) {
          console.error('File parse error:', err);
        } finally {
          fs.unlinkSync(file.path);
        }
      }

      if (extractedText.trim()) {
        fs.writeFileSync(sampleDataPath, extractedText);
        return res.status(200).json({ message: 'Analysis Finished!', sessionId });
      } else {
        return res.status(400).json({ message: 'Uploaded file had no readable content.' });
      }
    }

    // Parse JSON body for non-file upload requests
    let body = {};
    if (req.headers['content-type'] && req.headers['content-type'].includes('application/json')) {
      try {
        // Read the raw body data
        const chunks = [];
        for await (const chunk of req) {
          chunks.push(chunk);
        }
        const rawBody = Buffer.concat(chunks).toString('utf8');
        body = JSON.parse(rawBody);
      } catch (err) {
        console.error('JSON parse error:', err);
        return res.status(400).json({ message: 'Invalid JSON in request body' });
      }
    }
    
    const { prompt, sampleDataReady, conversation, sessionId: bodySessionId } = body;
    
    // Log the values for debugging
    console.log('Received sessionId from body:', bodySessionId);
    console.log('sampleDataReady:', sampleDataReady);
    
    let sampleData = '';
    let sessionId = bodySessionId || req.headers['x-session-id'] || uuidv4();
    const sampleDataPath = path.join('/tmp', `sample_data_${sessionId}.txt`);
    
    console.log('Resolved sampleDataPath:', sampleDataPath);
    console.log('File exists?', fs.existsSync(sampleDataPath));

    if (sampleDataReady) {
      try {
        if (fs.existsSync(sampleDataPath)) {
          sampleData = fs.readFileSync(sampleDataPath, 'utf8');
        }
      } catch (err) {
        console.error('Sample data read error:', err);
      }
    }

    const messages = [
      {
        role: 'system',
        content:    
            `You are a privacy-first, enterprise-grade AI Data Analyst. 
            You ALWAYS use the uploaded sample data provided by the user to answer their questions. 
            If the user asks about trends, predictions, or insights, base your answer strictly on the uploaded data. 
            If the data is insufficient, politely explain what is missing.
            Your analysis should be: Clear and easy to understand for non-technical users
            ;Rooted in trends, changes, patterns, and deltas across the data
            ;Respectful of privacy — never refer to raw data, filenames, or user-uploaded content
            ;Optimistic and forward-looking when possible.
            You should: Identify the *primary reason or reasons* behind what the user is asking
            ;Use comparisons (e.g., \"vs. last quarter\", \"up 14%\") where relevant
            ;If appropriate, *suggest one or two business actions* the user could take
            ;If the insight is limited or partially uncertain, *mention your assumptions or limitations due to the data
            You should avoid: Guessing or speculation not grounded in the structured dataset
            ;Referring to charts, graphs, or visuals unless the user asks for it
            ;Overloading users with technical terms or irrelevant numeric dumps
            ;Mentioning any raw data, source files, or external systems.
            Your tone is professional, confident, and trustworthy — always delivering high-value insights with respect for data privacy.
            Here is the data you should use to answer my questions:\n\n${sampleData}`
      }
    ];

    if (sampleData) {
      messages.push({ role: 'user', content: `Here is the data you should use to answer my questions:\n\n${sampleData}` });
    }
    if (Array.isArray(conversation)) {
      for (const msg of conversation) {
        if (msg && msg.role && msg.content) {
          messages.push({ role: msg.role, content: msg.content });
        }
      }
    }
    if (prompt) {
      messages.push({ role: 'user', content: prompt });
    }

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      max_tokens: 1500
    });

    const aiMessage = completion.choices[0].message.content.trim();
    res.status(200).json({ message: aiMessage });

  } catch (error) {
    console.error('Chatbot Error:', error);
    res.status(500).json({ message: 'Failed to process request', error: error.message });
  }
}
