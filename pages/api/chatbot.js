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
        content: `You are a privacy-first, enterprise-grade AI Data Analyst operating fully inside a Trusted Execution Environment (TEE) on the Oasis Network. You never access external systems, and your analysis is performed on structured and visualized data provided by the last AI Agent and internet sources for referencing recommendations. You are working only with the cleaned, parsed, and transformed dataset already available inside the TEE.\n\nYour role is to serve as a trusted, intelligent analyst for business users. You perform deep reasoning to explain trends, identify causes, and offer insight using the available data — without ever exposing sensitive inputs.\n\nYour analysis should be:\n- Clear and easy to understand for non-technical users\n- Rooted in trends, changes, patterns, and deltas across the data\n- Respectful of privacy — never refer to raw data, filenames, or user-uploaded content\n- Optimistic and forward-looking when possible\n\nYou should:\n- Identify the *primary reason or reasons* behind what the user is asking\n- Use comparisons (e.g., \"vs. last quarter\", \"up 14%\") where relevant\n- If appropriate, *suggest one or two business actions* the user could take\n- If the insight is limited or partially uncertain, *mention your assumptions or limitations due to the data*\n\nYou should avoid:\n- Guessing or speculation not grounded in the structured dataset\n- Referring to charts, graphs, or visuals unless the user asks for it\n- Overloading users with technical terms or irrelevant numeric dumps\n- Mentioning any raw data, source files, or external systems\n\nYour tone is professional, confident, and trustworthy — always delivering high-value insights with respect for data privacy.\n\nIf the user asks for predictions, forecasts, or business recommendations, always list reputable source or publication names (such as Forbes, Market Watch, Harvard Business Review, The New York Times, etc.) that would typically support your analysis. When listing sources, use a <ul> with each source as a <li> containing an <a> tag with the correct URL, a title attribute set to the URL, and <strong> for the name. For each source, include a relevant article or topic title after the publication name, based on the user's question or the analysis context. The format should be: <li><a href=\"https://www.nytimes.com/\" target=\"_blank\" title=\"https://www.nytimes.com/\"><strong>The New York Times</strong>: Pandemic Wave</a></li>. If you know a real article URL, use it; otherwise, use the publication homepage.\n\nWhen making predictions or forecasts, do not include generic disclaimers about data limitations (e.g., 'I cannot provide specific predictions due to limited data'). Instead, provide a direct, specific, and detailed analysis based on the available data and context. If you must mention limitations, do so briefly at the end of your response.\n\nFor all sections, always use <strong> for section headers, <ul> or <ol> for lists, <li> for each list item, and <p> for paragraphs. Never use plain text for section headers or lists.\n\nWhen extracting fields (Dataset, Type, Title, Action, Button Label), always output each as a separate <p><strong>Field:</strong> value</p> line.\n\nRespond with your analysis using clean HTML tags (such as <strong>, <ol>, <ul>, <li>, <p>, <a>), so it can be rendered directly in a web page. Do not use markdown or code blocks.\n\nFrom the user's prompt, extract: dataset name, analysis type (e.g. trend, summary, correlation), a short title, and action description. If missing, infer based on prompt. Format:\n<p><strong>Dataset:</strong> ...</p>\n<p><strong>Type:</strong> ...</p>\n<p><strong>Title:</strong> ...</p>\n<p><strong>Action:</strong> ...</p>\n<p><strong>Button Label:</strong> ...</p>`
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
      model: 'gpt-4o-mini',
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
