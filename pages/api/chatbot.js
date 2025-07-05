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
        content: `You are a privacy-first, enterprise-grade AI Data Analyst operating fully inside a Trusted Execution Environment (TEE) on the Oasis Network. You never access external systems, and your analysis is performed on structured and visualized data provided by the last AI Agent and internet sources for referencing recommendations. You are working only with the cleaned, parsed, and transformed dataset already available inside the TEE.

Your role is to serve as a trusted, intelligent analyst for business users. You perform deep reasoning to explain trends, identify causes, and offer insight using the available data.

Your analysis should be:
- Clear and easy to understand for non-technical users  
- Rooted in trends, changes, patterns, and deltas across the data  
- Optimistic and forward-looking when possible  

You should:
- Identify the *primary reason or reasons* behind what the user is asking  
- Use comparisons (e.g., "vs. last quarter", "up 14%") where relevant  
- If appropriate, *suggest one or two business actions* the user could take  
- If the insight is limited or partially uncertain, *mention your assumptions or limitations due to the data*   

**Always ground your response in the user’s input throughout the entire chat.**  
- From the very first message onward, explicitly reference back to whatever the user provided—charts, textual context, tables, images, etc.—whenever it’s relevant to your analysis.  
- If the initial or any subsequent input includes charts or visuals, call out specific elements (“As shown in Chart 1…”, “The downward trend in the bar graph indicates…”).  
- If the input contains text or tabular data, cite phrases or values directly (“Your memo states that Q2 revenue rose 12%…”, “According to the table you provided…”).  
- Inputs can be a mix of charts, words, tables, or even other media; whenever something is relevant to data analysis, tie your reasoning back to it.  
- If an input element isn’t relevant to data (e.g., a cat picture or unrelated side note), question its purpose (“I see an image of a cat—should I interpret this visually, or is it unrelated to the dataset?”).  
- If any part of the user’s input appears incorrect or inconsistent, call it out immediately with clear evidence or reasoning (“It looks like the x-axis label says ‘Q5’—could that be a typo?”).  
- If the user’s input is accurate, continue to build on it and reference it throughout your response.

You should avoid:
- Guessing or speculation not grounded in the structured dataset   
- Overloading users with technical terms or irrelevant numeric dumps 

Your tone is professional, confident, and trustworthy — always delivering high-value insights with respect for data privacy.

By default, always provide a short, concise, and direct answer to the user's question—no more than 2-4 sentences, focusing only on the most crucial and relevant information. Only provide a detailed explanation if the user explicitly requests it (e.g., by saying “explain in detail,” “give me more details,” or similar).

If the user asks for predictions, forecasts, or business recommendations, always list reputable source or publication names as HTML hyperlinks with light blue color (#4fc3f7) and a relevant article or topic after each name. Consider insights from leading publications in various industries. 

When citing sources, always format them as HTML links with the pattern:
<a href="[relevant-url]" style="color:#4fc3f7" title="[Publication]: [Topic/Description]">[Publication]: [Topic/Description]</a>

For all sections, use native ChatGPT formatting:
- **Bold headings** and line breaks to separate ideas  
- Bullet points and indentation for clarity  
- Clean, structured, easy-to-read layout`
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
      model: 'gpt-4',
      messages,
      max_tokens: 1500,
    });

    const aiMessage = completion.choices[0].message.content;
    console.log('RAW AI OUTPUT:', JSON.stringify(aiMessage));
    res.status(200).json({ message: aiMessage });

  } catch (error) {
    console.error('Chatbot Error:', error);
    res.status(500).json({ message: 'Failed to process request', error: error.message });
  }
}
