import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';

const openai = new OpenAI({
  apiKey: process.env.NEXT_PUBLIC_API_URL
});

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Only POST requests allowed' });
  }

  const { prompt, walletAddress } = req.body;

  if (!prompt || !walletAddress) {
    return res.status(400).json({ message: 'Prompt and wallet address are required' });
  }

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are a data analyst. From the user's prompt, extract: dataset name, analysis type (e.g. trend, summary, correlation), a short title, and action description. If missing, infer based on prompt. Format:\nDataset: ...\nType: ...\nTitle: ...\nAction: ...\nButton Label: Run Analysis"
        },
        {
          role: "user",
          content: prompt
        }
      ],
      max_tokens: 150
    });

    const result = completion.choices[0].message.content.trim();

    const extracted = {
      dataset: '',
      type: '',
      title: '',
      action: '',
      buttonLabel: ''
    };

    result.split('\n').forEach(line => {
      const [key, ...val] = line.split(':');
      const value = val.join(':').trim();
      const lowerKey = key.toLowerCase().trim();
      if (lowerKey === 'dataset') extracted.dataset = value;
      else if (lowerKey === 'type') extracted.type = value;
      else if (lowerKey === 'title') extracted.title = value;
      else if (lowerKey === 'action') extracted.action = value;
      else if (lowerKey === 'button label') extracted.buttonLabel = value;
    });

    if (!extracted.dataset || !extracted.type || !extracted.title || !extracted.action || !extracted.buttonLabel) {
      throw new Error('Missing fields from AI output');
    }

    const payload = {
      icon: "https://cdn-icons-png.flaticon.com/512/3500/3500833.png",
      label: extracted.dataset,
      title: extracted.title,
      description: `Analysis Type: ${extracted.type}\nDetails: ${extracted.action}`,
      links: {
        actions: [
          {
            label: extracted.buttonLabel,
            href: "/api/runDataAnalysis"
          }
        ]
      }
    };

    const folderPath = path.join(process.cwd(), 'pages/api/actions', walletAddress);
    if (!fs.existsSync(folderPath)) fs.mkdirSync(folderPath, { recursive: true });

    const filePath = path.join(folderPath, 'runAnalysis.js');
    const fileContent = `
import { ActionGetResponse } from "@solana/actions";

const ACTIONS_CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,PUT,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, Content-Encoding, Accept-Encoding",
  "Content-Type": "application/json",
};

export default function handler(req, res) {
  if (req.method === 'OPTIONS') {
    res.writeHead(200, ACTIONS_CORS_HEADERS);
    res.end();
    return;
  }

  if (req.method === 'GET') {
    const payload = ${JSON.stringify(payload, null, 2)};
    res.writeHead(200, ACTIONS_CORS_HEADERS);
    res.end(JSON.stringify(payload));
    return;
  }

  res.writeHead(405, ACTIONS_CORS_HEADERS);
  res.end(JSON.stringify({ error: 'Method Not Allowed' }));
}
`;
    fs.writeFileSync(filePath, fileContent);

    const actionFilePath = path.join(process.cwd(), 'public', 'actions.json');
    let actionFileContent = {};

    if (fs.existsSync(actionFilePath)) {
      const existing = fs.readFileSync(actionFilePath, 'utf8');
      try {
        actionFileContent = JSON.parse(existing);
      } catch (err) {
        console.error('Failed to parse actions.json:', err);
      }
    }

    if (!actionFileContent.rules) actionFileContent.rules = [];

    actionFileContent.rules.push({
      pathPattern: `/api/actions/${walletAddress}/runAnalysis`,
      apiPath: `/api/actions/${walletAddress}/runAnalysis`
    });

    fs.writeFileSync(actionFilePath, JSON.stringify(actionFileContent, null, 2));

    res.status(200).json({ message: 'Data analysis action created successfully.' });

  } catch (error) {
    console.error('Chatbot Error:', error);
    res.status(500).json({ message: 'Failed to process request', error: error.message });
  }
}
