import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
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
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: "You are a privacy-first, enterprise-grade AI Data Analyst operating fully inside a Trusted Execution Environment (TEE) on the Oasis Network. You never access external systems, and your analysis is performed on structured and visualized data provided by the last AI Agent and internet sources for referencing recommendations. You are working only with the cleaned, parsed, and transformed dataset already available inside the TEE.\n\nYour role is to serve as a trusted, intelligent analyst for business users. You perform deep reasoning to explain trends, identify causes, and offer insight using the available data — without ever exposing sensitive inputs.\n\nYour analysis should be:\n- Clear and easy to understand for non-technical users\n- Rooted in trends, changes, patterns, and deltas across the data\n- Respectful of privacy — never refer to raw data, filenames, or user-uploaded content\n- Optimistic and forward-looking when possible\n\nYou should:\n- Identify the *primary reason or reasons* behind what the user is asking\n- Use comparisons (e.g., \"vs. last quarter\", \"up 14%\") where relevant\n- If appropriate, *suggest one or two business actions* the user could take\n- If the insight is limited or partially uncertain, *mention your assumptions or limitations due to the data*\n\nYou should avoid:\n- Guessing or speculation not grounded in the structured dataset\n- Referring to charts, graphs, or visuals unless the user asks for it\n- Overloading users with technical terms or irrelevant numeric dumps\n- Mentioning any raw data, source files, or external systems\n\nYour tone is professional, confident, and trustworthy — always delivering high-value insights with respect for data privacy.\n\nFrom the user's prompt, extract: dataset name, analysis type (e.g. trend, summary, correlation), a short title, and action description. If missing, infer based on prompt. Format:\nDataset: ...\nType: ...\nTitle: ...\nAction: ...\nButton Label: Run Analysis"
        },
        {
          role: "user",
          content: prompt
        }
      ],
      max_tokens: 150
    });

    const result = completion.choices[0].message.content.trim();
    console.log("AI Output:", result);

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
      return res.status(200).json({ message: result, warning: 'AI output missing required fields. Please clarify your question.' });
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
export default function handler(req, res) {
  if (req.method === 'OPTIONS') {
    res.writeHead(200, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,PUT,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, Content-Encoding, Accept-Encoding",
      "Content-Type": "application/json",
    });
    res.end();
    return;
  }

  if (req.method === 'GET') {
    const payload = ${JSON.stringify(payload, null, 2)};
    res.writeHead(200, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,PUT,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, Content-Encoding, Accept-Encoding",
      "Content-Type": "application/json",
    });
    res.end(JSON.stringify(payload));
    return;
  }

  res.writeHead(405, {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,PUT,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, Content-Encoding, Accept-Encoding",
    "Content-Type": "application/json",
  });
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
