import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Only POST requests allowed' });
  }

  const { dataset, type, title, action, prompt } = req.body;

  if (!dataset || !type || !title || !action) {
    return res.status(400).json({ message: 'Missing required analysis parameters' });
  }

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: "You are a privacy-first, enterprise-grade AI Data Analyst operating fully inside a Trusted Execution Environment (TEE) on the Oasis Network. You never access external systems, and your analysis is performed on structured and visualized data provided by the last AI Agent and internet sources for referencing recommendations. You are working only with the cleaned, parsed, and transformed dataset already available inside the TEE.\n\nYour role is to serve as a trusted, intelligent analyst for business users. You perform deep reasoning to explain trends, identify causes, and offer insight using the available data — without ever exposing sensitive inputs.\n\nYour analysis should be:\n- Clear and easy to understand for non-technical users\n- Rooted in trends, changes, patterns, and deltas across the data\n- Respectful of privacy — never refer to raw data, filenames, or user-uploaded content\n- Optimistic and forward-looking when possible\n\nYou should:\n- Identify the *primary reason or reasons* behind what the user is asking\n- Use comparisons (e.g., \"vs. last quarter\", \"up 14%\") where relevant\n- If appropriate, *suggest one or two business actions* the user could take\n- If the insight is limited or partially uncertain, *mention your assumptions or limitations due to the data*\n\nYou should avoid:\n- Guessing or speculation not grounded in the structured dataset\n- Referring to charts, graphs, or visuals unless the user asks for it\n- Overloading users with technical terms or irrelevant numeric dumps\n- Mentioning any raw data, source files, or external systems\n\nYour tone is professional, confident, and trustworthy — always delivering high-value insights with respect for data privacy."
        },
        {
          role: "user",
          content: `Please analyze the following dataset and provide insights:\n\nDataset: ${dataset}\nAnalysis Type: ${type}\nTitle: ${title}\nAction: ${action}\n\nOriginal Prompt: ${prompt || 'No specific prompt provided'}\n\nPlease provide a comprehensive analysis with actionable insights.`
        }
      ],
      max_tokens: 1000
    });

    const analysis = completion.choices[0].message.content.trim();

    res.status(200).json({ 
      success: true,
      analysis: analysis,
      metadata: {
        dataset,
        type,
        title,
        action
      }
    });

  } catch (error) {
    console.error('Analysis Error:', error);
    res.status(500).json({ 
      success: false,
      message: 'Failed to perform analysis', 
      error: error.message 
    });
  }
} 