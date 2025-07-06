
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
    const payload = {
  "icon": "https://cdn-icons-png.flaticon.com/512/3500/3500833.png",
  "label": "unspecified",
  "title": "Overview of Data Insights",
  "description": "Analysis Type: summary\nDetails: Provide a general overview of insights from the data",
  "links": {
    "actions": [
      {
        "label": "Run Analysis",
        "href": "/api/runDataAnalysis"
      }
    ]
  }
};
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
