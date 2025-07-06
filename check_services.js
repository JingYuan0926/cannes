#!/usr/bin/env node

const http = require('http');

const SERVICES = {
  etl: { url: 'http://localhost:3030', name: 'ETL Service' },
  preprocessing: { url: 'http://localhost:3031', name: 'Preprocessing Service' },
  eda: { url: 'http://localhost:3035', name: 'EDA Service' },
  analysis: { url: 'http://localhost:3040', name: 'ML Analysis Service' }
};

function checkService(serviceUrl, serviceName) {
  return new Promise((resolve) => {
    const url = new URL(serviceUrl);
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: '/',
      method: 'GET',
      timeout: 5000
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        try {
          const jsonData = JSON.parse(data);
          console.log(`✅ ${serviceName} - HEALTHY`);
          console.log(`   Status: ${jsonData.status}`);
          console.log(`   Service: ${jsonData.service || 'N/A'}`);
          console.log(`   URL: ${serviceUrl}`);
          resolve(true);
        } catch (error) {
          console.log(`❌ ${serviceName} - UNHEALTHY (Invalid JSON response)`);
          resolve(false);
        }
      });
    });

    req.on('error', (error) => {
      console.log(`❌ ${serviceName} - UNAVAILABLE`);
      console.log(`   Error: ${error.message}`);
      console.log(`   URL: ${serviceUrl}`);
      resolve(false);
    });

    req.on('timeout', () => {
      console.log(`❌ ${serviceName} - TIMEOUT`);
      console.log(`   URL: ${serviceUrl}`);
      req.destroy();
      resolve(false);
    });

    req.setTimeout(5000);
    req.end();
  });
}

async function checkAllServices() {
  console.log('🔍 Checking Data Analysis Pipeline Services...\n');
  
  const results = [];
  
  for (const [key, service] of Object.entries(SERVICES)) {
    const isHealthy = await checkService(service.url, service.name);
    results.push({ name: service.name, healthy: isHealthy });
    console.log('');
  }
  
  console.log('📊 Summary:');
  const healthyCount = results.filter(r => r.healthy).length;
  const totalCount = results.length;
  
  console.log(`   Healthy Services: ${healthyCount}/${totalCount}`);
  
  if (healthyCount === totalCount) {
    console.log('🎉 All services are running! You can start using the pipeline.');
  } else {
    console.log('⚠️  Some services are not running. Please start them before using the pipeline.');
    console.log('\n💡 To start services:');
    results.filter(r => !r.healthy).forEach(service => {
      const key = Object.keys(SERVICES).find(k => SERVICES[k].name === service.name);
      console.log(`   • ${service.name}: cd ${key} && python app.py`);
    });
  }
}

// Run the check
checkAllServices().catch(console.error); 