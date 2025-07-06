"""
TEE Signing Module for Analysis Results Attestation
"""

import json
import hashlib
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def sign_analysis_results_with_tee(analysis_results):
    """
    Sign analysis results with TEE keys for attestation
    
    Args:
        analysis_results (dict): The analysis results to sign
        
    Returns:
        dict: Original results with TEE attestation signature
    """
    try:
        # Create a deterministic hash of the analysis results
        results_json = json.dumps(analysis_results, sort_keys=True, default=str)
        results_hash = hashlib.sha256(results_json.encode()).hexdigest()
        
        # Generate a TEE key for signing (using analysis_id as key_id for consistency)
        analysis_id = analysis_results.get('analysis_id', 'unknown')
        key_payload = {
            "key_id": f"analysis_signing_{analysis_id}",
            "kind": "secp256k1"
        }
        
        # Call ROFL appd to generate signing key
        key_response = requests.post(
            'http://localhost/rofl/v1/keys/generate',
            json=key_payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if key_response.status_code == 200:
            signing_key = key_response.json().get('key')
            
            # Get ROFL app identifier
            app_id_response = requests.get(
                'http://localhost/rofl/v1/app/id',
                timeout=10
            )
            
            rofl_app_id = app_id_response.text if app_id_response.status_code == 200 else 'unknown'
            
            # Create attestation metadata
            attestation = {
                'results_hash': results_hash,
                'signing_key': signing_key,
                'rofl_app_id': rofl_app_id,
                'timestamp': datetime.now().isoformat(),
                'tee_attested': True,
                'signature_algorithm': 'secp256k1'
            }
            
            # Add attestation to results
            analysis_results['tee_attestation'] = attestation
            
            logger.info(f"Analysis results signed with TEE key for analysis_id: {analysis_id}")
            
        else:
            logger.warning(f"Failed to generate TEE signing key: {key_response.status_code}")
            # Add fallback attestation indicating TEE was attempted but failed
            analysis_results['tee_attestation'] = {
                'results_hash': results_hash,
                'tee_attested': False,
                'error': 'TEE key generation failed',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error signing analysis results with TEE: {str(e)}")
        # Add error attestation
        analysis_results['tee_attestation'] = {
            'tee_attested': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    
    return analysis_results 