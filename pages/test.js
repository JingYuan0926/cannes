import Chat from './chat';

export default function TestPage() {
  // Mock wallet address for testing
  const mockWalletAddress = "test-wallet-123";

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#121212' }}>
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h1 style={{ color: '#fff', marginBottom: '20px' }}>AI Data Analyst Test</h1>
        <p style={{ color: '#ccc', marginBottom: '40px' }}>
          Test the privacy-first AI Data Analyst chatbot
        </p>
      </div>
      <Chat walletAddress={mockWalletAddress} />
    </div>
  );
} 