import WalletConnect from './WalletConnect';

const Header = () => {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <WalletConnect />
        </div>
      </div>
      
      <style jsx>{`
        .header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          padding: 1rem 0;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 0 1rem;
        }
        
        .header-content {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .logo {
          color: white;
          font-size: 1.5rem;
          font-weight: bold;
          margin: 0;
        }
        
        .wallet-connect {
          display: flex;
          align-items: center;
        }
      `}</style>
    </header>
  );
};

export default Header; 