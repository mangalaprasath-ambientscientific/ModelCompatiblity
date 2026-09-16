import React from 'react'

const ModelDeployment = () => {
    return (
        <div style={{
          fontSize: '20px',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          textAlign: 'center',
          height: 'calc(100vh - 130px)',
          fontFamily: "var(--es-sans)",
          backgroundColor: '#151515',
          color: '#ECECEC',
        }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <h1>Developers are working on it</h1>
            <h3>Will be available soon...</h3>
          </div>
        </div>
      );
}

export default ModelDeployment
