// src/PropertyModal.js

import React from 'react';
import './App.css'; // We can reuse some styles

const PropertyModal = ({ property, onClose }) => {
  // Prevent clicks inside the modal from closing it
  const handleContentClick = (e) => {
    e.stopPropagation();
  };

  const formatCurrency = (value) => value.toLocaleString('en-US', { style: 'currency', currency: 'USD' });

  return (
    // The overlay is the dark background. Clicking it calls onClose.
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={handleContentClick}>
        <button className="close-button" onClick={onClose}>&times;</button>
        
        <h2>{property.address}</h2>
        
        <div className="details-grid">
          <div className="detail-item">
            <span className="detail-label">Grade</span>
            <span className="detail-value grade-highlight">{property.overall_grade}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Est. ROI</span>
            <span className="detail-value roi-highlight">{property.roi_percentage.toFixed(1)}%</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">List Price</span>
            <span className="detail-value">{formatCurrency(property.list_price)}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Est. Resale Price</span>
            <span className="detail-value">{formatCurrency(property.estimated_resale_price)}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Est. Renovation Cost</span>
            <span className="detail-value">{formatCurrency(property.renovation_cost)}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Other Costs</span>
            <span className="detail-value">{formatCurrency(property.carrying_and_selling_costs)}</span>
          </div>
           <div className="detail-item">
            <span className="detail-label">Expected Profit</span>
            <span className="detail-value">{formatCurrency(property.expected_profit)}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Risk Score</span>
            <span className="detail-value">{property.risk_score} / 10</span>
          </div>
        </div>

        <div className="explanation">
          <h4>Scoring Explanation</h4>
          <p>{property.explanation}</p>
        </div>

      </div>
    </div>
  );
};

export default PropertyModal;